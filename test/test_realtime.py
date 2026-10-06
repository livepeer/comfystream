import asyncio
import json
import os
import site
import subprocess
import sys
import textwrap
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from comfystream.realtime import spec as spec_module
from comfystream.realtime.params import ParamError, validate_params
from comfystream.realtime.spec import RealtimeSpecError, load_realtime_specs
from comfystream.realtime.supervisor import RealtimeSupervisor
from comfystream.realtime.worker import SHUTDOWN_HEADER, SHUTDOWN_PATH, PipelineWorker, WorkerConfig

REPO = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "realtime.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def report_env(config: dict) -> str:
    keys = ("CUDA_VISIBLE_DEVICES", "COMFYSTREAM_TEST_ENV")
    own_site = all(path in sys.path for path in site.getsitepackages())
    return "|".join(
        [os.environ.get(key, "") for key in keys]
        + [sys.executable, str(own_site), config["spec"]["name"]]
    )


class FakeBackend:
    width = 8
    height = 8

    def __init__(self, spec):
        self.batch = 1
        self.loads = 0
        self.params = {}
        self.fail = False

    async def load(self):
        self.loads += 1

    def defaults(self):
        return {"prompt": "default prompt", "seed": -1, "input_blend": 0.5}

    def reset(self):
        self.params = {}

    async def idle(self):
        return None

    async def apply(self, params):
        self.params.update(params)

    def describe(self):
        return {"resolution": "8x8"}

    async def process(self, frames):
        if self.fail:
            raise RuntimeError("CUDA error: illegal memory access")
        outs = []
        for frame in frames:
            out = av.VideoFrame.from_ndarray(np.full((8, 8, 3), 255, np.uint8), format="rgb24")
            out.pts, out.time_base = frame.pts, frame.time_base
            outs.append(out)
        return outs


class FakeOutput:
    def __init__(self, url, *, on_frame, max_segments):
        self.url = url
        self.on_frame = on_frame

    def callback_tasks(self):
        return []

    async def close(self):
        pass

    def get_stats(self):
        return {}


class FakePublish:
    instances: list["FakePublish"] = []

    def __init__(self, url, *, config):
        self.frames: list[av.VideoFrame] = []
        FakePublish.instances.append(self)

    async def write_frame(self, frame):
        self.frames.append(frame)

    async def close(self):
        pass

    def get_stats(self):
        return {}

    def white_frames(self) -> int:
        return sum(int(frame.to_ndarray(format="rgb24").min() == 255) for frame in self.frames)


class FakeRegistration:
    runner_id = "runner_test"

    async def create_trickle_channels(self, request, channels):
        return [
            {"name": "in", "url": "https://orch/in"},
            {"name": "out", "url": "https://orch/out"},
        ]

    async def close(self):
        pass


def test_repo_config_parses():
    specs = {spec.name: spec for spec in load_realtime_specs(REPO / "configs" / "realtime.yaml")}
    assert specs["flux-klein"].app == "livepeer-example/flux-klein"
    assert specs["flux-klein"].port == 8720
    assert specs["sd-turbo"].python == "/workspace/venvs/streamdiffusion/bin/python"
    assert specs["sd-turbo"].gpu != specs["flux-klein"].gpu
    assert all(spec.capacity == 1 for spec in specs.values())


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """
            pipelines:
              a: {app: x/a, backend: flux_klein, port: 9000}
              b: {app: x/b, backend: flux_klein, port: 9000}
            """,
            "ports must be unique",
        ),
        (
            """
            pipelines:
              a: {app: x/a, backend: flux_klein, port: 9000, policy: lukewarm}
            """,
            "policy",
        ),
        (
            """
            pipelines:
              a: {app: x/a, backend: nope, port: 9000}
            """,
            "backend",
        ),
        (
            """
            pipelines:
              a: {app: x/a, backend: flux_klein, port: 9000, capacity: 2}
            """,
            "capacity",
        ),
        (
            """
            pipelines:
              a: {app: x/a, backend: flux_klein, port: 9000, gpus: 0}
            """,
            "unknown keys",
        ),
    ],
)
def test_invalid_configs(tmp_path, body, message):
    with pytest.raises(RealtimeSpecError, match=message):
        load_realtime_specs(_write(tmp_path, body))


def test_spawn_pins_gpu_env_and_interpreter(tmp_path):
    venv = tmp_path / "venv"
    venv_python = venv / "bin" / "python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(venv)],
        check=True,
    )
    path = _write(
        tmp_path,
        f"""
        pipelines:
          pinned:
            app: x/pinned
            backend: flux_klein
            port: 9101
            gpu: GPU-test-uuid
            python: {venv_python}
            env: {{COMFYSTREAM_TEST_ENV: from-spec}}
        """,
    )
    (spec,) = load_realtime_specs(path)
    supervisor = RealtimeSupervisor([spec], orchestrator="", orch_secret="", serve_fn=report_env)
    try:
        result = supervisor._spawn(spec).result(timeout=60)
    finally:
        supervisor._discard(spec.name)
    assert result == f"GPU-test-uuid|from-spec|{venv_python}|True|pinned"
    assert "COMFYSTREAM_TEST_ENV" not in os.environ


def _worker(monkeypatch, tmp_path, policy: str, extra: str = "") -> PipelineWorker:
    monkeypatch.setitem(spec_module.BACKENDS, "flux_klein", f"{__name__}:FakeBackend")
    path = _write(
        tmp_path,
        f"""
        pipelines:
          fake:
            app: x/fake
            backend: flux_klein
            port: 9102
            policy: {policy}
            presets:
              neon: {{prompt: neon stage, input_blend: 0.4}}
            {extra}
        """,
    )
    (spec,) = load_realtime_specs(path)
    config = WorkerConfig(
        spec=spec.to_dict(),
        orchestrator="",
        orch_secret="",
        runner_host="http://127.0.0.1",
        bind_host="127.0.0.1",
        shutdown_token="secret-token",
        usage_log=str(tmp_path / "usage.jsonl"),
    )
    worker = PipelineWorker(spec, config, media_output=FakeOutput, media_publish=FakePublish)
    worker.registration = FakeRegistration()
    return worker


def _frame(pts: int) -> SimpleNamespace:
    frame = av.VideoFrame.from_ndarray(np.zeros((8, 8, 3), np.uint8), format="rgb24")
    frame.pts, frame.time_base = pts, Fraction(1, 30)
    return SimpleNamespace(kind="video", frame=frame)


async def _feed(worker: PipelineWorker, *pts: int) -> None:
    for value in pts:
        await worker.session.output.on_frame(_frame(value))
        for _ in range(5):
            await asyncio.sleep(0)


def test_cold_worker_loads_once_and_shutdown_requires_token(monkeypatch, tmp_path):
    async def scenario():
        worker = _worker(monkeypatch, tmp_path, "cold")
        async with TestClient(TestServer(worker.build_app())) as client:
            status = await (await client.get("/status")).json()
            assert status["state"] == "cold"
            assert (await client.get("/health")).status == 200

            await asyncio.gather(worker.ensure_loaded(), worker.ensure_loaded())
            assert worker.backend.loads == 1
            assert (await (await client.get("/status")).json())["state"] == "ready"

            assert (
                await client.post("/update", headers={"Livepeer-Session-Id": "s1"})
            ).status == 404
            assert (
                await client.post(SHUTDOWN_PATH, headers={SHUTDOWN_HEADER: "wrong"})
            ).status == 404
            assert not worker.stopped.done()
            response = await client.post(SHUTDOWN_PATH, headers={SHUTDOWN_HEADER: "secret-token"})
            assert response.status == 200
            assert await worker.stopped == "shutdown"

    asyncio.run(scenario())


def test_failed_load_reports_error(monkeypatch, tmp_path):
    async def scenario():
        worker = _worker(monkeypatch, tmp_path, "cold")

        async def boom():
            raise RuntimeError("out of memory")

        worker.backend.load = boom
        with pytest.raises(RuntimeError):
            await worker.ensure_loaded()
        async with TestClient(TestServer(worker.build_app())) as client:
            assert (await client.get("/health")).status == 503
            status = await (await client.get("/status")).json()
            assert status["compute"] == "unavailable"
            assert "out of memory" in status["error"]
            response = await client.post("/stream", headers={"Livepeer-Session-Id": "s1"})
            assert response.status == 503
            assert (await response.json())["error"]["code"] == "runner_unavailable"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("backend", "params", "code", "field"),
    [
        ("flux_klein", {"prompt": "  "}, "invalid_param", "prompt"),
        ("flux_klein", {"seed": True}, "invalid_param", "seed"),
        ("flux_klein", {"seed": 1.5}, "invalid_param", "seed"),
        ("flux_klein", {"input_blend": 2}, "invalid_param", "input_blend"),
        ("flux_klein", {"negative_prompt": "x"}, "unsupported_param", "negative_prompt"),
        ("comfy_workflow", {"seed": 3}, "unsupported_param", "seed"),
        ("comfy_workflow", {"workflow": {"1": {}}}, "unsupported_param", "workflow"),
    ],
)
def test_params_are_validated_deterministically(backend, params, code, field):
    with pytest.raises(ParamError) as raised:
        validate_params(backend, params)
    assert (raised.value.code, raised.value.field) == (code, field)


def test_params_are_normalized():
    assert validate_params("flux_klein", {"prompt": " neon ", "seed": 7.0, "input_blend": 1}) == {
        "prompt": "neon",
        "seed": 7,
        "input_blend": 1.0,
    }
    assert validate_params("comfy_workflow", {"negative_prompt": ""}) == {"negative_prompt": ""}


def test_repo_config_advertises_capabilities():
    for spec in load_realtime_specs(REPO / "configs" / "realtime.yaml"):
        metadata = json.loads(spec.metadata())
        assert metadata["model"] and metadata["streaming"] is True
        assert metadata["inputs"] == ["video"] and metadata["outputs"] == ["video"]
        assert metadata["presets"] and "pause" in metadata["surfaces"]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("presets: {bad: {seed: nope}}", "preset 'bad'"),
        ("presets: {bad: {workflow: x}}", "not supported"),
        ("fallbacks: {default: red}", "fallback 'default'"),
    ],
)
def test_invalid_presets_and_fallbacks(tmp_path, extra, message):
    body = f"""
    pipelines:
      a: {{app: x/a, backend: flux_klein, port: 9000, {extra}}}
    """
    with pytest.raises(RealtimeSpecError, match=message):
        load_realtime_specs(_write(tmp_path, body))


def test_session_lifecycle_fallback_and_usage(monkeypatch, tmp_path):
    headers = {"Livepeer-Session-Id": "room-12-booking-9"}

    async def scenario():
        worker = _worker(monkeypatch, tmp_path, "warm")
        await worker.ensure_loaded()
        async with TestClient(TestServer(worker.build_app())) as client:
            bad = await client.post(
                "/stream", headers=headers, json={"metadata": {"room_id": "r 12"}}
            )
            assert bad.status == 400
            assert (await bad.json())["error"] == {
                "code": "invalid_metadata",
                "message": "room_id must be 1-128 characters of letters, digits or ._:@-",
                "field": "metadata.room_id",
            }
            unknown = await client.post("/stream", headers=headers, json={"preset": "nope"})
            assert (await unknown.json())["error"]["code"] == "unknown_preset"

            started = await client.post(
                "/stream",
                headers=headers,
                json={
                    "preset": "neon",
                    "seed": 42,
                    "metadata": {
                        "room_id": "r12",
                        "venue_id": "v3",
                        "booking_id": "b9",
                        "environment": "pilot",
                        "lyrics_overlay": True,
                    },
                },
            )
            assert started.status == 200
            body = await started.json()
            assert body["status"] == "active" and body["compute"] == "warm"
            assert body["estimated_startup_s"] == 0.0
            assert body["params"] == {"prompt": "neon stage", "seed": 42, "input_blend": 0.4}
            assert worker.backend.params == body["params"]
            repeat = await client.post("/stream", headers=headers, json={"prompt": "ignored"})
            assert (await repeat.json())["session"] == "room-12-booking-9"
            other = await client.post("/stream", headers={"Livepeer-Session-Id": "other"})
            assert (await other.json())["error"]["code"] == "session_conflict"

            publisher = FakePublish.instances[-1]
            await _feed(worker, 1, 2)
            assert publisher.white_frames() == 2

            paused = await (await client.post("/pause", headers=headers)).json()
            assert paused["status"] == "paused" and paused["gpu_reserved"] is True
            await _feed(worker, 3)
            assert len(publisher.frames) == 3 and publisher.white_frames() == 2
            assert (await (await client.post("/pause", headers=headers)).json())[
                "status"
            ] == "paused"
            resumed = await (await client.post("/resume", headers=headers)).json()
            assert resumed["status"] == "active" and resumed["output"] == "live"

            worker.backend.fail = True
            await _feed(worker, 4)
            session = await (await client.get("/session", headers=headers)).json()
            assert session["output"] == "fallback"
            assert session["fallback_reason"] == "generation_error"
            assert len(publisher.frames) == 4 and publisher.white_frames() == 2
            worker.backend.fail = False
            await _feed(worker, 5, 5)
            session = await (await client.get("/session", headers=headers)).json()
            assert session["output"] == "live" and publisher.white_frames() == 3

            updated = await client.post("/update", headers=headers, json={"prompt": "a nebula"})
            assert (await updated.json())["params"]["prompt"] == "a nebula"
            rejected = await client.post("/update", headers=headers, json={"metadata": {}})
            assert (await rejected.json())["error"]["code"] == "unsupported_param"

            stopped = await (await client.post("/stop", headers=headers)).json()
            assert stopped["status"] == "stopped" and stopped["already_stopped"] is False
            assert (stopped["room_id"], stopped["venue_id"], stopped["environment"]) == (
                "r12",
                "v3",
                "pilot",
            )
            assert stopped["frames_fallback"] == 2 and stopped["errors"] == 1
            assert stopped["fallback_events"][0]["reason"] == "generation_error"
            assert stopped["cost_estimate"]["currency"] == "usd"
            again = await (await client.post("/stop", headers=headers)).json()
            assert again["already_stopped"] is True
            late = await client.post("/resume", headers=headers)
            assert (await late.json())["error"] == {
                "code": "invalid_state",
                "message": "session already stopped",
                "field": None,
                "session_status": "stopped",
            }
            restart = await client.post("/stream", headers=headers)
            assert (await restart.json())["error"]["code"] == "invalid_state"

        records = (tmp_path / "usage.jsonl").read_text().splitlines()
        assert [json.loads(line)["session"] for line in records] == ["room-12-booking-9"]

    asyncio.run(scenario())


def test_cold_start_limit_and_idle_expiry(monkeypatch, tmp_path):
    headers = {"Livepeer-Session-Id": "s1"}

    async def scenario():
        worker = _worker(monkeypatch, tmp_path, "cold", "cold_start_s: 60")
        async with TestClient(TestServer(worker.build_app())) as client:
            status = await (await client.get("/status")).json()
            assert (status["compute"], status["estimated_startup_s"]) == ("cold", 60.0)
            limited = await client.post("/stream", headers=headers, json={"max_startup_s": 35})
            assert limited.status == 503
            error = (await limited.json())["error"]
            assert error["code"] == "startup_exceeds_limit"
            assert error["estimated_startup_s"] == 60.0
            assert worker.backend.loads == 0

            started = await client.post("/stream", headers=headers, json={"idle_timeout_s": 0.5})
            body = await started.json()
            assert body["compute"] == "cold" and worker.backend.loads == 1
            assert (await (await client.get("/status")).json())["compute"] == "warm"
            await asyncio.sleep(1.6)
            session = await (await client.get("/session", headers=headers)).json()
            assert session["status"] == "expired" and session["reason"] == "no_input"
            assert worker.session is None

    asyncio.run(scenario())
