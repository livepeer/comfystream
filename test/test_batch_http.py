import asyncio
import importlib.util
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from comfystream.batch_pipeline import BatchPipeline
from comfystream.capabilities.catalog import load_catalog
from comfystream.capabilities.receipts import response_headers, success_receipt
from comfystream.job_pool import JobPool

CATALOG = Path(__file__).resolve().parents[1] / "configs" / "fal"
SERVER_DIR = Path(__file__).resolve().parents[1] / "server"


def _fake_execute(job: dict) -> dict:
    return success_receipt(
        endpoint_id=job["endpoint_id"],
        request_id="req-test",
        schema_sha256=job["schema_sha256"],
        output={"images": [{"url": "https://example/img.jpeg"}]},
    )


def _subset_catalog():
    catalog = load_catalog(CATALOG, overlay={})
    return {"flux-schnell": catalog["flux-schnell"]}


def _make_app(pipeline: BatchPipeline) -> web.Application:
    async def health(request: web.Request) -> web.Response:
        item = pipeline.get(request.match_info["capability"])
        if item is None:
            raise web.HTTPNotFound(text="unknown capability")
        return web.json_response(item.health_payload())

    async def schema(request: web.Request) -> web.Response:
        item = pipeline.get(request.match_info["capability"])
        if item is None:
            raise web.HTTPNotFound(text="unknown capability")
        return web.Response(body=item.schema_bytes, content_type="application/json")

    async def execute(request: web.Request) -> web.Response:
        capability = request.match_info["capability"]
        body = await request.read()
        status, payload = await pipeline.execute(capability, body)
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        return web.json_response(
            payload,
            status=status,
            headers=response_headers(request_id if isinstance(request_id, str) else None),
        )

    app = web.Application(client_max_size=pipeline.max_request_bytes)
    app.router.add_get("/fal/{capability}/health", health)
    app.router.add_get("/fal/{capability}/schema", schema)
    app.router.add_post("/fal/{capability}", execute)
    app.router.add_post("/fal/{capability}/", execute)
    return app


def test_batch_http_health_schema_execute_and_unknown():
    catalog = _subset_catalog()
    pipeline = BatchPipeline(
        catalog=catalog,
        pool=JobPool(max_workers=1, execute_fn=_fake_execute),
        orchestrator="",
    )

    async def _run():
        app = _make_app(pipeline)
        async with TestServer(app) as server:
            async with TestClient(server) as client:
                missing = await client.get("/fal/not-a-cap/health")
                assert missing.status == 404

                health = await client.get("/fal/flux-schnell/health")
                assert health.status == 200
                body = await health.json()
                assert body["endpoint_id"] == "fal-ai/flux/schnell"

                schema = await client.get("/fal/flux-schnell/schema")
                assert schema.status == 200
                document = await schema.json()
                assert document["endpoint_id"] == "fal-ai/flux/schnell"

                executed = await client.post(
                    "/fal/flux-schnell",
                    data=json.dumps({"prompt": "a lighthouse"}),
                    headers={"Authorization": "Key should-be-dropped"},
                )
                assert executed.status == 200
                receipt = await executed.json()
                assert receipt["request_id"] == "req-test"
                assert receipt["endpoint_id"] == "fal-ai/flux/schnell"
                assert receipt["billable_units"] is None
                assert receipt["output"]["images"][0]["url"].startswith("https://")
                assert executed.headers.get("Livepeer-Provider-Request-Id") == "req-test"

                bad_json = await client.post("/fal/flux-schnell", data=b"not-json")
                assert bad_json.status == 400

                unknown = await client.post("/fal/missing", data=b"{}")
                assert unknown.status == 404

    asyncio.run(_run())


def test_batch_pipeline_rejects_oversized_body():
    catalog = _subset_catalog()
    catalog["flux-schnell"] = replace(catalog["flux-schnell"], max_request_bytes=8)
    pipeline = BatchPipeline(
        catalog=catalog,
        pool=JobPool(max_workers=1, execute_fn=_fake_execute),
        orchestrator="",
    )

    async def _run():
        status, payload = await pipeline.execute("flux-schnell", b'{"prompt":"too-big"}')
        assert status == 413
        assert payload["error"]["stage"] == "submit"

    asyncio.run(_run())


def test_batch_pipeline_saturated_returns_503():
    def blocking(job: dict) -> dict:
        time.sleep(0.3)
        return _fake_execute(job)

    pipeline = BatchPipeline(
        catalog=_subset_catalog(),
        pool=JobPool(max_workers=1, overflow=1, execute_fn=blocking),
        orchestrator="",
    )

    async def _run():
        first = asyncio.create_task(pipeline.execute("flux-schnell", b'{"prompt":"a"}'))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(pipeline.execute("flux-schnell", b'{"prompt":"b"}'))
        await asyncio.sleep(0.05)
        status, payload = await pipeline.execute("flux-schnell", b'{"prompt":"c"}')
        assert status == 503
        assert payload["error"]["stage"] == "submit"
        await first
        await second

    asyncio.run(_run())


def test_batch_pipeline_registers_single_shot_execute_url():
    captured = []

    async def fake_register(_orch, **kwargs):
        captured.append(kwargs)

        class _Reg:
            runner_id = "runner-1"

            async def close(self):
                return None

        return _Reg()

    item = _subset_catalog()["flux-schnell"]
    pipeline = BatchPipeline(
        catalog={"flux-schnell": item},
        pool=JobPool(max_workers=1, execute_fn=_fake_execute),
        orchestrator="https://orch.example",
        runner_base_url="http://127.0.0.1:8991",
        register_runner_fn=fake_register,
    )

    async def _run():
        await pipeline.start()
        await pipeline.close()

    asyncio.run(_run())
    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs["app"] == "comfystream/fal-flux-schnell"
    assert kwargs["mode"] == "single-shot"
    assert kwargs["unit"] == "fixed"
    assert kwargs["capacity"] == 4
    assert kwargs["runner_url"] == "http://127.0.0.1:8991/fal/flux-schnell"
    assert kwargs["auto_detect_gpu"] is True
    assert len(kwargs["metadata"].encode()) <= 1024


def test_smoke_client_posts_advertised_url_without_session_endpoint():
    spec = importlib.util.spec_from_file_location(
        "live_runner_batch_client",
        SERVER_DIR / "live_runner_batch_client.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    calls = []

    async def fake_call_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data={"request_id": "req-1", "output": {"ok": True}})

    async def _run():
        advertised = "http://127.0.0.1:8991/fal/flux-schnell"
        receipt = await module.execute_single_shot(
            runner_url=advertised,
            arguments={"prompt": "a lighthouse"},
            timeout=30,
            call_runner_fn=fake_call_runner,
        )
        assert receipt["request_id"] == "req-1"
        assert calls[0]["runner_url"] == advertised
        assert calls[0]["payload"] == {"prompt": "a lighthouse"}
        assert module._runner_url(advertised, "./schema") == advertised + "/schema"

    asyncio.run(_run())
    src = (SERVER_DIR / "live_runner_batch_client.py").read_text(encoding="utf-8")
    assert "reserve" not in src.lower()
    assert "stop_runner" not in src
    assert "call_runner" in src
    assert "runner_selector" in src
