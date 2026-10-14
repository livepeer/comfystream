"""Single-shot registration and isolated execution for batch video tools."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any

from comfystream.job_pool import JobPool, PoolSaturatedError
from comfystream.tools.catalog import TOOLS, VideoTool
from comfystream.tools.execute import run_job, set_queue

logger = logging.getLogger(__name__)


def _video_worker_init(client_kwargs: dict[str, Any]) -> None:
    """Start one Comfy client in this worker. CUDA stays hidden from the stream."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cwd = str(client_kwargs.get("cwd") or "")
    if not cwd:
        set_queue(None)
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        from comfystream.batch_client import BatchComfyStreamClient

        client = BatchComfyStreamClient(**client_kwargs)
        loop.run_until_complete(client.ensure_started())
    except Exception:
        logger.exception("video worker could not start Comfy; file tools still run")
        set_queue(None)
        return

    def _queue(prompt: dict[str, Any]) -> Any:
        return loop.run_until_complete(client.queue_once(prompt))

    set_queue(_queue)


def _video_worker_execute(job: dict[str, Any]) -> dict[str, Any]:
    return run_job(job)


class ToolPipeline:
    """One process pool shared by every video tool."""

    def __init__(
        self,
        *,
        orchestrator: str = "",
        orch_secret: str = "",
        runner_base_url: str = "",
        batch_workers: int = 2,
        workspace: str = "",
        price: float = 0.0,
        pool: JobPool | None = None,
        register_runner_fn: Any = None,
    ):
        self.catalog = dict(TOOLS)
        self._orchestrator = orchestrator
        self._orch_secret = orch_secret
        self._runner_base_url = runner_base_url.rstrip("/")
        self._price = price
        self._register_runner_fn = register_runner_fn
        self._registrations: list[Any] = []
        self.pool = pool or JobPool(
            max_workers=batch_workers,
            initializer=_video_worker_init,
            initargs=(
                {
                    "cwd": workspace,
                    "disable_cuda_malloc": True,
                    "gpu_only": False,
                    "preview_method": "none",
                },
            ),
            pool_worker=_video_worker_execute,
            timeout_message="video tool deadline exceeded",
        )

    def get(self, name: str) -> VideoTool | None:
        return self.catalog.get(name)

    def execute_url(self, name: str) -> str:
        return f"{self._runner_base_url}/tools/{name}"

    async def start(self) -> None:
        await self.pool.start()
        if not self._orchestrator:
            logger.info("video tools loaded without orchestrator registration")
            return
        register_runner = self._register_runner_fn
        if register_runner is None:
            from livepeer_gateway.live_runner import register_runner
        from comfystream.realtime.gpu import pinned_gpu

        gpu = pinned_gpu()
        try:
            for item in self.catalog.values():
                registration = await register_runner(
                    self._orchestrator,
                    secret=self._orch_secret,
                    runner_url=self.execute_url(item.name),
                    app=item.app_id,
                    mode="single-shot",
                    capacity=self.pool.max_workers,
                    price=self._price if self._price else item.price,
                    currency=item.currency,
                    unit=item.unit,
                    metadata=item.metadata(),
                    label=item.name,
                    version="comfystream.video-tool.v1",
                    gpu=gpu,
                    auto_detect_gpu=gpu is not None,
                )
                self._registrations.append(registration)
                logger.info(
                    "registered video tool app=%s runner_id=%s",
                    item.app_id,
                    registration.runner_id,
                )
        except Exception:
            logger.exception("video tool registration failed; closing registrations")
            await self.close()
            raise

    async def close(self) -> None:
        for registration in self._registrations:
            try:
                await registration.close()
            except Exception:
                logger.debug("failed to close video tool registration", exc_info=True)
        self._registrations.clear()
        await self.pool.close()

    async def execute(self, name: str, body: bytes) -> tuple[int, dict[str, Any]]:
        item = self.catalog.get(name)
        if item is None:
            return 404, _error(f"unknown tool {name}", 404)
        if len(body) > item.max_request_bytes:
            return 413, _error("request body is too large", 413)
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return 400, _error("body must be JSON", 400)
        if not isinstance(payload, dict):
            return 400, _error("body must be a JSON object", 400)
        encoded = payload.get("video_base64")
        if not isinstance(encoded, str) or not encoded:
            return 400, _error("video_base64 is required", 400)
        try:
            video = base64.b64decode(encoded, validate=True)
        except Exception:
            return 400, _error("video_base64 is not valid base64", 400)
        params = {key: value for key, value in payload.items() if key != "video_base64"}
        try:
            result = await self.pool.submit(
                {"tool": name, "video": video, "params": params},
                timeout=item.deadline_seconds,
            )
        except PoolSaturatedError:
            return 503, _error("video tool pool is saturated", 503)
        if isinstance(result, dict) and "error" in result:
            status = int(result["error"].get("status") or 500)
            return status, result
        return 200, result


def _error(message: str, status: int) -> dict[str, Any]:
    return {"error": {"message": message, "status": status, "stage": "submit"}}
