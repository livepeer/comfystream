"""One-shot ComfyUI client for fal batch jobs.

Does not use convert_prompt, tensor_cache, or the streaming runner loop.
Used only inside ProcessPoolExecutor workers.
"""

from __future__ import annotations

import logging
from typing import Any

from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient

from comfystream.capabilities.prompt import build_fal_prompt, extract_submit_collect
from comfystream.capabilities.receipts import error_payload, success_receipt

logger = logging.getLogger(__name__)

_BATCH_BLACKLIST = [
    "ComfyUI-Manager",
    "ComfyUI_TensorRT",
    "ComfyUI-Depth-Anything-Tensorrt",
    "ComfyUI-StreamDiffusion",
    "ComfyUI-FasterLivePortrait",
]


class BatchComfyStreamClient:
    def __init__(
        self,
        max_workers: int = 1,
        **kwargs,
    ):
        kwargs.setdefault("disable_cuda_malloc", True)
        kwargs.setdefault("gpu_only", False)
        kwargs.setdefault("preview_method", "none")
        kwargs.setdefault("blacklist_custom_nodes", list(_BATCH_BLACKLIST))
        config = Configuration(**kwargs)
        self.comfy_client = EmbeddedComfyClient(config, max_workers=max_workers)
        self._started = False

    async def ensure_started(self) -> None:
        if self._started:
            return
        await self.comfy_client.__aenter__()
        self._started = True

    async def queue_once(self, prompt: dict[str, Any]) -> Any:
        await self.ensure_started()
        return await self.comfy_client.queue_prompt(prompt)

    async def execute(self, job: dict[str, Any]) -> dict[str, Any]:
        endpoint_id = job["endpoint_id"]
        schema_sha256 = job["schema_sha256"]
        arguments_json = job["arguments_json"]
        prompt = build_fal_prompt(endpoint_id, arguments_json)
        try:
            outputs = await self.queue_once(prompt)
            request_id, output = extract_submit_collect(outputs)
            return success_receipt(
                endpoint_id=endpoint_id,
                request_id=request_id,
                schema_sha256=schema_sha256,
                output=output,
            )
        except Exception as exc:
            logger.exception("batch fal job failed endpoint=%s", endpoint_id)
            stage = _stage_from_error(exc)
            return error_payload(
                endpoint_id=endpoint_id,
                schema_sha256=schema_sha256,
                stage=stage,
                status=_status_from_stage(stage),
                message=str(exc) or type(exc).__name__,
            )

    async def cleanup(self) -> None:
        if not self._started:
            return
        try:
            await self.comfy_client.__aexit__(None, None, None)
        except Exception:
            logger.debug("batch client cleanup failed", exc_info=True)
        self._started = False


def _stage_from_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__} {exc}".lower()
    if "timeout" in text:
        return "timeout"
    if "status" in text:
        return "status"
    if "result" in text or "collect" in text:
        return "result"
    return "submit"


def _status_from_stage(stage: str) -> int:
    if stage == "timeout":
        return 504
    return 502
