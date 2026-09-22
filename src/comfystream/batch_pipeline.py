"""Batch analogue of Pipeline: catalog, pricing, registration, job execution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from comfystream.capabilities.catalog import FalCapability, load_catalog
from comfystream.capabilities.prompt import encode_arguments
from comfystream.capabilities.receipts import error_payload
from comfystream.job_pool import JobPool, PoolSaturatedError
from comfystream.modalities import CapabilityModality, detect_capability_modality

logger = logging.getLogger(__name__)


class BatchPipeline:
    """Owns fal capability registration and isolated batch execution."""

    def __init__(
        self,
        *,
        catalog: Mapping[str, FalCapability] | None = None,
        catalog_root: str | Path | None = None,
        pool: JobPool | None = None,
        orchestrator: str = "",
        orch_secret: str = "",
        runner_base_url: str = "",
        fal_key: str = "",
        batch_workers: int = 8,
        workspace: str = "",
        client_kwargs: dict[str, Any] | None = None,
        register_runner_fn: Any = None,
    ):
        self.catalog = dict(catalog) if catalog is not None else load_catalog(catalog_root)
        self._orchestrator = orchestrator
        self._orch_secret = orch_secret
        self._runner_base_url = runner_base_url.rstrip("/")
        self._register_runner_fn = register_runner_fn
        self._registrations: list[Any] = []
        self._modalities: dict[str, CapabilityModality] = {
            capability: detect_capability_modality(item.schema_document, item.endpoint_id)
            for capability, item in self.catalog.items()
        }
        self.pool = pool or JobPool(
            max_workers=batch_workers,
            fal_key=fal_key,
            client_kwargs=client_kwargs
            or {
                "cwd": workspace,
                "disable_cuda_malloc": True,
                "gpu_only": False,
                "preview_method": "none",
                "blacklist_custom_nodes": [
                    "ComfyUI-Manager",
                    "ComfyUI_TensorRT",
                    "ComfyUI-Depth-Anything-Tensorrt",
                    "ComfyUI-StreamDiffusion",
                    "ComfyUI-FasterLivePortrait",
                ],
            },
        )

    @property
    def max_request_bytes(self) -> int:
        if not self.catalog:
            return 1024 * 1024
        return max(item.max_request_bytes for item in self.catalog.values())

    def get(self, capability: str) -> FalCapability | None:
        return self.catalog.get(capability)

    def get_capability_modality(self, capability: str) -> CapabilityModality | None:
        return self._modalities.get(capability)

    def execute_url(self, capability: str) -> str:
        return f"{self._runner_base_url}/fal/{capability}"

    async def start(self) -> None:
        await self.pool.start()
        if not self._orchestrator:
            logger.info("batch catalog loaded without orchestrator registration")
            return
        register_runner = self._register_runner_fn
        if register_runner is None:
            from livepeer_gateway.live_runner import register_runner
        try:
            for item in self.catalog.values():
                registration = await register_runner(
                    self._orchestrator,
                    secret=self._orch_secret,
                    runner_url=self.execute_url(item.capability),
                    app=item.app_id,
                    mode="single-shot",
                    capacity=item.capacity,
                    price=item.price,
                    currency=item.currency,
                    unit=item.unit,
                    metadata=item.metadata,
                    label=item.capability,
                    version="livepeer.fal.runner-route.v1",
                    auto_detect_gpu=True,
                )
                self._registrations.append(registration)
                logger.info(
                    "registered batch app=%s runner_id=%s price=%s capacity=%s",
                    item.app_id,
                    registration.runner_id,
                    item.price,
                    item.capacity,
                )
        except Exception:
            logger.exception("batch capability registration failed; closing registrations")
            await self.close()
            raise

    async def close(self) -> None:
        for registration in self._registrations:
            try:
                await registration.close()
            except Exception:
                logger.debug("failed to close batch registration", exc_info=True)
        self._registrations.clear()
        await self.pool.close()

    async def execute(self, capability: str, body: bytes) -> tuple[int, dict[str, Any]]:
        item = self.catalog.get(capability)
        if item is None:
            return 404, {
                "error": {
                    "message": f"unknown capability {capability!r}",
                    "stage": "submit",
                    "status": 404,
                }
            }
        if len(body) > item.max_request_bytes:
            return 413, error_payload(
                endpoint_id=item.endpoint_id,
                schema_sha256=item.schema_sha256,
                stage="submit",
                status=413,
                message="request body exceeds endpoint max_request_bytes",
            )
        try:
            arguments_json = encode_arguments(body)
        except ValueError as error:
            return 400, error_payload(
                endpoint_id=item.endpoint_id,
                schema_sha256=item.schema_sha256,
                stage="submit",
                status=400,
                message=str(error),
            )
        job = {
            "capability": item.capability,
            "endpoint_id": item.endpoint_id,
            "schema_sha256": item.schema_sha256,
            "arguments_json": arguments_json,
        }
        try:
            result = await self.pool.submit(job, timeout=item.deadline_seconds)
        except PoolSaturatedError:
            return 503, error_payload(
                endpoint_id=item.endpoint_id,
                schema_sha256=item.schema_sha256,
                stage="submit",
                status=503,
                message="batch job pool saturated",
            )
        if "error" in result:
            status = int(result["error"].get("status") or 502)
            return status, result
        return 200, result
