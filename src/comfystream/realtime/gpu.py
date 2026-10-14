"""Resolve the GPU a process is pinned to for live-runner registration."""

from __future__ import annotations

import os
from typing import Any

import pynvml
from livepeer_gateway.live_runner import LiveRunnerGPU


def _pinned_uuid() -> str:
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return first if first.startswith("GPU-") else ""


def pinned_gpu() -> LiveRunnerGPU | None:
    """Describe a UUID-pinned CUDA_VISIBLE_DEVICES GPU.

    The SDK's fallback maps non-numeric CUDA_VISIBLE_DEVICES to NVML index 0,
    which misreports UUID-pinned processes that have no CUDA context yet.
    Returns None for unpinned or index-pinned processes so auto-detect applies.
    """
    uuid = _pinned_uuid()
    if not uuid:
        return None
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        return LiveRunnerGPU(id=uuid, name=str(name), vram_mb=int(total) // (1024 * 1024))
    finally:
        pynvml.nvmlShutdown()


def gpu_usage() -> dict[str, Any] | None:
    """Memory and utilization of the pinned GPU, for soak and leak monitoring."""
    uuid = _pinned_uuid()
    if not uuid:
        return None
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "id": uuid,
            "memory_used_mb": int(memory.used) // (1024 * 1024),
            "memory_total_mb": int(memory.total) // (1024 * 1024),
            "utilization_pct": int(utilization.gpu),
        }
    except pynvml.NVMLError:
        return None
    finally:
        pynvml.nvmlShutdown()
