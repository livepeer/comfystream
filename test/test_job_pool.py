import asyncio
import time

import pytest

from comfystream.capabilities.receipts import success_receipt
from comfystream.job_pool import JobPool, PoolSaturatedError


def _echo(job: dict) -> dict:
    return success_receipt(
        endpoint_id=job["endpoint_id"],
        request_id="req-echo",
        schema_sha256=job["schema_sha256"],
        output={"ok": True, "prompt": job["arguments_json"]},
    )


def _slow(job: dict) -> dict:
    time.sleep(float(job.get("sleep", 0.3)))
    return _echo(job)


def test_job_pool_returns_receipt():
    pool = JobPool(max_workers=1, execute_fn=_echo)

    async def _run():
        return await pool.submit(
            {
                "capability": "flux-schnell",
                "endpoint_id": "fal-ai/flux/schnell",
                "schema_sha256": "abc",
                "arguments_json": '{"prompt":"hi"}',
            },
            timeout=2,
        )

    result = asyncio.run(_run())
    assert result["request_id"] == "req-echo"
    assert result["output"]["ok"] is True


def test_job_pool_timeout_maps_to_stage():
    pool = JobPool(max_workers=1, execute_fn=_slow)

    async def _run():
        return await pool.submit(
            {
                "capability": "flux-schnell",
                "endpoint_id": "fal-ai/flux/schnell",
                "schema_sha256": "abc",
                "arguments_json": "{}",
                "sleep": 1.0,
            },
            timeout=0.05,
        )

    result = asyncio.run(_run())
    assert result["error"]["stage"] == "timeout"
    assert result["error"]["status"] == 504


def test_job_pool_saturation_raises():
    pool = JobPool(max_workers=1, overflow=1, execute_fn=_slow)

    async def _run():
        first = asyncio.create_task(
            pool.submit(
                {
                    "endpoint_id": "fal-ai/flux/schnell",
                    "schema_sha256": "abc",
                    "arguments_json": "{}",
                    "sleep": 0.4,
                },
                timeout=5,
            )
        )
        await asyncio.sleep(0.05)
        second = asyncio.create_task(
            pool.submit(
                {
                    "endpoint_id": "fal-ai/flux/schnell",
                    "schema_sha256": "abc",
                    "arguments_json": "{}",
                    "sleep": 0.4,
                },
                timeout=5,
            )
        )
        await asyncio.sleep(0.05)
        with pytest.raises(PoolSaturatedError):
            await pool.submit(
                {
                    "endpoint_id": "fal-ai/flux/schnell",
                    "schema_sha256": "abc",
                    "arguments_json": "{}",
                    "sleep": 0.1,
                },
                timeout=5,
            )
        await first
        await second

    asyncio.run(_run())
