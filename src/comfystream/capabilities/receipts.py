"""Receipt envelope for fal single-shot Live Runner calls."""

from __future__ import annotations

from typing import Any

BILLABLE_UNITS_SOURCE = "x-fal-billable-units"
PROVIDER_REQUEST_ID_HEADER = "Livepeer-Provider-Request-Id"


def success_receipt(
    *,
    endpoint_id: str,
    request_id: str,
    schema_sha256: str,
    output: dict[str, Any],
    billable_units: float | int | None = None,
) -> dict[str, Any]:
    return {
        "billable_units": billable_units,
        "billable_units_source": BILLABLE_UNITS_SOURCE,
        "endpoint_id": endpoint_id,
        "output": output,
        "request_id": request_id,
        "schema_sha256": schema_sha256,
    }


def error_payload(
    *,
    endpoint_id: str,
    schema_sha256: str,
    stage: str,
    status: int,
    message: str,
    request_id: str | None = None,
    upstream_body: Any | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"message": message, "stage": stage, "status": status}
    if upstream_body is not None:
        error["upstream_body"] = upstream_body
    payload: dict[str, Any] = {
        "endpoint_id": endpoint_id,
        "error": error,
        "schema_sha256": schema_sha256,
    }
    if request_id:
        payload["request_id"] = request_id
    return payload


def response_headers(request_id: str | None = None) -> dict[str, str]:
    if not request_id:
        return {}
    return {PROVIDER_REQUEST_ID_HEADER: request_id}
