"""Pinned FalSubmit + FalCollect graphs for batch jobs.

Callers supply provider-native JSON. The endpoint_id is pinned by the
capability catalog and never taken from the request.
"""

from __future__ import annotations

import json
from typing import Any


FAL_SUBMIT_NODE = "FalSubmit_fal"
FAL_COLLECT_NODE = "FalCollect_fal"
# FalCollect returns data but is not an OUTPUT_NODE. ComfyUI's validate_prompt
# rejects that graph with prompt_no_outputs, and queue_prompt only returns UI
# data from output nodes. SaveFalResult is the text output type that publishes
# request_id plus result_json. Image/video/audio sinks are not attached:
# those FalCollect slots are None when the result has no matching media.
TEXT_OUTPUT_NODE = "SaveFalResult"
FAL_REQUEST_ID_SLOT = 1
FAL_RESULT_JSON_SLOT = 3


def encode_arguments(body: bytes) -> str:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("request body is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    sanitized = {
        key: item
        for key, item in value.items()
        if not (isinstance(key, str) and key.casefold() == "authorization")
    }
    return json.dumps(sanitized, separators=(",", ":"), ensure_ascii=False)


def build_fal_prompt(endpoint_id: str, arguments_json: str) -> dict[str, Any]:
    if not isinstance(endpoint_id, str) or not endpoint_id.strip():
        raise ValueError("endpoint_id must be a non-empty string")
    if not isinstance(arguments_json, str):
        raise ValueError("arguments_json must be a string")
    return {
        "1": {
            "class_type": FAL_SUBMIT_NODE,
            "inputs": {
                "endpoint_id": endpoint_id,
                "arguments_json": arguments_json,
                "seed": -1,
            },
        },
        "2": {
            "class_type": FAL_COLLECT_NODE,
            "inputs": {"handle": ["1", 0]},
        },
        "3": {
            "class_type": TEXT_OUTPUT_NODE,
            "inputs": {
                "request_id": ["1", FAL_REQUEST_ID_SLOT],
                "result_json": ["2", FAL_RESULT_JSON_SLOT],
            },
        },
    }


def _unwrap(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return _unwrap(value[0])
    return value


def _node_outputs(outputs: Any, node_id: str) -> dict[str, Any]:
    if not isinstance(outputs, dict):
        return {}
    node = outputs.get(node_id)
    if node is None:
        try:
            node = outputs.get(int(node_id))
        except (TypeError, ValueError):
            node = None
    if isinstance(node, dict) and "inputs" not in node and "class_type" not in node:
        inner = node.get("outputs") or node.get("output")
        if isinstance(inner, dict):
            return inner
        return node
    return node if isinstance(node, dict) else {}


def extract_submit_collect(outputs: Any) -> tuple[str, dict[str, Any]]:
    """Pull request_id and provider JSON from a FalSubmit/FalCollect prompt result."""

    submit = _node_outputs(outputs, "1")
    collect = _node_outputs(outputs, "2")
    sink = _node_outputs(outputs, "3")

    request_id = _unwrap(submit.get("request_id")) or _unwrap(sink.get("request_id"))
    handle = _unwrap(submit.get("handle"))
    if not request_id and isinstance(handle, dict):
        request_id = handle.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("FalSubmit did not return a request_id")

    result_json = _unwrap(collect.get("result_json")) or _unwrap(sink.get("result_json"))
    if isinstance(result_json, str):
        try:
            output = json.loads(result_json)
        except json.JSONDecodeError as error:
            raise ValueError("FalCollect result_json was not valid JSON") from error
    elif isinstance(result_json, dict):
        output = result_json
    else:
        raise ValueError("FalCollect did not return result_json")
    if not isinstance(output, dict):
        raise ValueError("FalCollect result_json must be a JSON object")
    return request_id, output
