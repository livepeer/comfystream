import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from comfystream.capabilities.catalog import METADATA_LIMIT_BYTES, load_catalog
from comfystream.capabilities.pricing import derived_price_info, resolve_price_info
from comfystream.capabilities.prompt import build_fal_prompt, encode_arguments, extract_submit_collect
from comfystream.modalities import detect_capability_modality

CATALOG = Path(__file__).resolve().parents[1] / "configs" / "fal"
REPO = Path(__file__).resolve().parents[1]


def test_load_catalog_rewrites_app_ids_and_keeps_schema_hash():
    catalog = load_catalog(CATALOG, overlay={})
    assert len(catalog) == 73
    flux = catalog["flux-schnell"]
    assert flux.app_id == "comfystream/fal-flux-schnell"
    assert flux.endpoint_id == "fal-ai/flux/schnell"
    assert flux.unit == "fixed"
    assert flux.capacity == 4
    assert flux.price == pytest.approx(0.003)
    assert len(flux.metadata.encode()) <= METADATA_LIMIT_BYTES
    schema_path = CATALOG / "contracts" / "flux-schnell" / "schema.json"
    assert hashlib.sha256(schema_path.read_bytes()).hexdigest() == flux.schema_sha256
    meta = json.loads(flux.metadata)
    assert meta["schema_url"] == "./schema"
    assert meta["provider"] == "fal"
    assert meta["transport"] == "queue"
    assert set(meta) == {
        "deadline_seconds",
        "endpoint_id",
        "provider",
        "schema_sha256",
        "schema_url",
        "transport",
    }


def test_sell_price_derivation_and_overrides():
    derived = derived_price_info(
        unit_price=Decimal("0.05"),
        expected_units=Decimal("8"),
        upcharge_bps=1000,
    )
    assert derived["unit"] == "fixed"
    assert derived["price"] == pytest.approx(0.44)

    route_document = {"pricing": {"unit_price": "0.01", "unit": "megapixels"}}
    explicit = resolve_price_info(
        capability="demo",
        route_document=route_document,
        selection={"price_info": {"currency": "usd", "price": 0.25, "unit": "fixed"}},
        default_upcharge_bps=0,
        default_expected_units=Decimal("1"),
    )
    assert explicit["price"] == 0.25
    assert explicit["unit"] == "fixed"

    catalog = load_catalog(
        CATALOG,
        overlay={
            "defaults": {
                "capacity": 2,
                "deadline_seconds": 90,
                "max_request_bytes": 2048,
                "upcharge_bps": 0,
                "expected_units": 1,
            },
            "routes": {
                "flux-schnell": {"upcharge_bps": 10000, "capacity": 8},
            },
        },
    )
    assert catalog["flux-schnell"].price == pytest.approx(0.006)
    assert catalog["flux-schnell"].capacity == 8
    assert catalog["flux-2-pro"].capacity == 2


def test_capability_modalities_from_schema():
    catalog = load_catalog(CATALOG, overlay={})
    flux = detect_capability_modality(
        catalog["flux-schnell"].schema_document, catalog["flux-schnell"].endpoint_id
    )
    assert flux["image"]["output"] is True
    kling = detect_capability_modality(
        catalog["kling-v3-pro-i2v"].schema_document,
        catalog["kling-v3-pro-i2v"].endpoint_id,
    )
    assert kling["video"]["output"] is True
    whisper = detect_capability_modality(
        catalog["whisper-transcribe"].schema_document,
        catalog["whisper-transcribe"].endpoint_id,
    )
    assert whisper["audio"]["input"] is True
    assert whisper["text"]["output"] is True
    ffmpeg = detect_capability_modality(
        catalog["ffmpeg-scale-video"].schema_document,
        catalog["ffmpeg-scale-video"].endpoint_id,
    )
    assert ffmpeg["video"]["input"] is True
    assert ffmpeg["video"]["output"] is True
    mesh = detect_capability_modality(
        catalog["meshy-v7-i3d"].schema_document, catalog["meshy-v7-i3d"].endpoint_id
    )
    assert mesh["mesh"]["output"] is True


def test_batch_prompt_pins_endpoint_and_drops_caller_auth():
    prompt = build_fal_prompt(
        "fal-ai/flux/schnell",
        encode_arguments(b'{"prompt":"a lighthouse","Authorization":"Key leaked"}'),
    )
    assert prompt["1"]["class_type"] == "FalSubmit_fal"
    assert prompt["1"]["inputs"]["endpoint_id"] == "fal-ai/flux/schnell"
    assert prompt["2"]["class_type"] == "FalCollect_fal"
    assert prompt["3"]["class_type"] == "SaveFalResult"
    assert prompt["3"]["inputs"]["request_id"] == ["1", 1]
    assert prompt["3"]["inputs"]["result_json"] == ["2", 3]
    assert set(prompt["1"]["inputs"]) == {"endpoint_id", "arguments_json", "seed"}
    arguments = json.loads(prompt["1"]["inputs"]["arguments_json"])
    assert arguments == {"prompt": "a lighthouse"}
    dumped = json.dumps(prompt)
    assert "Authorization" not in dumped
    assert "Key leaked" not in dumped
    assert "Authorization" not in prompt["1"]["inputs"]
    for variant in (b"authorization", b"Authorization", b"AUTHORIZATION"):
        stripped = json.loads(encode_arguments(b'{"prompt":"ok","' + variant + b'":"secret"}'))
        assert stripped == {"prompt": "ok"}
    kept = json.loads(
        encode_arguments(b'{"prompt":"ok","authorization_scheme":"Key","fal_key":"native"}')
    )
    assert kept == {"prompt": "ok", "authorization_scheme": "Key", "fal_key": "native"}
    nested = json.loads(
        encode_arguments(b'{"prompt":"ok","nested":{"Authorization":"keep"}}')
    )
    assert nested == {"prompt": "ok", "nested": {"Authorization": "keep"}}

    request_id, output = extract_submit_collect(
        {
            "1": {
                "request_id": ["req-1"],
                "handle": [{"endpoint_id": "fal-ai/flux/schnell", "request_id": "req-1"}],
            },
            "2": {"result_json": [json.dumps({"images": [{"url": "https://example/img.jpeg"}]})]},
        }
    )
    assert request_id == "req-1"
    assert output["images"][0]["url"] == "https://example/img.jpeg"

    sink_request_id, sink_output = extract_submit_collect(
        {
            "3": {
                "request_id": ["req-ui"],
                "result_json": [json.dumps({"images": [{"url": "https://example/ui.jpeg"}]})],
            }
        }
    )
    assert sink_request_id == "req-ui"
    assert sink_output["images"][0]["url"] == "https://example/ui.jpeg"


def test_batch_client_skips_convert_prompt_and_tensor_cache():
    client_src = (REPO / "src" / "comfystream" / "batch_client.py").read_text(encoding="utf-8")
    assert "convert_prompt" not in client_src.replace(
        "Does not use convert_prompt, tensor_cache, or the streaming runner loop.",
        "",
    )
    assert "tensor_cache" not in client_src.replace(
        "Does not use convert_prompt, tensor_cache, or the streaming runner loop.",
        "",
    )
    assert "from comfystream.utils import convert_prompt" not in client_src
    assert "from comfystream import tensor_cache" not in client_src
    assert "set_prompts" not in client_src
    assert "_runner_loop" not in client_src
