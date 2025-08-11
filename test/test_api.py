import pytest
from server.api import StreamStartRequest, ComfyUIParams


@pytest.fixture
def valid_stream_start_data():
    """Basic valid stream start request data."""
    return {
        "subscribe_url": "https://example.com/subscribe",
        "publish_url": "https://example.com/publish", 
        "data_url": "https://example.com/data",
        "gateway_request_id": "test-request-123",
        "manifest_id": "test-manifest",
        "model_id": "comfystream"
    }


@pytest.fixture
def stream_start_data_with_dimensions():
    """Stream start request with width and height parameters."""
    return {
        "subscribe_url": "https://example.com/subscribe",
        "publish_url": "https://example.com/publish",
        "data_url": "https://example.com/data", 
        "gateway_request_id": "test-request-123",
        "manifest_id": "test-manifest",
        "model_id": "comfystream",
        "params": {
            "width": 1280,
            "height": 720,
            "prompts": ""
        }
    }


@pytest.fixture 
def stream_start_data_with_string_dimensions():
    """Stream start request with string width and height parameters."""
    return {
        "subscribe_url": "https://example.com/subscribe",
        "publish_url": "https://example.com/publish",
        "data_url": "https://example.com/data",
        "gateway_request_id": "test-request-123", 
        "manifest_id": "test-manifest",
        "model_id": "comfystream",
        "params": {
            "width": "1920",
            "height": "1080", 
            "prompts": ""
        }
    }


def test_stream_start_request_no_params(valid_stream_start_data):
    """Test StreamStartRequest validation with no params."""
    request = StreamStartRequest(**valid_stream_start_data)
    
    assert request.params is None
    
    # Should return default ComfyUIParams
    comfy_params = request.get_comfy_params()
    assert comfy_params.width == 512  # Default width
    assert comfy_params.height == 512  # Default height
    assert isinstance(comfy_params.prompts, list)


def test_stream_start_request_with_dimensions(stream_start_data_with_dimensions):
    """Test StreamStartRequest validation with width/height parameters."""
    request = StreamStartRequest(**stream_start_data_with_dimensions)
    
    # Params should be preserved and not None
    assert request.params is not None
    assert request.params["width"] == 1280
    assert request.params["height"] == 720
    assert isinstance(request.params["width"], int)
    assert isinstance(request.params["height"], int)
    
    # ComfyUI params should extract the correct dimensions
    comfy_params = request.get_comfy_params()
    assert comfy_params.width == 1280
    assert comfy_params.height == 720
    assert isinstance(comfy_params.prompts, list)


def test_stream_start_request_with_string_dimensions(stream_start_data_with_string_dimensions):
    """Test StreamStartRequest validation with string width/height parameters."""
    request = StreamStartRequest(**stream_start_data_with_string_dimensions)
    
    # String dimensions should be converted to integers
    assert request.params is not None
    assert request.params["width"] == 1920
    assert request.params["height"] == 1080
    assert isinstance(request.params["width"], int)
    assert isinstance(request.params["height"], int)
    
    # ComfyUI params should extract the correct dimensions
    comfy_params = request.get_comfy_params()
    assert comfy_params.width == 1920
    assert comfy_params.height == 1080


def test_stream_start_request_invalid_dimensions():
    """Test StreamStartRequest validation with invalid dimensions."""
    invalid_data = {
        "subscribe_url": "https://example.com/subscribe",
        "publish_url": "https://example.com/publish",
        "data_url": "https://example.com/data",
        "gateway_request_id": "test-request-123",
        "manifest_id": "test-manifest", 
        "model_id": "comfystream",
        "params": {
            "width": "invalid",
            "height": "720"
        }
    }
    
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StreamStartRequest(**invalid_data)


def test_stream_start_request_missing_one_dimension():
    """Test StreamStartRequest validation with only one dimension provided."""
    invalid_data = {
        "subscribe_url": "https://example.com/subscribe", 
        "publish_url": "https://example.com/publish",
        "data_url": "https://example.com/data",
        "gateway_request_id": "test-request-123",
        "manifest_id": "test-manifest",
        "model_id": "comfystream", 
        "params": {
            "width": 1280
            # Missing height
        }
    }
    
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StreamStartRequest(**invalid_data)


def test_stream_start_request_negative_dimensions():
    """Test StreamStartRequest validation with negative dimensions."""
    invalid_data = {
        "subscribe_url": "https://example.com/subscribe",
        "publish_url": "https://example.com/publish", 
        "data_url": "https://example.com/data",
        "gateway_request_id": "test-request-123",
        "manifest_id": "test-manifest",
        "model_id": "comfystream",
        "params": {
            "width": -1280,
            "height": 720
        }
    }
    
    # The actual error comes from Pydantic ValidationError, not direct ValueError
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StreamStartRequest(**invalid_data)


def test_comfy_ui_params_default_values():
    """Test ComfyUIParams default values."""
    params = ComfyUIParams()
    
    assert params.width == 512
    assert params.height == 512
    assert isinstance(params.prompts, list)
    assert len(params.prompts) == 1  # Should have default workflow


def test_comfy_ui_params_custom_dimensions():
    """Test ComfyUIParams with custom dimensions."""
    params = ComfyUIParams(width=1920, height=1080)
    
    assert params.width == 1920
    assert params.height == 1080


def test_comfy_ui_params_string_dimensions():
    """Test ComfyUIParams with string dimensions (should be converted).""" 
    params = ComfyUIParams(width="640", height="480")
    
    assert params.width == 640
    assert params.height == 480
    assert isinstance(params.width, int)
    assert isinstance(params.height, int)


def test_stream_start_request_regression_bug_fix():
    """
    Regression test for the specific bug that was fixed.
    
    Previously, width and height parameters would be lost during validation,
    causing the stream to default to 512x512 instead of the requested resolution.
    This test ensures that parameters are preserved through validation.
    """
    # This is the exact data structure from the logs that was failing
    log_data = {
        "control_url": "https://192.168.10.61:9995/ai/trickle/af620840-control",
        "data_url": "https://192.168.10.61:9995/ai/trickle/af620840-data",
        "events_url": "https://192.168.10.61:9995/ai/trickle/af620840-events",
        "gateway_request_id": "eb913f07",
        "manifest_id": "af620840",
        "model_id": "comfystream",
        "params": {
            "height": 720,
            "prompts": "",
            "width": 1280
        },
        "publish_url": "https://192.168.10.61:9995/ai/trickle/af620840-out",
        "stream_id": "stream-oiks8t-me6ksurl",
        "subscribe_url": "https://192.168.10.61:9995/ai/trickle/af620840"
    }
    
    # Validate the request
    request = StreamStartRequest(**log_data)
    
    # The bug was that request.params would be None after validation
    assert request.params is not None, "Parameters should not be None after validation"
    assert request.params["width"] == 1280, "Width parameter should be preserved"
    assert request.params["height"] == 720, "Height parameter should be preserved"
    
    # Ensure ComfyUIParams extraction works correctly
    comfy_params = request.get_comfy_params()
    assert comfy_params.width == 1280, "ComfyUI params should extract correct width"
    assert comfy_params.height == 720, "ComfyUI params should extract correct height"
    
    # Verify that dimensions are integers (not defaulted to 512x512)
    assert comfy_params.width != 512, "Width should not default to 512"
    assert comfy_params.height != 512, "Height should not default to 512"
