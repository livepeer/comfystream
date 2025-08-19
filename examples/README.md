# ComfyStream Examples

This directory contains example scripts demonstrating various ComfyStream features.

## Translation Example

### `translation_example.py`

Demonstrates how to use the VLLM translation functionality when running ComfyStream with the sidecar container.

**Prerequisites:**
- ComfyStream running with VLLM sidecar using Docker Compose
- Python with `aiohttp` installed

**Usage:**
```bash
# First, start ComfyStream with VLLM support
docker-compose -f docker/docker-compose.yml up -d

# Then run the example
python examples/translation_example.py
```

**Features demonstrated:**
- Health check for translation service
- Single text translation
- Batch text translation
- Error handling

The example will test translation endpoints and display results, showing how to integrate translation capabilities into your ComfyStream applications.