# ComfyStream Docker

This folder contains the Docker files that can be used to run ComfyStream in a containerized fashion or to work on the codebase within a dev container. This README contains the general usage instructions while the [Devcontainer Readme](../.devcontainer/README.md) contains instructions on how to use Comfystream inside a dev container and get quickly started with your development journey.

## Containers

- [Dockerfile](Dockerfile) - The main Dockerfile that can be used to run ComfyStream in a containerized fashion.
- [Dockerfile.base](Dockerfile.base) - The base Dockerfile that can be used to build the base image for ComfyStream.
- [Dockerfile.vllm](Dockerfile.vllm) - VLLM container for translation services (sidecar pattern).
- [docker-compose.yml](docker-compose.yml) - Docker Compose configuration for running ComfyStream with VLLM sidecar.

## Pre-requisites

- [Docker](https://docs.docker.com/get-docker/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Usage

### Build the Base Image

To build the base image, run the following command:

```bash
docker build -t livepeer/comfyui-base -f docker/Dockerfile.base .
```

### Build the Main Image

To build the main image, run the following command:

```bash
docker build -t livepeer/comfystream -f docker/Dockerfile .
```

### Run the Container

To start the container in interactive mode, run the following command:

```bash
docker run -it --gpus all livepeer/comfystream
```

To start the Comfystream server, run the following command:

```bash
docker run --gpus all livepeer/comfystream --server
```

There are multiple options that can be passed to the Comfystream server. To see the list of available options, run the following command:

```bash
docker run --gpus all livepeer/comfystream --help
```

### Run with VLLM Translation Support (Docker Compose)

To run ComfyStream with VLLM translation support using Docker Compose:

```bash
# Build and start both ComfyStream and VLLM containers
docker-compose -f docker/docker-compose.yml up --build

# Or run in detached mode
docker-compose -f docker/docker-compose.yml up -d --build
```

This will start:
- ComfyStream container on ports 8188 (ComfyUI), 8889 (API), 3000 (UI)
- VLLM container on port 8000 for translation services

### Translation API Usage

Once running with VLLM support, you can use the translation endpoints:

```bash
# Single text translation
curl -X POST http://localhost:8889/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source_lang": "en", "target_lang": "es"}'

# Batch translation
curl -X POST http://localhost:8889/translate/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "World"], "source_lang": "en", "target_lang": "es"}'

# Check translation service health
curl http://localhost:8889/translate/health
```

### Environment Variables

- `VLLM_ENDPOINT`: VLLM service endpoint (default: http://localhost:8000)
- `VLLM_MODEL`: Model to use for translation (default: microsoft/DialoGPT-medium)
- `VLLM_HOST`: VLLM service host (default: 0.0.0.0)
- `VLLM_PORT`: VLLM service port (default: 8000)
