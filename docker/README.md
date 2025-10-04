# ComfyStream Docker

This folder contains the Docker files that can be used to run ComfyStream in a containerized fashion or to work on the codebase within a dev container. This README contains the general usage instructions while the [Devcontainer Readme](../.devcontainer/README.md) contains instructions on how to use Comfystream inside a dev container and get quickly started with your development journey.

## Containers

- [Dockerfile](Dockerfile) - The main Dockerfile that can be used to run ComfyStream in a containerized fashion.
- [Dockerfile.base](Dockerfile.base) - The base Dockerfile that can be used to build the base image for ComfyStream.

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

### Run in BYOC Mode

To start ComfyStream in BYOC (Bring Your Own Compute) mode, you can use the dedicated BYOC entrypoint:

```bash
docker run --gpus all -e ORCH_SECRET=your-secret-here \
  --entrypoint /workspace/comfystream/docker/entrypoint-byoc.sh \
  livepeer/comfystream
```

#### Using Docker Compose

For easier management, you can use the provided docker-compose configuration:

```bash
# Create a .env file with your ORCH_SECRET
echo "ORCH_SECRET=your-secret-here" > .env

# Start BYOC service
docker-compose -f docker/docker-compose.byoc.yml up
```

#### BYOC Environment Variables

The BYOC mode supports the following environment variables with defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCH_SECRET` | **Required** | Orchestrator secret for authentication |
| `ORCH_URL` | `https://172.17.0.1:9995` | Orchestrator URL |
| `CAPABILITY_NAME` | `comfystream` | Capability name |
| `CAPABILITY_DESCRIPTION` | `ComfyUI streaming processor for BYOC mode` | Capability description |
| `CAPABILITY_URL` | `http://172.17.0.1:8000` | Capability URL |
| `CAPABILITY_PRICE_PER_UNIT` | `0` | Price per unit |
| `CAPABILITY_PRICE_SCALING` | `1` | Price scaling |
| `CAPABILITY_CAPACITY` | `1` | Capability capacity |
| `COMFYUI_WORKSPACE` | `/workspace/ComfyUI` | ComfyUI workspace path |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Log level |
| `COMFYUI_INFERENCE_LOG_LEVEL` | `DEBUG` | ComfyUI inference log level |
| `WIDTH` | `512` | Default width |
| `HEIGHT` | `512` | Default height |

#### Custom BYOC Configuration

```bash
docker run --gpus all \
  -v "$HOME/models/ComfyUI--models:/workspace/ComfyUI/models" \
  -v "$HOME/models/ComfyUI--output:/workspace/ComfyUI/output" \
  -e ORCH_SECRET="orch-secret" \
  -e ORCH_URL="https://your_orchestrator_url:9995" \
  -p 8000:8000 \
  --entrypoint /workspace/comfystream/docker/entrypoint-byoc.sh \
  livepeer/comfystream
```

### Available Options

There are multiple options that can be passed to the ComfyStream entrypoint. To see the list of available options, run the following command:

```bash
docker run --gpus all livepeer/comfystream --help
```

Available modes (for main entrypoint):
- `--server`: Start ComfyUI only
- `--api`: Start ComfyStream API Server only  
- `--ui`: Start ComfyStream UI only
- `--help`: Show help message

For BYOC mode, use the dedicated `entrypoint-byoc.sh` entrypoint script.
