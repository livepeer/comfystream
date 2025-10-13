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

There are multiple options that can be passed to the Comfystream server. To see the list of available options, run the following command:

```bash
docker run --gpus all livepeer/comfystream --help
```

## Bring Your Own Workspace (BYOW)

The `Dockerfile.uv` and `entrypoint-byow.sh` support mounting your own ComfyUI workspace, allowing you to persist your custom nodes, models, and configurations across container restarts.

### Usage

To mount your own workspace:

```bash
docker run -it --gpus all -p 8188:8188 \
  -v ~/my-comfyui-workspace:/workspace/ComfyUI \
  comfystream:uv-nodes
```

### How It Works

The entrypoint script automatically detects whether you're using a mounted workspace or the built-in workspace:

1. **Built-in Workspace**: If the container finds a `.comfystream_builtin_workspace` marker file, it uses the pre-built workspace from the Docker image.

2. **Mounted Workspace**: If no marker file is present, the script assumes you've mounted your own workspace and will:
   - Create a Python virtual environment (`.venv`) in your workspace if it doesn't exist
   - Install comfy-cli for workspace management
   - Clone ComfyUI directly into your workspace (if not already present) to avoid nested directory issues
   - Clone ComfyUI-Manager for custom node management
   - Copy the `comfy-lock.yaml` configuration file
   - Restore custom nodes from the lock file using comfy-cli
   - Install comfystream as a custom node

### Key Design Decisions

- **Direct Git Clone**: The script uses `git clone` directly instead of `comfy install` to avoid creating nested `ComfyUI/ComfyUI` directories when the mount point is already `/workspace/ComfyUI`.

- **Skip Requirements**: ComfyUI's `requirements.txt` is intentionally NOT installed. Dependencies are managed via `constraints.txt` and comfystream's requirements to ensure compatibility.

- **Custom Node Management**: Custom nodes are managed through comfy-cli's snapshot restore feature using the `comfy-lock.yaml` file.
