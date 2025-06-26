# ComfyStream Docker

This folder contains the Docker files that can be used to run ComfyStream in a containerized fashion or to work on the codebase within a dev container. This README contains the general usage instructions while the [Devcontainer Readme](../.devcontainer/README.md) contains instructions on how to use Comfystream inside a dev container and get quickly started with your development journey.

## Containers

- [Dockerfile](Dockerfile) - The main Dockerfile that can be used to run ComfyStream in a containerized fashion.
- [Dockerfile.base](Dockerfile.base) - The base Dockerfile that can be used to build the base image for ComfyStream.
- Main Docker files for ComfyStream containerization
- BYOC files are located in the top-level `byoc/` directory

## Pre-requisites

- [Docker](https://docs.docker.com/get-docker/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Usage

### Build the Base Image

To build the base image, run the following command:

```bash
docker build -t livepeer/comfyui-base:runner -f docker/Dockerfile.base .
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

## BYOC (Bring Your Own Compute) Setup

For Livepeer BYOC integration, use the docker-compose setup:

```bash
# Start the complete BYOC environment
cd byoc
docker-compose up --build
```

This provides:
- **Livepeer Orchestrator** (port 8936) - Manages compute resources
- **Livepeer Gateway** (port 8937) - HTTP ingest endpoint  
- **ComfyStream Server** (port 8889) - AI video processing with trickle streaming
- **Capability Registration** - Automatically registers ComfyStream with the orchestrator

For detailed BYOC documentation, see [../byoc/BUILD.md](../byoc/BUILD.md).
