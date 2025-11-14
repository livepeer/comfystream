# ComfyStream Docker

This folder contains the Docker files that can be used to run ComfyStream in a containerized fashion or to work on the codebase within a dev container. This README contains the general usage instructions while the [Devcontainer Readme](../.devcontainer/README.md) contains instructions on how to use Comfystream inside a dev container and get quickly started with your development journey.

## Containers

- [Dockerfile](Dockerfile) - The main Dockerfile that can be used to run ComfyStream in a containerized fashion.
- [Dockerfile.base](Dockerfile.base) - The base Dockerfile that can be used to build the base image for ComfyStream. Uses precompiled OpenCV CUDA binaries.
- [Dockerfile-cv2.base](Dockerfile-cv2.base) - Alternative base image with OpenCV CUDA support (uses precompiled binaries via pip install).
- [Dockerfile.opencv](Dockerfile.opencv) - Alternative base Dockerfile that builds OpenCV 4.11.0 from source with full CUDA support.

## Pre-requisites

- [Docker](https://docs.docker.com/get-docker/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Local Development (Non-Docker)

If you're installing ComfyStream locally with `pip install -e .`, OpenCV CUDA installation may not trigger automatically due to PEP 517 build isolation. After installation, manually run:

```bash
python src/comfystream/scripts/install_opencv_cuda.py
```

**Important:** The OpenCV CUDA binaries require `numpy<2.0`. The installation script will automatically downgrade NumPy if needed.

Verify the installation:

```bash
python src/comfystream/scripts/verify_opencv_cuda.py
```

Or to skip OpenCV CUDA installation entirely:

```bash
export COMFYSTREAM_SKIP_OPENCV_CUDA=true
pip install -e .
```

## Docker Usage

### Build the Base Image

To build the base image, run the following command:

```bash
docker build -t livepeer/comfyui-base -f docker/Dockerfile.base .
```

**OpenCV CUDA Installation:** The base image installs OpenCV with CUDA support via an explicit call to `install_opencv_cuda.py` after the `pip install -e .` step (line 101). The installation:
- Downloads precompiled OpenCV CUDA binaries (configured in `pyproject.toml`)
- Installs the cv2 package with CUDA acceleration
- Copies necessary libraries to `/usr/lib/x86_64-linux-gnu`

**Note:** The installation script is called explicitly because setuptools hooks may not trigger reliably during editable installs with PEP 517.

### Build with OpenCV from Source (Alternative)

To build OpenCV 4.11.0 from source instead of using precompiled binaries:

```bash
docker build -t livepeer/comfyui-opencv-base -f docker/Dockerfile.opencv .
```

This builds OpenCV with full CUDA, cuDNN, and CUBLAS support optimized for your target GPU architecture.

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
