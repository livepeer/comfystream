# ComfyStream

ComfyStream is a package for running img2img [ComfyUI](https://www.comfy.org/) workflows on real-time video streams. It integrates with PyTrickle to provide high-performance streaming with HTTP APIs for remote control.

This repo includes:
- **Real-time video processing** with ComfyUI workflows
- **HTTP streaming server** with REST API endpoints  
- **WebRTC server and UI** for webcam streaming
- **PyTrickle integration** for production-grade streaming infrastructure

If you have an existing ComfyUI installation, the same custom nodes used to create workflows in ComfyUI will be re-used when processing video streams.

## Simple Integration Example

Here's how ComfyStream integrates PyTrickle to create a streaming server:

```python
import asyncio
from aiohttp import web
from comfystream.pipeline import Pipeline
from pytrickle import TrickleApp
from pytrickle.frames import VideoFrame, VideoOutput

class ComfyStreamProcessor:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.last_processed = None
    
    async def start(self):
        """Initialize the ComfyUI pipeline."""
        await self.pipeline.warm_pipeline()
        
    def process_frame_sync(self, frame: VideoFrame) -> VideoOutput:
        """Process video frames through ComfyUI workflow."""
        try:
            # Convert to AV frame and process through pipeline
            av_frame = frame.to_av_frame()
            
            # This would normally be async, but we use caching for real-time performance
            # See the full ComfyStream implementation for the complete async bridge
            
            # For now, return the frame (real implementation uses async processing)
            return VideoOutput(frame, "comfy_processed")
            
        except Exception as e:
            print(f"Processing error: {e}")
            return VideoOutput(frame, "passthrough")

async def create_comfystream_app():
    """Create a ComfyStream application with PyTrickle integration."""
    
    # Initialize ComfyUI pipeline
    pipeline = Pipeline(workspace_path="/workspace", width=512, height=512)
    
    # Create processor
    processor = ComfyStreamProcessor(pipeline)
    await processor.start()
    
    # Create TrickleApp with ComfyUI processing
    app = TrickleApp(
        frame_processor=processor.process_frame_sync,
        port=8080,
        host="0.0.0.0"
    )
    
    return app

async def main():
    app = await create_comfystream_app()
    await app.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
```

**Key Integration Points:**
- `Pipeline` class handles ComfyUI workflow execution
- `process_frame_sync()` bridges between PyTrickle's sync interface and ComfyUI's async processing
- Production version uses async queues and frame caching for optimal performance
- HTTP endpoints automatically available: `/api/stream/start`, `/api/stream/params`, etc.

For the complete production implementation, see `server/trickle_integration.py` and `server/comfy_stream_handler.py`.

**Parameter Updates:**
ComfyStream uses standardized parameter validation through the `ComfyUIParams` model, ensuring consistent handling of prompts, width, height, and other workflow parameters across WebRTC, HTTP API, and control channel updates.

- [comfystream](#comfystream)
  - [Quick Start](#quick-start)
    - [Docker DevContainer](#docker-devcontainer)
    - [Docker Image](#docker-image)
      - [RunPod](#runpod)
      - [Tensordock](#tensordock)
      - [Other Cloud Providers](#other-cloud-providers)
  - [Download Models](#download-models)
  - [Install package](#install-package)
    - [Custom Nodes](#custom-nodes)
    - [Usage](#usage)
  - [Run tests](#run-tests)
  - [Run server](#run-server)
  - [Run UI](#run-ui)
  - [Limitations](#limitations)
  - [Troubleshoot](#troubleshoot)

## Quick Start

### Docker DevContainer

Refer to [.devcontainer/README.md](.devcontainer/README.md) to setup ComfyStream in a devcontainer using a pre-configured ComfyUI docker environment.

For other installation options, refer to [Install ComfyUI and ComfyStream](https://pipelines.livepeer.org/docs/technical/install/local-testing) in the Livepeer pipelines documentation.

For additional information, refer to the remaining sections below.

### Docker Image

You can quickly deploy ComfyStream using the docker image `livepeer/comfystream`

Refer to the documentation at [https://pipelines.livepeer.org/docs/technical/getting-started/install-comfystream](https://pipelines.livepeer.org/docs/technical/getting-started/install-comfystream) for instructions to run locally or on a remote server.

#### RunPod

The RunPod template [livepeer-comfystream](https://runpod.io/console/deploy?template=w01m180vxx&ref=u8tlskew) can be used to deploy to RunPod.

#### Tensordock

We also have a python script that can be used to spin up a ComfyStream instance on a [Tensordock server](https://tensordock.com/). Refer to [scripts/README.md](./scripts/README.md#tensordock-auto-setup-fully-automated) for instructions.

#### Other Cloud Providers

We also provide an [Ansible playbook](https://docs.ansible.com/ansible/latest/installation_guide/index.html) for deploying ComfyStream on any cloud provider. Refer to [scripts/README.md](./scripts/README.md#cloud-agnostic-automated-setup-ansible-based-deployment) for instructions.

## Download Models

Refer to [scripts/README.md](src/comfystream/scripts/README.md) for instructions to download commonly used models.

## Install package

**Prerequisites**

- [Miniconda](https://docs.anaconda.com/miniconda/index.html#latest-miniconda-installer-links)

A separate environment can be used to avoid any dependency issues with an existing ComfyUI installation.

Create the environment:

```bash
conda create -n comfystream python=3.11
```

Activate the environment:

```bash
conda activate comfystream
```

Make sure you have [PyTorch](https://pytorch.org/get-started/locally/) installed.

Install `comfystream`:

```bash
pip install git+https://github.com/livepeer/comfystream.git

# This can be used to install from a local repo
# pip install .
# This can be used to install from a local repo in edit mode
# pip install -e .
```

### Custom Nodes

Comfystream uses a few auxiliary custom nodes to support running workflows.

**Note:** If you are using comfystream as a custom node in ComfyUI, you can skip the following steps.

If you are using comfystream as a standalone application, copy the auxiliary custom nodes into the `custom_nodes` folder of your ComfyUI workspace:

```bash
cp -r nodes/* custom_nodes/
```

For example, if your ComfyUI workspace is under `/home/user/ComfyUI`:

```bash
cp -r nodes/* /home/user/ComfyUI/custom_nodes
```

### Usage

See `example.py`.

## Run tests

Install dev dependencies:

```bash
pip install .[dev]
```

Run tests:

```bash
pytest
```

## Run server

Install dependencies:

```bash
pip install -r requirements.txt
```

If you have existing custom nodes in your ComfyUI workspace, you will need to install their requirements in your current environment:

```bash
python install.py --workspace <COMFY_WORKSPACE>
```

Run the server:

```bash
python server/app.py --workspace <COMFY_WORKSPACE>
```

Show additional options for configuring the server:

```bash
python server/app.py -h
```

**Server Options**

- `--warmup-workflow`: Specify a workflow file name to use for pipeline warmup (e.g., `sd15-tensorrt-api.json`). The workflow file must exist in the `workflows/comfystream/` directory relative to the ComfyStream project root. If not specified, uses the default workflow for warmup.
- `--skip-warmup`: Skip warming the pipeline on startup (reduces startup time but increases latency for first user)

**Available Warmup Workflows**

The following workflow files are available in `workflows/comfystream/` for use with `--warmup-workflow`:

- `inverted-color-api.json` - Simple color inversion workflow
- `sd15-tensorrt-api.json` - Stable Diffusion 1.5 with TensorRT optimization
- `depth-anything-v2-trt-example-api.json` - Depth estimation with TensorRT
- `tensor-utils-example-api.json` - Basic tensor operations
- And many others (see `workflows/comfystream/` directory)

Example usage:

```bash
# Use a specific workflow for warmup
python server/app.py --workspace <COMFY_WORKSPACE> --warmup-workflow sd15-tensorrt-api.json

# Skip warmup entirely
python server/app.py --workspace <COMFY_WORKSPACE> --skip-warmup
```

**How Warmup Works**

The warmup process loads and initializes the specified workflow before the server starts accepting connections. This pre-loads models and optimizes the pipeline, reducing latency for the first user. The warmup workflow should match the type of processing you expect to perform:

- Use `inverted-color-api.json` for simple image processing
- Use `sd15-tensorrt-api.json` for Stable Diffusion workflows with TensorRT optimization
- Use `depth-anything-v2-trt-example-api.json` for depth estimation workflows

**Important**: The workflow file must exist in the `workflows/comfystream/` directory relative to the ComfyStream project root. If no warmup workflow is specified, the server uses a default lightweight workflow. If the specified workflow file is not found, the server falls back to the default workflow and logs a warning.

**Remote Setup**

A local server should connect with a local UI out-of-the-box. It is also possible to run a local UI and connect with a remote server, but there may be additional dependencies.

In order for the remote server to connect with another peer (i.e. a browser) without any additional dependencies you will need to allow inbound/outbound UDP traffic on ports 1024-65535 ([source](https://github.com/aiortc/aiortc/issues/490#issuecomment-788807118)).

If you only have a subset of those UDP ports available, you can use the `--media-ports` flag to specify a comma delimited list of ports to use:

```bash
python server/app.py --workspace <COMFY_WORKSPACE> --media-ports 1024,1025,...
```

If you are running the server in a restrictive network environment where this is not possible, you will need to use a TURN server.

At the moment, the server supports using Twilio's TURN servers (although it is easy to make the update to support arbitrary TURN servers):

1. Sign up for a [Twilio](https://www.twilio.com/en-us) account.
2. Copy the Account SID and Auth Token from [https://console.twilio.com/](https://console.twilio.com/).
3. Set the `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` environment variables.

```bash
export TWILIO_ACCOUNT_SID=...
export TWILIO_AUTH_TOKEN=...
```

## Run UI

**Prerequisities**

- [Node.js](https://nodejs.org/en/download/package-manager)

Install dependencies

```bash
cd ui
npm install --legacy-peer-deps
npm install --save-dev cross-env
```

Run local dev server:

```bash
npm run dev
```

By default the app will be available at <http://localhost:3000>.

The Stream URL is the URL of the [server](#run-server) which defaults to <http://127.0.0.1:8889>.

> [!NOTE]
> To run the UI on HTTPS (necessary for webcam functionality), use `npm run dev:https`. You'll need to accept the self-signed certificate in your browser.

## Limitations

At the moment, a workflow must fufill the following requirements:

- The workflow must have a single primary input node that will receive individual video frames
  - The primary input node is designed by one of the following:
    - A single [PrimaryInputLoadImage](./nodes/video_stream_utils/primary_input_load_image.py) node (see [this workflow](./workflows/liveportait.json) for example usage)
      - This node can be used as a drop-in replacement for a LoadImage node
      - In this scenario, any number of additional LoadImage nodes can be used
    - A single LoadImage node
      - In this scenario, the workflow can only contain the single LoadImage node
  - At runtime, this node is replaced with a LoadTensor node
- The workflow must have a single output using a PreviewImage or SaveImage node
  - At runtime, this node is replaced with a SaveTensor node

## Troubleshoot

This project has been tested locally successfully with the following setup:

- OS: Ubuntu
- GPU: Nvidia RTX 4090
- Driver: 550.127.05
- CUDA: 12.5
- torch: 2.5.1+cu121
