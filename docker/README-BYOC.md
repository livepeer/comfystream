# ComfyStream BYOC Docker

This Docker setup runs ComfyStream in BYOC (Bring Your Own Compute) mode using the pytrickle framework.

## Building the Image

```bash
docker build -f docker/Dockerfile.byoc -t comfystream-byoc .
```

## Running the Container

The only required environment variable is `ORCH_SECRET`. All other variables have sensible defaults based on the launch.json configuration.

### Basic Usage

```bash
docker run -e ORCH_SECRET=your-secret-here comfystream-byoc
```

### With Custom Configuration

```bash
docker run \
  -e ORCH_SECRET=your-secret-here \
  -e ORCH_URL=https://your-orchestrator:9995 \
  -e COMFYSTREAM_PORT=8001 \
  -e COMFYSTREAM_WIDTH=1024 \
  -e COMFYSTREAM_HEIGHT=1024 \
  -p 8001:8001 \
  comfystream-byoc
```

### Additional Arguments

You can pass additional arguments to the ComfyStream BYOC server:

```bash
docker run -e ORCH_SECRET=your-secret-here comfystream-byoc --some-extra-arg value
```

## Environment Variables

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
| `COMFYSTREAM_WORKSPACE` | `/workspace/ComfyUI` | ComfyUI workspace path |
| `COMFYSTREAM_HOST` | `0.0.0.0` | Server host |
| `COMFYSTREAM_PORT` | `8000` | Server port |
| `COMFYSTREAM_LOG_LEVEL` | `INFO` | Log level |
| `COMFYUI_INFERENCE_LOG_LEVEL` | `DEBUG` | ComfyUI inference log level |
| `COMFYSTREAM_WIDTH` | `512` | Default width |
| `COMFYSTREAM_HEIGHT` | `512` | Default height |

## Exposed Ports

- `8000`: ComfyStream BYOC server (default)

