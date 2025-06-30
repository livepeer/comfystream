# Building Docker Images

All Docker images should be built from the root directory of the repository for consistency and proper context.

## Available Images

### Option 1: Using Docker Compose (Recommended)
```bash
# Build and run all services from the byoc directory
cd byoc
docker-compose up --build
```

This will automatically build both images and start the full BYOC environment with:
- Livepeer orchestrator and gateway
- ComfyStream reverse server 
- Capability registration service

### Option 2: Manual Docker Build

#### Reverse Server (ComfyStream with BYOC support)
```bash
# Build from root directory
docker build -f byoc/Dockerfile.reverse_server -t reverse-server .
```

#### Capability Registration (BYOC)
```bash
# Build from root directory
docker build -f byoc/Dockerfile.register_capability -t register-capability .
```

## Directory Structure

- `byoc/` - BYOC-specific scripts, utilities, and Dockerfiles
  - `register_capability.py` - Script to register capabilities with Livepeer orchestrators
  - `Dockerfile.reverse_server` - Main ComfyStream application with BYOC support
  - `Dockerfile.register_capability` - Capability registration service
  - `docker-compose.yaml` - Complete BYOC environment setup
  - `BUILD.md` - This build documentation
- `docker/` - Contains other Docker-related files
- `server/` - Contains main application files

## Build Context

All Docker builds use the repository root as the build context, which allows:
- Access to `server/app.py` for the reverse server (referenced from repo root)
- Access to `byoc/register_capability.py` for capability registration (referenced from repo root)
- Proper path resolution for all dependencies

## Notes

- The reverse server copies `server/app.py` to the container working directory
- The capability registration service copies `byoc/register_capability.py` from the repository root
- Both containers are optimized to include only necessary files
- All BYOC-related files are consolidated in the top-level `byoc/` directory 