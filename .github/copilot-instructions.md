# ComfyStream Development Instructions

ComfyStream is a Python package for running img2img ComfyUI workflows on video streams with a WebRTC server and Next.js UI frontend. This repository uses Python 3.11, PyTorch, Docker multi-stage builds, and Node.js for the frontend.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Environment Setup
- Install Python 3.11 (3.12+ may work but 3.11 is tested and recommended):
  - `conda create -n comfystream python=3.11`
  - `conda activate comfystream`
- Install PyTorch first (required dependency): Follow [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- Install Node.js 18+ for UI development
- Ensure Docker is available for containerization

### Package Installation and Dependencies
- **CRITICAL**: Network timeouts are common with ComfyUI dependency installation. This is a known issue.
- Install development dependencies: `pip install .[dev]` -- **may fail due to network timeouts with ComfyUI dependency**
- Alternative: Install test dependencies only: `pip install pytest pytest-cov` -- works reliably
- Install basic requirements: `pip install -r requirements.txt` -- **may fail due to ComfyUI dependency timeouts**
- Install server dependencies: `pip install -r requirements.txt` (when network allows)

### UI Development (Always Works)
- Navigate to UI directory: `cd ui`
- Install dependencies: `npm install --legacy-peer-deps` -- takes ~21 seconds, always works
- **Development server**: `npm run dev` -- starts in 1.4 seconds, available at http://localhost:3000
- **HTTPS development**: `npm run dev:https` -- for webcam functionality requiring HTTPS
- **Build production**: `npm run build` -- takes ~25 seconds. NEVER CANCEL. Set timeout to 60+ seconds.
- **Lint code**: `npm run lint` -- takes ~3 seconds
- **Format code**: `npm run format` -- takes ~1 second

### Testing
- **Python tests**: `pytest --cov --verbose --showlocals` -- **requires ComfyUI module to be installed**
- If ComfyUI installation fails due to network issues, tests cannot run
- Tests are in `test/` directory and validate prompt conversion utilities
- **UI has no separate test suite** - manual validation through dev server required

### Running the Application
- **Server**: `python server/app.py --workspace <COMFY_WORKSPACE>` -- requires all dependencies installed
- **Server help**: `python server/app.py -h` -- shows configuration options
- **Install script**: `python install.py --workspace <workspace>` -- installs custom nodes
- **Benchmark**: `python benchmark.py` -- performance testing (requires full setup)

### Docker Builds (High Time Investment)
- **Base image build**: `docker build -t livepeer/comfyui-base -f docker/Dockerfile.base .` -- **NEVER CANCEL: Expected 45+ minutes. Set timeout to 120+ minutes.**
- **Main image build**: `docker build -t livepeer/comfystream -f docker/Dockerfile .` -- **NEVER CANCEL: Expected 45+ minutes. Set timeout to 120+ minutes.**
- **Pull pre-built**: `docker pull livepeer/comfyui-base:latest` -- faster alternative to building
- **Run container**: `docker run -it --gpus all livepeer/comfystream`
- **CI builds have 200-minute timeouts** - indicates extremely long build times, base image downloads 3GB+ of CUDA libraries

## Validation Scenarios

### Always Test After Changes
- **UI Validation**: Run `npm run dev` in ui/ directory and verify server starts at http://localhost:3000
- **Build Validation**: Run `npm run build` in ui/ directory and verify successful compilation
- **Linting**: Always run `npm run lint` and `npm run format` in ui/ before committing
- **Python Validation**: If dependencies install successfully, run `pytest` to validate core functionality

### Manual Testing Requirements
- **UI Functionality**: Access development server, test basic UI interactions
- **Code Quality**: Run linting and formatting tools before any commits
- **Build Process**: Verify production build completes without errors

### When Full Installation Fails
- Document network connectivity issues with ComfyUI dependency
- Focus on UI development and validation which always works
- Use Docker approach if local installation fails
- Validate using devcontainer setup as alternative

## Timing Expectations and Timeouts

### UI Operations (Fast and Reliable)
- `npm install --legacy-peer-deps`: ~21 seconds
- `npm run build`: ~25 seconds -- **Set timeout to 60+ seconds**
- `npm run lint`: ~3 seconds
- `npm run format`: ~1 second
- `npm run dev`: Ready in ~1.4 seconds

### Python Operations (Network Dependent)
- `pip install .[dev]`: **Often fails due to network timeouts**
- `pytest`: **Requires successful ComfyUI installation**
- Test dependencies only: `pip install pytest pytest-cov` (~3 seconds, reliable)

### Docker Operations (Very Long)
- Base image build: **45+ minutes minimum. NEVER CANCEL. Set timeout to 120+ minutes.**
- Main image build: **45+ minutes minimum. NEVER CANCEL. Set timeout to 120+ minutes.**
- **Base image downloads 3GB+ of CUDA libraries** - this is why builds take so long
- **CI uses 200-minute timeouts** - indicates extremely long build processes
- **Alternative**: Use `docker pull livepeer/comfyui-base:latest` to avoid building base image

## Key Project Structure

### Important Directories
- `src/comfystream/`: Main Python package source
- `ui/`: Next.js frontend application  
- `server/`: WebRTC server implementation
- `test/`: Python test suite
- `docker/`: Docker build configurations
- `workflows/`: ComfyUI workflow examples
- `nodes/`: Custom ComfyUI nodes
- `scripts/`: Helper scripts for deployment and setup

### Key Files
- `pyproject.toml`: Python project configuration and dependencies
- `ui/package.json`: Node.js dependencies and scripts
- `requirements.txt`: Python runtime dependencies
- `server/app.py`: Main WebRTC server entry point
- `example.py`: Basic usage example
- `benchmark.py`: Performance testing script

## Common Issues and Workarounds

### Network Connectivity Problems
- ComfyUI dependency installation frequently times out
- Fallback to UI-only development when Python deps fail
- Use Docker images from registry when builds fail
- Document timeout issues rather than continuing to retry indefinitely

### Development Environment
- Use conda environment for Python 3.11
- Ensure GPU drivers and CUDA are available for full functionality
- UI development works independently of Python environment issues

### Build Process
- **NEVER CANCEL** long-running Docker builds - they are expected to take 45+ minutes
- Always set appropriate timeouts (60+ minutes for builds, 120+ minutes for Docker)
- UI builds are fast and reliable - use for rapid iteration
- **Use pre-built Docker images** when possible to avoid long build times

## CI/CD Pipeline Requirements
- Python tests: `pytest --cov --verbose --showlocals`
- UI linting: `npm run lint` in ui/
- UI formatting: `npm run format` in ui/
- Docker builds with 200-minute timeouts in GitHub Actions
- CodeQL security analysis for Python, TypeScript, and JavaScript

Always run UI linting and formatting before committing. Python tests require successful dependency installation which may not be possible in all environments due to network connectivity issues.