#!/bin/bash
set -e

# If an command was passed, run it instead of setting up ComfyUI workspace
if [[ $# -ne 0 ]]; then
  exec "$@"
else
  echo "Setting up ComfyUI workspace..."
  export VIRTUAL_ENV=/workspace/ComfyUI/.venv
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  source $VIRTUAL_ENV/bin/activate
  comfy set-default /workspace/ComfyUI

  echo "Refreshing Custom Nodes..."
  cat /workspace/ComfyUI/custom_nodes/comfystream/src/comfystream/scripts/requirements.txt
  uv pip install -r /workspace/ComfyUI/custom_nodes/comfystream/src/comfystream/scripts/requirements.txt
#  python custom_nodes/comfystream/src/comfystream/scripts/setup_nodes.py --workspace /workspace/ComfyUI --pull-branches

  echo "ComfyUI workspace setup complete!"
  # Set up completion if not already done
  if [[ ! -f ~/.local/share/bash-completion/completions/comfy ]]; then
      comfy --install-completion 2>/dev/null || true
  fi

  echo "Starting ComfyUI server..."
  exec comfy -- launch -- --listen 0.0.0.0 --port 8188 --front-end-version Comfy-Org/ComfyUI_frontend@v1.24.2
fi

# # Check if /workspace/ComfyUI is a symlink (indicating mounted volume)
# if [[ -L "/workspace/ComfyUI" ]]; then
#     echo "Detected mounted workspace (symlink found)"

#     cd /workspace/ComfyUI

#     # Check if ComfyUI is already installed in the mounted workspace
#     if [[ -f "main.py" ]]; then
#         echo "ComfyUI installation found in mounted workspace, restoring..."
#         # Ensure .venv exists in the workspace
#         if [[ ! -d ".venv" ]]; then
#             echo "Creating virtual environment in workspace..."
#             uv venv .venv
#         fi

#         # Activate the virtual environment
#         echo "Activating virtual environment..."
#         source .venv/bin/activate

#         # Ensure comfy-cli is available in the venv
#         if ! command -v comfy &> /dev/null; then
#             echo "Installing comfy-cli in workspace venv..."
#             uv pip install comfy-cli
#         fi

#         # Restore the existing ComfyUI workspace
#         echo "Restoring ComfyUI workspace..."
#         comfy --skip-prompt --here install --nvidia --restore --skip-requirement

#     else
#         echo "No ComfyUI installation found in mounted workspace, installing..."

#         # Install ComfyUI in the mounted workspace
#         comfy --skip-prompt --here install --nvidia

#         # Ensure .venv exists in the workspace
#         if [[ ! -d ".venv" ]]; then
#             echo "Creating virtual environment in workspace..."
#             uv venv .venv
#         fi

#         # Activate the virtual environment
#         echo "Activating virtual environment..."
#         source .venv/bin/activate

#         # Ensure comfy-cli is available in the venv
#         if ! command -v comfy &> /dev/null; then
#             echo "Installing comfy-cli in workspace venv..."
#             uv pip install comfy-cli
#         fi
#     fi

#     # Install comfystream node if not already installed
#     if [[ ! -d "custom_nodes/comfystream" ]]; then
#         echo "Installing comfystream node..."
#         comfy node registry-install comfystream
#     fi

# else
#     echo "Using built-in ComfyUI workspace (no mount detected)"
#     cd /workspace/ComfyUI

#     # Activate the pre-built virtual environment
#     echo "Activating virtual environment..."
#     source .venv/bin/activate
# fi

else
    exec "$@"
fi
