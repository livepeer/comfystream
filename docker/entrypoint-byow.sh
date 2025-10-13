#!/bin/bash
set -e
export UV_LINK_MODE=copy

# If a command was passed, run it instead of setting up ComfyUI workspace
if [[ $# -ne 0 ]]; then
  exec "$@"
fi

echo "Starting ComfyUI workspace setup..."

# Detect if /workspace/ComfyUI is a mounted workspace
IS_MOUNTED_WORKSPACE=false
COMFYUI_PATH="/workspace/ComfyUI"
COMFYSTREAM_BUILTIN_PATH="/workspace/ComfyUI/custom_nodes/comfystream"

# Check if the workspace appears to be mounted
# A built-in workspace will have a marker file created during Docker build
MARKER_FILE="$COMFYUI_PATH/.comfystream_builtin_workspace"

if [[ -f "$MARKER_FILE" ]]; then
    IS_MOUNTED_WORKSPACE=false
    echo "Detected built-in workspace (marker file present)"
else
    IS_MOUNTED_WORKSPACE=true
    echo "Detected mounted workspace (no marker file)"
fi

if [[ "$IS_MOUNTED_WORKSPACE" == "true" ]]; then
    echo "=== Mounted Workspace Mode ==="
    cd "$COMFYUI_PATH"

    # Activate the venv from mounted workspace
    export VIRTUAL_ENV="$COMFYUI_PATH/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    export UV_NO_BUILD_ISOLATION=1
    export UV_OVERRIDES=/tmp/comfystream/constraints.txt

    FRESH_INSTALL=false
    # Check if ComfyUI is already installed
    if [[ ! -f "main.py" ]]; then
        $FRESH_INSTALL=true
        echo "ComfyUI not found in mounted directory, installing..."
        echo "Cloning ComfyUI directly (comfy-cli would create nested directory)..."

        # Clone ComfyUI directly instead of using comfy-cli install
        # This avoids the nested directory issue
        git clone --branch v0.3.60 --depth 1 https://github.com/comfyanonymous/ComfyUI.git /tmp/ComfyUI

        # Move contents to current directory
        shopt -s dotglob
        mv /tmp/ComfyUI/* "$COMFYUI_PATH/"
        rmdir /tmp/ComfyUI

        echo "ComfyUI cloned successfully!"

        # Clone ComfyUI-Manager
        if [[ ! -d "custom_nodes/ComfyUI-Manager" ]]; then
            echo "Installing ComfyUI-Manager..."
            mkdir -p custom_nodes
            git clone https://github.com/Comfy-Org/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager
        fi
    fi

    # Check if .venv exists in mounted workspace, create if needed
    if [[ ! -d ".venv" ]]; then
        echo "Creating virtual environment in mounted workspace..."
        uv venv .venv --python 3.12
        source "$VIRTUAL_ENV/bin/activate"
    fi

    # Ensure pip is available in existing venv (in case it was created without pip)
    if ! python -m pip --version &> /dev/null; then
        echo "Pip not found in venv, installing..."
        python -m ensurepip
        python -m pip install --upgrade pip
    fi

    # Install comfy-cli first if not available
    if ! command -v comfy &> /dev/null; then
        echo "Installing comfy-cli..."
        source "$VIRTUAL_ENV/bin/activate"
        uv pip install comfy-cli
        comfy tracking disable
    fi

    # Set this workspace as default for comfy-cli
    if [[ $FRESH_INSTALL == true ]]; then
        comfy --skip-prompt --workspace=$COMFYUI_PATH install --nvidia --skip-requirement
    else
        comfy --skip-prompt --workspace=$COMFYUI_PATH install --nvidia --restore
    fi
    comfy set-default "$COMFYUI_PATH"

    # Copy comfy-lock file if it doesn't exist in workspace
    if [[ ! -f ".comfy-lock.yaml" ]] && [[ -f "/tmp/comfystream/comfy-lock.yaml" ]]; then
        echo "Copying comfy-lock.yaml to workspace..."
        cp /tmp/comfystream/comfy-lock.yaml .comfy-lock.yaml
    fi

    # Note: ComfyUI requirements are intentionally NOT installed here
    # Dependencies are managed via constraints.txt and comfystream requirements

    # Restore custom nodes from comfy-lock.yaml if it exists
    if [[ -f ".comfy-lock.yaml" ]]; then
        echo "Restoring custom nodes from comfy-lock.yaml..."
        comfy node restore-snapshot .comfy-lock.yaml || echo "Warning: Some custom nodes may have failed to install"
    fi

    echo "ComfyUI installation complete!"

    # Configure git to trust the mounted workspace directory
    echo "Configuring git safe.directory for mounted workspace..."
    git config --global --add safe.directory "$COMFYUI_PATH"

    # Also configure for custom nodes
    for custom_node_dir in custom_nodes/*; do
        if [[ -d "$custom_node_dir/.git" ]]; then
            git config --global --add safe.directory "$COMFYUI_PATH/$custom_node_dir"
        fi
    done

    # Ensure comfystream is available in custom_nodes
    if [[ ! -d "custom_nodes/comfystream" ]]; then
        echo "Comfystream not found in custom_nodes, copying from built-in..."
        mkdir -p custom_nodes
        cp -r "$COMFYSTREAM_BUILTIN_PATH" custom_nodes/
    fi

    # Install comfystream in editable mode
    if [[ -d "custom_nodes/comfystream" ]]; then
        echo "Installing comfystream..."
        cd custom_nodes/comfystream
        uv pip install -e .
        cd "$COMFYUI_PATH"
    fi

    # Install comfystream requirements if needed
    if [[ -f "custom_nodes/comfystream/requirements.txt" ]]; then
        echo "Installing comfystream requirements..."
        uv pip install -r custom_nodes/comfystream/requirements.txt
    fi

    echo "Mounted workspace setup complete!"

else
    echo "=== Built-in Workspace Mode ==="
    cd "$COMFYUI_PATH"

    # Use the pre-built venv
    export VIRTUAL_ENV="$COMFYUI_PATH/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    export UV_NO_BUILD_ISOLATION=1
    comfy --skip-prompt set-default "$COMFYUI_PATH"
    source "$VIRTUAL_ENV/bin/activate"
    echo "Built-in workspace ready!"
fi

# Set up bash completion for comfy-cli if not already done
if [[ ! -f ~/.local/share/bash-completion/completions/comfy ]]; then
    comfy --install-completion 2>/dev/null || true
fi


echo "Starting ComfyUI server..."
exec comfy launch -- --listen 0.0.0.0 --port 8188 --front-end-version Comfy-Org/ComfyUI_frontend@v1.24.2
