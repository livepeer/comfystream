#!/bin/bash

set -e

# Validate required environment variable
if [ -z "$ORCH_SECRET" ]; then
    echo "ERROR: ORCH_SECRET environment variable is required"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate comfystream

# Change to ComfyStream directory
cd /workspace/comfystream

# Build the command with environment variables and any additional args
exec python server/byoc.py \
    --workspace="$COMFYSTREAM_WORKSPACE" \
    --host="$COMFYSTREAM_HOST" \
    --port="$COMFYSTREAM_PORT" \
    --log-level="$COMFYSTREAM_LOG_LEVEL" \
    --comfyui-inference-log-level="$COMFYUI_INFERENCE_LOG_LEVEL" \
    --width="$COMFYSTREAM_WIDTH" \
    --height="$COMFYSTREAM_HEIGHT" \
    "$@"

