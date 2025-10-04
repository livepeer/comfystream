#!/bin/bash

set -e

echo "Starting ComfyStream in BYOC mode..."

# Validate required environment variable
if [ -z "$ORCH_SECRET" ]; then
    echo "ERROR: ORCH_SECRET environment variable is required for BYOC mode"
    exit 1
fi

# Activate conda environment and start ComfyStream BYOC server
eval "$(conda shell.bash hook)" && conda activate comfystream && cd /workspace/ComfyUI && \
conda run -n comfystream --cwd /workspace/ComfyUI --no-capture-output python /workspace/comfystream/server/byoc.py \
    --workspace="/workspace/ComfyUI" \
    --host="${HOST:-0.0.0.0}" \
    --port="${PORT:-8000}" \
    --log-level="${LOG_LEVEL:-INFO}" \
    --comfyui-inference-log-level="${COMFYUI_INFERENCE_LOG_LEVEL:-DEBUG}" \
    --orch-url="$ORCH_URL" \
    --capability-name="${CAPABILITY_NAME:-comfystream}" \
    --capability-url="${CAPABILITY_URL:-http://172.17.0.1:8000}" \
    --capability-description="${CAPABILITY_DESCRIPTION:-ComfyUI streaming processor for BYOC mode}" \
    --capability-price-per-unit="${CAPABILITY_PRICE_PER_UNIT:-0}" \
    --capability-price-scaling="${CAPABILITY_PRICE_SCALING:-1}" \
    --capability-capacity="${CAPABILITY_CAPACITY:-1}"

#exec "$@"
