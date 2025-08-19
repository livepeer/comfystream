#!/bin/bash

set -e

# Default model if not specified
MODEL=${VLLM_MODEL:-"microsoft/DialoGPT-medium"}
HOST=${VLLM_HOST:-"0.0.0.0"}
PORT=${VLLM_PORT:-"8000"}

echo "Starting VLLM server with model: $MODEL"
echo "Host: $HOST, Port: $PORT"

# Start VLLM server
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "translation-model" \
    --disable-log-requests \
    "$@"