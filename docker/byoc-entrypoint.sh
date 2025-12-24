#!/usr/bin/env bash

set -e

DEFAULT_CMD=(python /workspace/comfystream/server/byoc.py)

# Allow `--` to separate entrypoint flags from the command
if [[ "${1:-}" == "--" ]]; then
  shift
fi

# If no args or first arg is an option, run the BYOC server with default args.
if [[ $# -eq 0 || "$1" == -* ]]; then
  set -- "${DEFAULT_CMD[@]}" "$@"
fi

exec conda run --no-capture-output -n comfystream --cwd /workspace/ComfyUI "$@"

