#!/bin/bash
set -euo pipefail

# Install torchaudio with the correct version matching the installed torch
echo "Installing torchaudio with matching torch version..."

# Get torch version and CUDA version using Python
TORCH_INFO=$(python -c "
import torch
import re

torch_version_full = torch.__version__
torch_ver_match = re.match(r'(\d+\.\d+\.\d+)', torch_version_full)
if not torch_ver_match:
    raise ValueError(f'Could not parse torch version from {torch_version_full}')

torch_ver = torch_ver_match.group(1)
cuda_ver_tag = f'cu{torch.version.cuda.replace(\".\", \"\")}'

print(f'{torch_ver}:{cuda_ver_tag}')
")

# Parse the output
IFS=':' read -r TORCH_VER CUDA_VER_TAG <<< "$TORCH_INFO"

echo "Detected torch version: $TORCH_VER"
echo "Detected CUDA version tag: $CUDA_VER_TAG"

# Install torchaudio with the matching version
echo "Installing torchaudio==${TORCH_VER}+${CUDA_VER_TAG}..."
uv pip install --no-deps \
    "torchaudio==${TORCH_VER}+${CUDA_VER_TAG}" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_VER_TAG}"

echo "torchaudio installation completed successfully!"
