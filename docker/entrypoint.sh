#!/bin/bash

set -e

# Add help command to show usage
show_help() {
  echo "Usage: entrypoint.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --use-volume     Initialize persistent volume mount"
  echo "  --download-models       Download default models"
  echo "  --download-streamdiffusion Download StreamDiffusion models for TensorRT engines"
  echo "  --build-engines         Build TensorRT engines for default models"
  echo "  --opencv-cuda           Setup OpenCV with CUDA support"
  echo "  --server                Start ComfyUI only"
  echo "  --api                   Start ComfyStream API Server only"
  echo "  --ui                    Start ComfyStream UI only"
  echo "  --help                  Show this help message"
  echo ""
}

if [ "$1" = "--help" ]; then
  show_help
  exit 0
fi

# Define reusable paths
WORKSPACE_STORAGE="/app/storage"
COMFYUI_DIR="/workspace/ComfyUI"
MODELS_DIR="$COMFYUI_DIR/models"
OUTPUT_DIR="$COMFYUI_DIR/output"

# Initialize variables to track which services to start
START_COMFYUI=false
START_API=false
START_UI=false

# First pass: check for service flags and set variables
for arg in "$@"; do
  case "$arg" in
    --server)
      START_COMFYUI=true
      ;;
    --api)
      START_API=true
      ;;
    --ui)
      START_UI=true
      ;;
  esac
done

# Map persistent volume mount for models and engines using symlinks
if [ "$1" = "--use-volume" ] && [ -d "$WORKSPACE_STORAGE" ]; then
  echo "Initializing persistent volume mount..."
  if [ ! -L "$MODELS_DIR" ]; then
      rm -rf "$MODELS_DIR"
      ln -s $WORKSPACE_STORAGE/ComfyUI--models "$MODELS_DIR"
      echo "created symlink for models at $MODELS_DIR"
  else
      echo "symlink for models already exists at $MODELS_DIR"
  fi

  if [ ! -L "$OUTPUT_DIR" ]; then
      rm -rf "$OUTPUT_DIR"
      ln -s $WORKSPACE_STORAGE/ComfyUI--output "$OUTPUT_DIR"
      if [ ! -d "$WORKSPACE_STORAGE/ComfyUI--output/tensorrt" ]; then
          mkdir -p "$WORKSPACE_STORAGE/ComfyUI--output/tensorrt"
      fi
      echo "created symlink for output at $OUTPUT_DIR"
  else
      echo "symlink for output already exists at $OUTPUT_DIR"
  fi
  shift
fi

if [ "$1" = "--download-models" ]; then
  cd /workspace/comfystream
  python -m comfystream.scripts.setup_models --workspace /workspace/ComfyUI
  shift
fi

if [ "$1" = "--download-streamdiffusion" ]; then
  cd /workspace/comfystream
  python -m comfystream.scripts.setup_streamdiffusion_models --workspace /workspace/ComfyUI
  shift
fi

TENSORRT_DIR="/workspace/ComfyUI/models/tensorrt"
DEPTH_ANYTHING_DIR="${TENSORRT_DIR}/depth-anything"
DEPTH_ANYTHING_ENGINE="depth_anything_vitl14-fp16.engine"
DEPTH_ANYTHING_ENGINE_LARGE="depth_anything_v2_vitl-fp16.engine"
FASTERLIVEPORTRAIT_DIR="/workspace/ComfyUI/models/liveportrait_onnx"

if [ "$1" = "--build-engines" ]; then
  cd /workspace/comfystream

  # Build Static Engine for Dreamshaper - Square (512x512)
  python src/comfystream/scripts/build_trt.py --model /workspace/ComfyUI/models/unet/dreamshaper-8-dmd-1kstep.safetensors --out-engine /workspace/ComfyUI/output/tensorrt/static-dreamshaper8_SD15_\$stat-b-1-h-512-w-512_00001_.engine --width 512 --height 512

  # Build Static Engine for Dreamshaper - Portrait (384x704)
  python src/comfystream/scripts/build_trt.py --model /workspace/ComfyUI/models/unet/dreamshaper-8-dmd-1kstep.safetensors --out-engine /workspace/ComfyUI/output/tensorrt/static-dreamshaper8_SD15_\$stat-b-1-h-704-w-384_00001_.engine --width 384 --height 704

  # Build Static Engine for Dreamshaper - Landscape (704x384)
  python src/comfystream/scripts/build_trt.py --model /workspace/ComfyUI/models/unet/dreamshaper-8-dmd-1kstep.safetensors --out-engine /workspace/ComfyUI/output/tensorrt/static-dreamshaper8_SD15_\$stat-b-1-h-384-w-704_00001_.engine --width 704 --height 384

  # Build Dynamic Engine for Dreamshaper
  python src/comfystream/scripts/build_trt.py \
                --model /workspace/ComfyUI/models/unet/dreamshaper-8-dmd-1kstep.safetensors \
                --out-engine /workspace/ComfyUI/output/tensorrt/dynamic-dreamshaper8_SD15_\$dyn-b-1-4-2-h-512-704-w-320-384-448_00001_.engine \
                --width 384 \
                --height 704 \
                --min-width 320 \
                --min-height 512 \
                --max-width 448 \
                --max-height 704

  # Build Engine for Depth Anything V2
  if [ ! -f "$DEPTH_ANYTHING_DIR/$DEPTH_ANYTHING_ENGINE" ]; then
    if [ ! -d "$DEPTH_ANYTHING_DIR" ]; then
      mkdir -p "$DEPTH_ANYTHING_DIR"
    fi
    cd "$DEPTH_ANYTHING_DIR"
    python /workspace/ComfyUI/custom_nodes/ComfyUI-Depth-Anything-Tensorrt/export_trt.py
  else
    echo "Engine for DepthAnything2 already exists at ${DEPTH_ANYTHING_DIR}/${DEPTH_ANYTHING_ENGINE}, skipping..."
  fi

  # Build Engine for Depth Anything2 (large)
  if [ ! -f "$DEPTH_ANYTHING_DIR/$DEPTH_ANYTHING_ENGINE_LARGE" ]; then
    cd "$DEPTH_ANYTHING_DIR"
    python /workspace/ComfyUI/custom_nodes/ComfyUI-Depth-Anything-Tensorrt/export_trt.py --trt-path "${DEPTH_ANYTHING_DIR}/${DEPTH_ANYTHING_ENGINE_LARGE}" --onnx-path "${DEPTH_ANYTHING_DIR}/depth_anything_v2_vitl.onnx"
  else
    echo "Engine for DepthAnything2 (large) already exists at ${DEPTH_ANYTHING_DIR}/${DEPTH_ANYTHING_ENGINE_LARGE}, skipping..."
  fi

  # Build StreamDiffusion Engines using the new build script
  cd /workspace/comfystream
  
  # Download and patch the build scripts
  if [ ! -f "build_tensorrt_internal.sh" ]; then
    curl -o build_tensorrt_internal.sh https://raw.githubusercontent.com/livepeer/ai-runner/refs/heads/main/runner/app/tools/streamdiffusion/build_tensorrt_internal.sh
    chmod +x build_tensorrt_internal.sh
    # Hack: Replace hardcoded conda python path with our venv python
    sed -i 's|CONDA_PYTHON="/workspace/miniconda3/envs/comfystream/bin/python"|CONDA_PYTHON="/workspace/.venv/bin/python"|g' build_tensorrt_internal.sh
  fi
    
  # First build: SD-Turbo with SD2.1 ControlNets
  echo "Building StreamDiffusion engines for SD-Turbo with SD2.1 ControlNets..."
  HUGGINGFACE_HUB_CACHE="/workspace/ComfyUI/models" ./build_tensorrt_internal.sh \
    --models 'stabilityai/sd-turbo' \
    --opt-timesteps '3' \
    --min-timesteps '1' \
    --max-timesteps '4' \
    --controlnets 'thibaud/controlnet-sd21-openpose-diffusers thibaud/controlnet-sd21-hed-diffusers thibaud/controlnet-sd21-canny-diffusers thibaud/controlnet-sd21-depth-diffusers thibaud/controlnet-sd21-color-diffusers thibaud/controlnet-sd21-ade20k-diffusers thibaud/controlnet-sd21-normalbae-diffusers' \
    --build-depth-anything \
    --build-pose

  # Second build: OpenJourney v4 and Dreamshaper 8 with SD1.5 ControlNets and IPAdapter
  echo "Building StreamDiffusion engines for OpenJourney v4 and Dreamshaper 8..."
  HUGGINGFACE_HUB_CACHE="/workspace/ComfyUI/models" ./build_tensorrt_internal.sh \
    --models 'prompthero/openjourney-v4 Lykon/dreamshaper-8' \
    --opt-timesteps '3' \
    --min-timesteps '1' \
    --max-timesteps '4' \
    --controlnets 'lllyasviel/control_v11f1p_sd15_depth lllyasviel/control_v11f1e_sd15_tile lllyasviel/control_v11p_sd15_canny' \
    --ipadapter-types 'regular faceid' \
    --build-depth-anything \
    --build-pose

  # Third build: SDXL-Turbo with SDXL ControlNets and IPAdapter
  echo "Building StreamDiffusion engines for SDXL-Turbo..."
  HUGGINGFACE_HUB_CACHE="/workspace/ComfyUI/models" ./build_tensorrt_internal.sh \
    --models 'stabilityai/sdxl-turbo' \
    --dimensions '1024x1024' \
    --opt-timesteps '1' \
    --min-timesteps '1' \
    --max-timesteps '4' \
    --controlnets 'xinsir/controlnet-depth-sdxl-1.0 xinsir/controlnet-canny-sdxl-1.0 xinsir/controlnet-tile-sdxl-1.0' \
    --ipadapter-types 'regular faceid' \
    --build-depth-anything \
    --build-pose
  shift
fi

if [ "$1" = "--opencv-cuda" ]; then
  cd /workspace/comfystream

  # Check if OpenCV CUDA build already exists
  if [ ! -f "/workspace/comfystream/opencv-cuda-release.tar.gz" ]; then
    # Download and extract OpenCV CUDA build
    DOWNLOAD_NAME="opencv-cuda-release.tar.gz"
    wget -q -O "$DOWNLOAD_NAME" https://github.com/JJassonn69/ComfyUI-Stream-Pack/releases/download/v2/opencv-cuda-release.tar.gz
    tar -xzf "$DOWNLOAD_NAME" -C /workspace/comfystream/
    rm "$DOWNLOAD_NAME"
  else
    echo "OpenCV CUDA build already exists, skipping download."
  fi

  # Install required libraries
  apt-get update && apt-get install -y \
    libgflags-dev \
    libgoogle-glog-dev \
    libjpeg-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev

  # Remove existing cv2 package
  SITE_PACKAGES_DIR="$(uv python dir --bin)/lib/python3.12/site-packages"
  rm -rf "${SITE_PACKAGES_DIR}/cv2"*

  # Copy new cv2 package
  cp -r /workspace/comfystream/cv2 "${SITE_PACKAGES_DIR}/"

  # Handle library dependencies
  UV_ENV_LIB="$(uv python dir --bin)/lib"

  # Remove existing libstdc++ and copy system one
  rm -f "${UV_ENV_LIB}/libstdc++.so"*
  cp /usr/lib/x86_64-linux-gnu/libstdc++.so* "${UV_ENV_LIB}/"

  # Copy OpenCV libraries
  cp /workspace/comfystream/opencv/build/lib/libopencv_* /usr/lib/x86_64-linux-gnu/

  # remove the opencv-contrib and cv2 folders
  rm -rf /workspace/comfystream/opencv_contrib
  rm -rf /workspace/comfystream/cv2

  echo "OpenCV CUDA installation completed"
  shift
fi

cd /workspace/comfystream

# If any service flags were specified, start supervisord and the requested services
if [ "$START_COMFYUI" = true ] || [ "$START_API" = true ] || [ "$START_UI" = true ]; then
  # Start supervisord in background
  /usr/bin/supervisord -c /etc/supervisor/supervisord.conf &
  sleep 2  # Give supervisord time to start

  # Start requested services
  if [ "$START_COMFYUI" = true ]; then
    supervisorctl -c /etc/supervisor/supervisord.conf start comfyui
  fi

  if [ "$START_API" = true ]; then
    supervisorctl -c /etc/supervisor/supervisord.conf start comfystream-api
  fi

  if [ "$START_UI" = true ]; then
    supervisorctl -c /etc/supervisor/supervisord.conf start comfystream-ui
  fi

  # Keep the script running
  tail -f /var/log/supervisord.log
fi

exec "$@"
