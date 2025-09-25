#!/bin/bash

# Internal TensorRT engine build script that runs inside the container
# This script is designed to be executed within the AI runner container environment

set -e
[ -v DEBUG ] && set -x

# Configuration
CONDA_PYTHON="/workspace/.venv/bin/python"
MODELS="stabilityai/sd-turbo KBlueLeaf/kohaku-v2.1"
OPT_TIMESTEPS="3" # Optimal number of timesteps for the t_index_list
MIN_TIMESTEPS="1" # Minimum number of timesteps for the t_index_list
MAX_TIMESTEPS="4" # Maximum number of timesteps for the t_index_list
DIMENSIONS="512x512" # Engines are now compiled for the 384-1024 range, but keep this in case it's useful in the future
CONTROLNETS="" # Default empty, will be set from command line
IPADAPTER_TYPES="" # Default empty, will be set from command line

# Function to display help
function display_help() {
    echo "Description: This script builds TensorRT engines for StreamDiffusion models inside the container."
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --models MODEL_LIST     Comma-separated list of models (default: stabilityai/sd-turbo,KBlueLeaf/kohaku-v2.1)"
    echo "  --opt-timesteps TIMESTEPS   Optimal number of timesteps in t_index_list (default: 3)"
    echo "  --min-timesteps TIMESTEPS   Minimum number of timesteps in t_index_list (default: 1)"
    echo "  --max-timesteps TIMESTEPS   Maximum number of timesteps in t_index_list (default: 4)"
    echo "  --dimensions DIMS       Space-separated list of dimensions in WxH format (default: 384x704 704x384)"
    echo "  --output-dir DIR        Output directory for TensorRT engines (default: engines)"
    echo "  --controlnets CONTROLNETS Space-separated list of controlnet models"
    echo "  --ipadapter-types TYPES   Space-separated list of IPAdapter types to compile ('regular' or 'faceid')"
    echo "  --build-depth-anything  Build Depth-Anything TensorRT engine (requires ONNX model to be downloaded)"
    echo "  --build-pose            Build YoloNas Pose TensorRT engine (requires ONNX model to be downloaded)"
    echo "  --help                  Display this help message"
}

# Default values
MODELS_DIR=${HUGGINGFACE_HUB_CACHE:-./models}
OUTPUT_DIR="./engines"
BUILD_DEPTH_ANYTHING=false
BUILD_POSE=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="${2//,/ }"  # Convert comma-separated to space-separated
            shift 2
            ;;
        --opt-timesteps)
            OPT_TIMESTEPS="$2"
            shift 2
            ;;
        --min-timesteps)
            MIN_TIMESTEPS="$2"
            shift 2
            ;;
        --max-timesteps)
            MAX_TIMESTEPS="$2"
            shift 2
            ;;
        --dimensions)
            DIMENSIONS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --controlnets)
            CONTROLNETS="$2"
            shift 2
            ;;
        --ipadapter-types)
            IPADAPTER_TYPES="$2"
            shift 2
            ;;
        --build-depth-anything)
            BUILD_DEPTH_ANYTHING=true
            shift
            ;;
        --build-pose)
            BUILD_POSE=true
            shift
            ;;
        --help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

# Ensure we're in the right environment
if [ ! -f "$CONDA_PYTHON" ]; then
    echo "ERROR: Python executable not found at $CONDA_PYTHON"
    echo "This script must be run inside the AI runner container environment."
    exit 1
fi

# Check if the build script exists
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
BUILD_SCRIPT="$SCRIPT_DIR/build_tensorrt.py"
if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "ERROR: Build script not found at $BUILD_SCRIPT"
    echo "Make sure the StreamDiffusion build script is available in the same directory as this script."
    exit 1
fi

# Create output directory
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Check GPU availability
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
else
    echo "GPU status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo
fi

# Build TensorRT engines
echo "Starting TensorRT engine build process..."
echo "Models: $MODELS"
echo "Opt Timesteps: $OPT_TIMESTEPS"
echo "Min Timesteps: $MIN_TIMESTEPS"
echo "Max Timesteps: $MAX_TIMESTEPS"
echo "Dimensions: $DIMENSIONS"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$CONTROLNETS" ]; then
    echo "ControlNets: $CONTROLNETS"
else
    echo "ControlNets: None"
fi
if [ -n "$IPADAPTER_TYPES" ]; then
    read -r -a IPADAPTER_TYPES_ARR <<< "$IPADAPTER_TYPES"
    echo "IPAdapter Types: $IPADAPTER_TYPES"
else
    IPADAPTER_TYPES_ARR=("")
    echo "IPAdapter Types: None"
fi
if [ "$BUILD_DEPTH_ANYTHING" = true ]; then
    echo "Depth-Anything: Enabled"
else
    echo "Depth-Anything: Disabled"
fi
if [ "$BUILD_POSE" = true ]; then
    echo "YoloNas Pose: Enabled"
else
    echo "YoloNas Pose: Disabled"
fi
echo

total_builds=0

# Calculate total number of builds
for model in $MODELS; do
    for dim in $DIMENSIONS; do
        for ip_type in "${IPADAPTER_TYPES_ARR[@]}"; do
            total_builds=$((total_builds + 1))
        done
    done
done

echo "Total builds to perform: $total_builds"
echo

current_build=0

for model in $MODELS; do
    for dim in $DIMENSIONS; do
        # Parse dimensions
        width=${dim%x*}
        height=${dim#*x}

        for ip_type in "${IPADAPTER_TYPES_ARR[@]}"; do
            current_build=$((current_build + 1))

            echo "[$current_build/$total_builds] Building TensorRT engine for:"
            echo "  Model: $model"
            echo "  Opt Timesteps: $OPT_TIMESTEPS"
            echo "  Min Timesteps: $MIN_TIMESTEPS"
            echo "  Max Timesteps: $MAX_TIMESTEPS"
            echo "  Dimensions: ${width}x${height}"
            echo "  IPAdapter Type: ${ip_type:-None}"

            build_args=(
                --model-id "$model"
                --opt-timesteps "$OPT_TIMESTEPS"
                --min-timesteps "$MIN_TIMESTEPS"
                --max-timesteps "$MAX_TIMESTEPS"
                --width "$width"
                --height "$height"
                --engine-dir "$OUTPUT_DIR"
                --controlnets "$CONTROLNETS"
                --ipadapter-type "$ip_type"
            )

            if $CONDA_PYTHON "$BUILD_SCRIPT" "${build_args[@]}"; then
                echo "  ✓ Success"
            else
                echo "  ✗ Failed"
                echo "Aborting build process due to failure."
                exit 1
            fi
            echo
        done
    done
done


# Function to build Depth-Anything TensorRT engine if requested.
# TODO: Remove this once streamdiffusion lib can do this automatically for depth ControlNet.
function build_depth_anything_engine() {
    echo "Building Depth-Anything TensorRT engine..."

    engine_path="$OUTPUT_DIR/depth-anything/depth_anything_v2_vits.engine"

    if [ -f "$engine_path" ]; then
        echo "Engine already exists at: $engine_path"
        echo "Skipping build."
        return 0
    fi
    mkdir -p $(dirname "$engine_path")

    if [ ! -d "ComfyUI-Depth-Anything-Tensorrt" ]; then
        git clone https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt.git \
            && cd ComfyUI-Depth-Anything-Tensorrt \
            && git checkout 1f4c161949b3616516745781fb91444e6443cc25 2>/dev/null \
            && cd ..
    fi

    echo "Locating Depth-Anything ONNX model..."
    onnx_path=$(find "$MODELS_DIR" -name "depth_anything_v2_vits.onnx" 2>/dev/null | head -1)

    if [ -z "$onnx_path" ] || [ ! -f "$onnx_path" ]; then
        echo "ERROR: Depth-Anything ONNX model not found"
        echo "Please ensure the model is downloaded using huggingface-cli"
        return 1
    fi

    echo "Found ONNX model at: $onnx_path"

    echo "Calling export_trt.py to build engine: $engine_path"
    if $CONDA_PYTHON ./ComfyUI-Depth-Anything-Tensorrt/export_trt.py \
        --trt-path="$engine_path" \
        --onnx-path="$onnx_path"; then
        echo "  ✓ Depth-Anything TensorRT engine built successfully"
    else
        echo "  ✗ Failed to build Depth-Anything TensorRT engine"
        return 1
    fi

    echo
}

# Function to build YoloNas Pose TensorRT engine if requested.
# TODO: Remove this once streamdiffusion lib can do this automatically for pose ControlNet.
function build_pose_engine() {
    echo "Building YoloNas Pose TensorRT engine..."

    engines_dir="$(readlink -f "$OUTPUT_DIR/pose")"
    mkdir -p "$engines_dir"

    echo "Locating YoloNas Pose ONNX models..."
    onnx_model_paths=$(find "$MODELS_DIR" -name "yolo_nas_pose_l_*.onnx" 2>/dev/null)

    if [ -z "$onnx_model_paths" ]; then
        echo "ERROR: YoloNas Pose ONNX model not found"
        echo "Please ensure the model is downloaded using huggingface-cli"
        return 1
    fi

    echo "Found ONNX models at: $onnx_model_paths"

    # Check if any engine is missing before cloning the repo
    missing_engines_onnx=()
    for onnx_path in $onnx_model_paths; do
        engine_path="$engines_dir/$(basename "$onnx_path" .onnx).engine"
        if [ -f "$engine_path" ]; then
            echo "Engine already exists at: $engine_path"
            echo "Skipping build for this model."
        else
            missing_engines_onnx+=("$onnx_path")
        fi
    done

    if [ ${#missing_engines_onnx[@]} -eq 0 ]; then
        echo "All YoloNas Pose TensorRT engines already exist. Skipping build."
        echo
        return 0
    fi

    if [ ! -d "ComfyUI-YoloNasPose-Tensorrt" ]; then
        git clone https://github.com/yuvraj108c/ComfyUI-YoloNasPose-Tensorrt.git \
            && cd ComfyUI-YoloNasPose-Tensorrt \
            && git checkout 873de560bb05bf3331e4121f393b83ecc04c324a 2>/dev/null \
            && $CONDA_PYTHON -m pip install -r requirements.txt \
            && cd ..
    fi

    echo "Building ${#missing_engines_onnx[@]} YoloNas Pose TensorRT engines..."
    cd ComfyUI-YoloNasPose-Tensorrt
    for onnx_path in "${missing_engines_onnx[@]}"; do
        engine_path="$engines_dir/$(basename "$onnx_path" .onnx).engine"

        # The export_trt.py script expects a hardcoded path to the model, so create a link
        rm -f "yolo_nas_pose_l.onnx" && ln -sf "$onnx_path" "yolo_nas_pose_l.onnx"

        echo "Calling export_trt.py to build engine: $engine_path"
        if $CONDA_PYTHON export_trt.py; then
            if [ -f "yolo_nas_pose_l.engine" ]; then
                mv "yolo_nas_pose_l.engine" "$engine_path"
                echo "  ✓ YoloNas Pose TensorRT engine built successfully"
            else
                echo "  ✗ Engine file not found in expected location"
                return 1
            fi
        else
            echo "  ✗ Failed to build YoloNas Pose TensorRT engine"
            return 1
        fi
    done

    cd ..
    echo
}

if [ "$BUILD_DEPTH_ANYTHING" = true ]; then
    build_depth_anything_engine || exit 1
fi

if [ "$BUILD_POSE" = true ]; then
    build_pose_engine || exit 1
fi

# Summary
echo "Build process completed!"
echo "Total builds: $total_builds"

# List built engines
if [ -d "$OUTPUT_DIR" ]; then
    echo
    echo "Built engines:"
    find "$OUTPUT_DIR" -name "*.engine" -o -name "*.trt" 2>/dev/null | sort
fi

