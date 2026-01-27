#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path

# Globals populated after workspace setup
comfy = None
detect_version_from_model = None
get_helper_from_model = None
export_onnx = None
TRTDiffusionBackbone = None


def setup_comfy(workspace_dir: str):
    """Ensure ComfyUI workspace is importable and load comfy modules.

    For TensorRT engine building, we use the cloned ComfyUI workspace directly
    (not the pip-installed comfyui package) because custom nodes like
    ComfyUI_TensorRT expect the traditional ComfyUI module structure.
    """
    global \
        comfy, \
        detect_version_from_model, \
        get_helper_from_model, \
        export_onnx, \
        TRTDiffusionBackbone

    # Normalize and export the workspace so downstream imports/tools see it
    workspace_dir = str(Path(workspace_dir).expanduser().resolve())
    os.environ["COMFYUI_CWD"] = workspace_dir
    os.environ["COMFYUI_WORKSPACE"] = workspace_dir

    print(f"[build_trt] Using COMFYUI_CWD={workspace_dir}")

    # Ensure workspace directories have __init__.py so they are proper packages
    # (not namespace packages) and take priority over pip-installed versions
    workspace_path = Path(workspace_dir)
    package_dirs = ["comfy_extras"]
    for pkg_dir in package_dirs:
        init_file = workspace_path / pkg_dir / "__init__.py"
        if init_file.parent.exists() and not init_file.exists():
            init_file.touch()
            print(f"[build_trt] Created {init_file}")

    # Add workspace and custom_nodes to sys.path FIRST so they take priority
    custom_nodes_dir = str(workspace_path / "custom_nodes")

    # Insert at the beginning so workspace takes priority over site-packages
    if workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)
    if custom_nodes_dir not in sys.path:
        sys.path.insert(0, custom_nodes_dir)

    # Clear any pip-installed comfy modules from sys.modules so the workspace
    # versions are imported instead. The pip-installed comfyui package has
    # __init__.py files which would otherwise take priority over the workspace's
    # namespace packages.
    modules_to_clear = ["comfy", "comfy_extras", "nodes"]
    for mod_prefix in modules_to_clear:
        to_delete = [key for key in sys.modules if key == mod_prefix or key.startswith(f"{mod_prefix}.")]
        for key in to_delete:
            del sys.modules[key]

    # Now import comfy from the workspace
    import comfy as _comfy
    import comfy.model_management as _cm

    comfy = _comfy
    comfy.model_management = _cm

    # Import TensorRT custom node modules
    from ComfyUI_TensorRT.models.supported_models import (
        detect_version_from_model as _detect,
        get_helper_from_model as _get_helper,
    )
    from ComfyUI_TensorRT.onnx_utils.export import export_onnx as _export
    from ComfyUI_TensorRT.tensorrt_diffusion_model import (
        TRTDiffusionBackbone as _TRTBackbone,
    )

    detect_version_from_model = _detect
    get_helper_from_model = _get_helper
    export_onnx = _export
    TRTDiffusionBackbone = _TRTBackbone


def parse_args():
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from a ComfyUI model.")
    parser.add_argument(
        "--workspace",
        type=str,
        default=os.environ.get(
            "COMFYUI_CWD", os.environ.get("COMFYUI_WORKSPACE", str(Path.home() / "ComfyUI"))
        ),
        help="Path to the ComfyUI workspace (default: $COMFYUI_CWD, else $COMFYUI_WORKSPACE, else ~/ComfyUI)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the .ckpt/.safetensors or ComfyUI model name you want to convert",
    )
    parser.add_argument(
        "--out-engine",
        type=str,
        required=True,
        help="Path to the output .engine file to produce",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the exported and built engine (default 1)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width in pixels for the exported model (default 512)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height in pixels for the exported model (default 512)",
    )

    # Dynamic Engine Optional Args
    parser.add_argument(
        "--min-width", type=int, default=None, help="Minimum width for dynamic shape (optional)"
    )
    parser.add_argument(
        "--min-height", type=int, default=None, help="Minimum height for dynamic shape (optional)"
    )
    parser.add_argument(
        "--max-width", type=int, default=None, help="Maximum width for dynamic shape (optional)"
    )
    parser.add_argument(
        "--max-height", type=int, default=None, help="Maximum height for dynamic shape (optional)"
    )

    parser.add_argument(
        "--context",
        type=int,
        default=1,
        help="Context multiplier for the exported model (default 1)",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        default=False,
        help="If set, attempts to export the ONNX with FP8 transformations (Flux or standard).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable more logging / debug prints."
    )
    return parser.parse_args()


def build_trt_engine(
    model_path: str,
    engine_out_path: str,
    batch_size_opt: int = 1,
    width_opt: int = 512,
    height_opt: int = 512,
    min_width: int = None,
    min_height: int = None,
    max_width: int = None,
    max_height: int = None,
    context_opt: int = 1,
    num_video_frames: int = 14,
    fp8: bool = False,
    verbose: bool = False,
    workspace_dir: str | None = None,
):
    """
    1) Load the model from ComfyUI by path or name
    2) Export to ONNX
    3) Build a TensorRT .engine file
    """

    # Check if the engine file already exists
    if os.path.exists(engine_out_path):
        print(f"[INFO] Engine file already exists: {engine_out_path}")
        return

    # Extract the directory from the file path and ensure it exists
    directory = os.path.dirname(engine_out_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if verbose:
        print(f"[INFO] Starting build for model: {model_path}")
        print(f"       Output Engine Path: {engine_out_path}")
        print(
            f"       (batch={batch_size_opt}, H={height_opt}, W={width_opt}, context={context_opt}, "
            f"num_video_frames={num_video_frames}, fp8={fp8})"
        )

    # 1) Load model in GPU:
    if workspace_dir:
        setup_comfy(workspace_dir)

    comfy.model_management.unload_all_models()

    loaded_model = comfy.sd.load_diffusion_model(model_path, model_options={})
    if loaded_model is None:
        raise ValueError("Failed to load model.")

    comfy.model_management.load_models_gpu(
        [loaded_model], force_patch_weights=True, force_full_load=True
    )

    # 2) Export to ONNX at the desired shape
    # We'll place the ONNX in a temporary folder
    timestamp_str = str(int(time.time()))
    temp_dir = os.path.join(comfy.model_management.get_torch_device().type + "_temp", timestamp_str)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    onnx_filename = f"model_{timestamp_str}.onnx"
    onnx_path = os.path.join(temp_dir, onnx_filename)

    if verbose:
        print(f"[INFO] Exporting ONNX to: {onnx_path}")

    export_onnx(
        model=loaded_model,
        path=onnx_path,
        batch_size=batch_size_opt,
        height=height_opt,
        width=width_opt,
        num_video_frames=num_video_frames,
        context_multiplier=context_opt,
        fp8=fp8,
    )

    # 3) Build the TRT engine
    model_version = detect_version_from_model(loaded_model)
    model_helper = get_helper_from_model(loaded_model)

    trt_model = TRTDiffusionBackbone(model_helper)

    # Dynamic engine support: only if min/max width/height provided
    is_dynamic = all(v is not None for v in [min_width, max_width, min_height, max_height])
    min_config = {
        "batch_size": batch_size_opt,
        "height": min_height if is_dynamic else height_opt,
        "width": min_width if is_dynamic else width_opt,
        "context_len": context_opt * model_helper.context_len,
    }
    opt_config = {
        "batch_size": batch_size_opt,
        "height": height_opt,
        "width": width_opt,
        "context_len": context_opt * model_helper.context_len,
    }
    max_config = {
        "batch_size": batch_size_opt,
        "height": max_height if is_dynamic else height_opt,
        "width": max_width if is_dynamic else width_opt,
        "context_len": context_opt * model_helper.context_len,
    }

    # The tensorrt_diffusion_model build() signature is typically:
    #   build(onnx_path, engine_path, timing_cache_path, opt_config, min_config, max_config)
    # If you have a separate 'timing_cache.trt', put it next to this script:
    timing_cache_path = os.path.join(
        workspace_dir or os.path.dirname(__file__), "output", "tensorrt", "timing_cache"
    )

    if verbose:
        print(f"[INFO] Building engine -> {engine_out_path}")

    success = trt_model.build(
        onnx_path=onnx_path,
        engine_path=engine_out_path,
        timing_cache_path=timing_cache_path,
        opt_config=opt_config,
        min_config=min_config,
        max_config=max_config,
    )
    if not success:
        raise RuntimeError("[ERROR] TensorRT engine build failed")

    print(f"[OK] Created TensorRT engine: {engine_out_path}")

    # Clean up
    comfy.model_management.unload_all_models()
    try:
        os.remove(onnx_path)
    except:
        pass
    try:
        os.rmdir(temp_dir)
    except:
        pass


def main():
    args = parse_args()
    setup_comfy(args.workspace)
    build_trt_engine(
        model_path=args.model,
        engine_out_path=args.out_engine,
        batch_size_opt=args.batch_size,
        height_opt=args.height,
        width_opt=args.width,
        min_width=args.min_width,
        min_height=args.min_height,
        max_width=args.max_width,
        max_height=args.max_height,
        context_opt=args.context,
        fp8=args.fp8,
        verbose=args.verbose,
        workspace_dir=args.workspace,
    )


if __name__ == "__main__":
    main()
