import argparse
from typing import List
import sys
import os

# Add current directory to Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sd_params import StreamDiffusionParams, ControlNetConfig, IPAdapterConfig  
from pipelines.streamdiffusion import load_streamdiffusion_sync

def create_controlnet_configs(controlnet_model_ids: List[str]) -> List[ControlNetConfig]:
    """
    Create dummy ControlNet configurations for compilation using typed models.
    The exact parameters don't matter for compilation, just need the model to be loaded.
    """
    controlnet_configs = []
    for model_id in controlnet_model_ids:
        config = ControlNetConfig(
            model_id=model_id,
            conditioning_scale=0.5,
            preprocessor="passthrough",  # Simplest preprocessor
            preprocessor_params={},
            enabled=True,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
        )
        controlnet_configs.append(config)
    return controlnet_configs

def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engines for StreamDiffusion")

    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID or path to load (e.g. KBlueLeaf/kohaku-v2.1, stabilityai/sd-turbo)"
    )
    parser.add_argument(
        "--opt-timesteps",
        type=int,
        default=3,
        help="Number of timesteps in t_index_list (default: 3)"
    )
    parser.add_argument(
        "--min-timesteps",
        type=int,
        default=1,
        help="Minimum number of timesteps in t_index_list (default: 1)"
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=4,
        help="Maximum number of timesteps in t_index_list (default: 4)"
    )
    parser.add_argument(
        "--engine-dir",
        type=str,
        default="engines",
        help="Directory to save TensorRT engines (default: engines)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of the output image (default: 512)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of the output image (default: 512)"
    )
    parser.add_argument(
        "--controlnets",
        type=str,
        default="",
        help="Space-separated list of ControlNet model IDs to compile (e.g. 'lllyasviel/control_v11f1e_sd15_tile lllyasviel/control_v11f1p_sd15_depth')"
    )
    parser.add_argument(
        "--ipadapter-type",
        type=str,
        default="",
        help="IPAdapter type to compile. If set, it must be either 'regular' or 'faceid'"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.ipadapter_type not in ["", "regular", "faceid"]:
        raise ValueError(f"Invalid IPAdapter type: {args.ipadapter_type}. Must be either '', 'regular' or 'faceid'")

    # Create t_index_list based on number of timesteps. Only the size matters...
    t_index_list = list(range(1, 50, 50 // args.opt_timesteps))[:args.opt_timesteps]

    print(f"Building TensorRT engines for model: {args.model_id}")
    print(f"Using {args.opt_timesteps} opt timesteps: {t_index_list} with min {args.min_timesteps} and max {args.max_timesteps}")
    print(f"Image dimensions: {args.width}x{args.height}")

    # Calculate latent dimensions (VAE downscales by factor of 8)
    latent_width = args.width // 8
    latent_height = args.height // 8
    print(f"Expected latent dimensions: {latent_width}x{latent_height}")
    print(f"Engines will be saved to: {args.engine_dir}")

    # Create ControlNet configurations if provided
    controlnets = None
    if args.controlnets:
        controlnet_model_ids = args.controlnets.split()
        controlnets = create_controlnet_configs(controlnet_model_ids)
        print(f"ControlNets ({len(controlnet_model_ids)}):")
        for i, cn_id in enumerate(controlnet_model_ids):
            print(f"  {i}: {cn_id}")

    load_streamdiffusion_sync(
        params=StreamDiffusionParams(
            model_id=args.model_id,
            t_index_list=t_index_list,
            acceleration="tensorrt",
            width=args.width,
            height=args.height,
            controlnets=controlnets,
            use_safety_checker=True,
            ip_adapter=IPAdapterConfig(type=args.ipadapter_type) if args.ipadapter_type else None,
        ),
        min_batch_size=args.min_timesteps,
        max_batch_size=args.max_timesteps,
        engine_dir=args.engine_dir,
        build_engines=True,
    )
    print("TensorRT engine building completed successfully!")

if __name__ == "__main__":
    main()
