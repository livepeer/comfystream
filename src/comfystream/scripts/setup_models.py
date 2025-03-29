import os
from pathlib import Path
import requests
from tqdm import tqdm
import yaml
import argparse
from typing import Optional, Union
from utils import get_config_path, load_model_config

def parse_args():
    parser = argparse.ArgumentParser(description='Setup ComfyUI models')
    parser.add_argument('--workspace',
                       default=os.environ.get('COMFY_UI_WORKSPACE', os.path.expanduser('~/comfyui')),
                       help='ComfyUI workspace directory (default: ~/comfyui or $COMFY_UI_WORKSPACE)')
    return parser.parse_args()

def download_file(url: str, destination: Union[str, Path], description: Optional[str] = None) -> None:
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    desc = description or os.path.basename(destination)
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()

def setup_model_files(workspace_dir: Union[str, Path], config_path: Optional[Union[str, Path]] = None) -> None:
    """Download and setup required model files based on configuration.
    
    Args:
        workspace_dir: Path to ComfyUI workspace directory
        config_path: Optional path to models.yaml config file. If None, will use default config path.
    """
    workspace_dir = Path(workspace_dir)
    if config_path is None:
        config_path = get_config_path('models.yaml')
    config_path = Path(config_path)
    
    try:
        config = load_model_config(config_path)
    except FileNotFoundError:
        print(f"Error: Model config file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing model config file: {e}")
        return

    models_path = workspace_dir / "models"
    base_path = workspace_dir

    for _, model_info in config['models'].items():
        # Determine the full path based on whether it's in custom_nodes or models
        if model_info['path'].startswith('custom_nodes/'):
            full_path = base_path / model_info['path']
        else:
            full_path = models_path / model_info['path']

        if not full_path.exists():
            print(f"Downloading {model_info['name']}...")
            download_file(
                model_info['url'],
                full_path,
                f"Downloading {model_info['name']}"
            )
            print(f"Downloaded {model_info['name']} to {full_path}")

            # Handle any extra files (like configs)
            if 'extra_files' in model_info:
                for extra in model_info['extra_files']:
                    extra_path = models_path / extra['path']
                    if not extra_path.exists():
                        download_file(
                            extra['url'],
                            extra_path,
                            f"Downloading {os.path.basename(extra['path'])}"
                        )
    print("Models download completed!")

def setup_directories(workspace_dir: Union[str, Path]) -> None:
    """Create required directories in the workspace.
    
    Args:
        workspace_dir: Path to ComfyUI workspace directory
    """
    workspace_dir = Path(workspace_dir)
    # Create base directories
    workspace_dir.mkdir(parents=True, exist_ok=True)
    models_dir = workspace_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create model subdirectories
    model_dirs = [
        "checkpoints/SD1.5",
        "controlnet",
        "vae",
        "tensorrt",
        "unet",
        "LLM",
    ]
    for dir_name in model_dirs:
        (models_dir / dir_name).mkdir(parents=True, exist_ok=True)

def setup_models(workspace_dir: Optional[Union[str, Path]] = None, config_path: Optional[Union[str, Path]] = None) -> None:
    """Set up ComfyUI models and environment.
    
    Args:
        workspace_dir: Optional path to ComfyUI workspace directory. If None, will use default from env or ~/comfyui
        config_path: Optional path to models.yaml config file. If None, will use default config path.
    """
    if workspace_dir is None:
        workspace_dir = os.environ.get('COMFY_UI_WORKSPACE', os.path.expanduser('~/comfyui'))
    
    workspace_dir = Path(workspace_dir)
    
    setup_directories(workspace_dir)
    setup_model_files(workspace_dir, config_path)

def main():
    """Entry point for command line usage."""
    args = parse_args()
    setup_models(args.workspace)

if __name__ == "__main__":
    main()
