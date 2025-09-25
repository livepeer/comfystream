#!/usr/bin/env python3
"""
Setup script for StreamDiffusion models using huggingface-cli.
Based on Livepeer AI runner model requirements.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path
import yaml
from comfystream.scripts.utils import (
    get_config_path,
    get_default_workspace,
    validate_and_prompt_workspace,
    setup_workspace_environment
)

def parse_args():
    parser = argparse.ArgumentParser(description='Setup StreamDiffusion models')
    parser.add_argument('--workspace',
                       default=get_default_workspace(),
                       help='ComfyUI workspace directory (default: ~/comfyui or $COMFYUI_WORKSPACE)')
    parser.add_argument('--config',
                       default=None,
                       help='StreamDiffusion models config file (default: streamdiffusion_models.yaml)')
    return parser.parse_args()

def check_huggingface_cli():
    """Check if huggingface-cli is available"""
    # Try multiple possible locations for huggingface-cli
    possible_paths = [
        'huggingface-cli',  # In PATH
        '/workspace/.venv/bin/huggingface-cli',  # Our venv
        sys.executable.replace('python', 'huggingface-cli')  # Same dir as python
    ]
    
    for cli_path in possible_paths:
        try:
            result = subprocess.run([cli_path, 'version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✓ Found huggingface-cli at {cli_path}")
            print(f"  {result.stdout.strip()}")
            return cli_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print("❌ huggingface-cli not found. Please install it with: pip install huggingface_hub[cli]")
    return None

def download_huggingface_model(model_id, cache_dir, cli_path, include=None, exclude=None):
    """Download a model from Hugging Face using huggingface-cli"""
    cmd = [cli_path, 'download', model_id, '--cache-dir', str(cache_dir)]
    
    if include:
        # Handle include patterns - convert space-separated to individual --include flags
        include_patterns = include.split()
        for pattern in include_patterns:
            cmd.extend(['--include', pattern])
    
    if exclude:
        # Handle exclude patterns - convert space-separated to individual --exclude flags  
        exclude_patterns = exclude.split()
        for pattern in exclude_patterns:
            cmd.extend(['--exclude', pattern])
    
    print(f"📥 Downloading {model_id}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully downloaded {model_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {model_id}: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return False

def setup_streamdiffusion_models(workspace_dir, config_path=None):
    """Download and setup StreamDiffusion models"""
    cli_path = check_huggingface_cli()
    if not cli_path:
        return False
    
    if config_path is None:
        config_path = get_config_path('streamdiffusion_models.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: StreamDiffusion models config file not found at {config_path}")
        return False
    except yaml.YAMLError as e:
        print(f"Error parsing StreamDiffusion models config file: {e}")
        return False

    models_dir = workspace_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_count = 0
    
    print(f"🚀 Starting StreamDiffusion model downloads to {models_dir}")
    print("=" * 60)
    
    # Group models by category for better organization
    categories = {
        "Base Models": ["sd-turbo", "sdxl-turbo", "openjourney-v4", "dreamshaper-8"],
        "SD2.1 ControlNets": [k for k in config['models'].keys() if k.startswith('controlnet-sd21')],
        "SD1.5 ControlNets": [k for k in config['models'].keys() if k.startswith('controlnet-sd15')],
        "SDXL ControlNets": [k for k in config['models'].keys() if k.startswith('controlnet-sdxl')],
        "IP-Adapter": ["ip-adapter", "ip-adapter-faceid"],
        "Preprocessing": ["depth-anything-onnx", "yolo-nas-pose-onnx"],
        "Safety": ["nsfw-detector"]
    }
    
    for category, model_keys in categories.items():
        if not any(key in config['models'] for key in model_keys):
            continue
            
        print(f"\n📁 {category}")
        print("-" * 40)
        
        for model_key in model_keys:
            if model_key not in config['models']:
                continue
                
            model_info = config['models'][model_key]
            total_count += 1
            
            if model_info.get('type') == 'huggingface':
                if download_huggingface_model(
                    model_info['url'], 
                    models_dir,
                    cli_path,
                    model_info.get('include'),
                    model_info.get('exclude')
                ):
                    success_count += 1
            else:
                print(f"⚠️  Skipping {model_key} - not a huggingface model")
    
    print("\n" + "=" * 60)
    print(f"✅ StreamDiffusion model setup completed!")
    print(f"📊 Successfully downloaded {success_count}/{total_count} models")
    
    if success_count < total_count:
        print(f"⚠️  {total_count - success_count} models failed to download")
        return False
    
    return True

def main():
    """Entry point for command line usage."""
    args = parse_args()
    workspace_dir = validate_and_prompt_workspace(args.workspace, "setup-streamdiffusion-models")
    
    setup_workspace_environment(workspace_dir)
    
    success = setup_streamdiffusion_models(workspace_dir, args.config)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
