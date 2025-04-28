import os
from pathlib import Path
import requests
from tqdm import tqdm
import yaml
import argparse
import subprocess
import sys

# Assuming utils is importable via standard mechanisms
from utils import get_config_path, load_model_config

# --- Constants ---
COMFYSTREAM_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_WORKSPACE = os.environ.get('COMFY_UI_WORKSPACE', os.path.expanduser('~/comfyui'))

def parse_args():
    parser = argparse.ArgumentParser(description='Download ComfyUI models and optionally build TensorRT engines.')
    parser.add_argument('--workspace',
                       default=DEFAULT_WORKSPACE,
                       help=f'ComfyUI workspace directory (default: {DEFAULT_WORKSPACE} or $COMFY_UI_WORKSPACE)')
    parser.add_argument('--build-engines',
                        action='store_true',
                        help='Build TensorRT engines for configured models after downloading.')
    # Add argument for config path, default relative to script location
    parser.add_argument('--config',
                        default=None,
                        help='Path to the models.yaml config file.')
    return parser.parse_args()

def download_file(url, destination, description=None):
    """Download a file with progress bar"""
    print(f"Attempting to download from {url} to {destination}")
    try:
        response = requests.get(url, stream=True, timeout=30, allow_redirects=True)
        response.raise_for_status() # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))

        desc = description or Path(destination).name
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc, leave=False)

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, 'wb') as file:
            for data in response.iter_content(chunk_size=8192): # Increased chunk size
                size = file.write(data)
                progress_bar.update(size)
        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
             print(f"Error: Downloaded size ({progress_bar.n}) does not match expected size ({total_size}) for {desc}")
             # Decide if this should be fatal? For now just print warning.
        print(f"Successfully downloaded {desc}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {description or url}: {e}")
        if destination and Path(destination).exists():
             # Clean up partial download
             try:
                 Path(destination).unlink()
             except OSError:
                 pass
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download of {description or url}: {e}")
        if destination and Path(destination).exists():
             try:
                 Path(destination).unlink()
             except OSError:
                 pass
        return False


def build_tensorrt_engine(workspace_dir: Path, model_path: Path, engine_config: dict):
    """Builds a single TensorRT engine based on the provided config."""
    script_path_str = engine_config.get('script')
    engine_rel_path = engine_config.get('engine_path')
    args_template = engine_config.get('args', [])
    cwd_rel_path = engine_config.get('cwd') # Get optional CWD from config

    if not script_path_str or not engine_rel_path:
        print(f"Error: Invalid engine config for model {model_path.name}. Missing 'script' or 'engine_path'. Config: {engine_config}")
        return False

    # Let's try resolving relative to workspace first, then comfystream root.
    script_path_ws = workspace_dir / script_path_str
    script_path_cs = COMFYSTREAM_ROOT / script_path_str

    if script_path_ws.exists():
        script_path = script_path_ws
    elif script_path_cs.exists():
        script_path = script_path_cs
    else:
        print(f"Error: Build script not found at {script_path_ws} or {script_path_cs}")
        return False

    # Resolve engine path relative to workspace/models
    if '/' in engine_rel_path and Path(engine_rel_path).parts[0] == 'output':
        engine_full_path = (workspace_dir / engine_rel_path).resolve()
    else:
        engine_full_path = (workspace_dir / "models" / engine_rel_path).resolve()

    # Ensure parent directory exists before creating engine file
    engine_full_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve CWD if specified in config, relative to workspace
    absolute_cwd = (workspace_dir / cwd_rel_path).resolve() if cwd_rel_path else None
    if absolute_cwd and not absolute_cwd.is_dir():
        print(f"Warning: Specified CWD '{absolute_cwd}' does not exist or is not a directory. Ignoring CWD.")
        absolute_cwd = None

    if engine_full_path.exists():
        print(f"Skipping build, engine already exists: {engine_full_path}")
        return True

    # Format arguments
    formatted_args = []
    try:
        for arg in args_template:
            formatted_args.append(
                str(arg).format(
                    model_path=model_path.resolve(),
                    engine_path=engine_full_path,
                    workspace_dir=workspace_dir.resolve(),
                    # Add more placeholders if needed
                )
            )
    except KeyError as e:
        print(f"Error formatting arguments for {script_path.name}: Missing placeholder {e}")
        return False
    except Exception as e:
         print(f"Error formatting arguments for {script_path.name}: {e}. Args: {args_template}")
         return False

    # Construct command
    command = [sys.executable, str(script_path)] + formatted_args
    print(f"\nBuilding engine: {' '.join(command)}")
    if absolute_cwd:
        print(f"Running in directory: {absolute_cwd}")

    try:
        # Use subprocess.run to execute the build script
        # Inherit environment, capture output
        # Let the build script's output stream directly to the console
        result = subprocess.run(command, cwd=absolute_cwd, check=True, env=os.environ) 
        if not engine_full_path.exists():
             print(f"Error: Build command completed but engine file not found at {engine_full_path}")
             return False
        print(f"Successfully built engine: {engine_full_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building engine with script {script_path.name}:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"Error: Python executable '{sys.executable}' or script '{script_path}' not found.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during engine build: {e}")
        return False


def setup_model_files(workspace_dir: Path, config_path: Path, build_engines_flag: bool):
    """Download and setup required model files based on configuration, optionally build engines."""
    try:
        config = load_model_config(config_path)
    except FileNotFoundError:
        print(f"Error: Model config file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing model config file: {e}")
        return

    models_path = workspace_dir / "models"
    base_path = workspace_dir # For resolving paths like custom_nodes/

    download_success_count = 0
    download_fail_count = 0
    build_success_count = 0
    build_fail_count = 0

    # Ensure base models directory exists
    models_path.mkdir(parents=True, exist_ok=True)

    for model_key, model_info in config['models'].items():
        # Determine the full path for the main model file
        path_str = model_info['path']
        if '/' in path_str and Path(path_str).parts[0] == 'custom_nodes':
            # custom_nodes paths are relative to workspace_dir (base_path)
            full_path = (base_path / path_str).resolve()
        else:
            # Paths with subdirectories (e.g. controlnet/) are relative to models_path
            # Also handles paths without '/' assuming they belong in models_path root (though less likely)
            full_path = (models_path / path_str).resolve()

        model_name = model_info.get('name', model_key)
        downloaded_main_file = False

        # Ensure parent directory exists before downloading
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if not full_path.exists():
            print(f"\nProcessing model: {model_name}")
            if 'url' in model_info:
                 if download_file(model_info['url'], full_path, f"Downloading {model_name}"):
                     download_success_count += 1
                     downloaded_main_file = True
                 else:
                     download_fail_count += 1
            else:
                 print(f"Warning: No URL specified for {model_name}, skipping download.")
        else:
            print(f"\nModel exists: {model_name} at {full_path}")
            downloaded_main_file = True # Treat existing file as successfully "downloaded" for build logic

        # Handle any extra files (like configs)
        if 'extra_files' in model_info:
            for extra in model_info['extra_files']:
                extra_path_str = extra['path']
                # Resolve extra file paths similarly
                if '/' in extra_path_str and Path(extra_path_str).parts[0] == 'custom_nodes':
                     extra_full_path = (base_path / extra_path_str).resolve()
                else:
                     extra_full_path = (models_path / extra_path_str).resolve()

                # Ensure parent directory exists before downloading extra file
                extra_full_path.parent.mkdir(parents=True, exist_ok=True)

                if not extra_full_path.exists():
                     if 'url' in extra:
                         print(f"Downloading extra file for {model_name}: {Path(extra_path_str).name}")
                         if download_file(extra['url'], extra_full_path, f"Downloading {Path(extra_path_str).name}"):
                             download_success_count += 1
                         else:
                             download_fail_count += 1
                     else:
                          print(f"Warning: No URL for extra file {extra_path_str}, skipping.")
                else:
                     print(f"Extra file exists: {extra_full_path}")

        # --- Build TensorRT Engines ---
        if build_engines_flag and downloaded_main_file and 'tensorrt' in model_info:
            tensorrt_config = model_info['tensorrt']
            if tensorrt_config.get('build'):
                print(f"\nAttempting to build TensorRT engines for {model_name}...")
                engines_to_build = tensorrt_config.get('engines', [])
                if not engines_to_build:
                     print(f"Warning: 'tensorrt: build: true' but no 'engines' listed for {model_name}")
                     continue

                for engine_conf in engines_to_build:
                     if build_tensorrt_engine(workspace_dir, full_path, engine_conf):
                         build_success_count += 1
                     else:
                         build_fail_count += 1
            else:
                 print(f"Skipping TensorRT build for {model_name} (build flag is false in config)")

    print("\n--- Summary ---")
    print(f"Model Downloads: {download_success_count} succeeded, {download_fail_count} failed.")
    if build_engines_flag:
        print(f"TensorRT Builds: {build_success_count} succeeded, {build_fail_count} failed.")
    print("Setup process finished!")


def main():
    args = parse_args()
    workspace_dir = Path(args.workspace).resolve() # Resolve workspace path

    # Create base workspace and output/tensorrt directories if they don't exist
    # Model subdirs are created on demand in setup_model_files
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "output" / "tensorrt").mkdir(parents=True, exist_ok=True)

    # Determine config path
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        # Default to configs/models.yaml relative to COMFYSTREAM_ROOT
        config_path = COMFYSTREAM_ROOT / 'configs' / 'models.yaml'

    print(f"Using workspace: {workspace_dir}")
    print(f"Using model config: {config_path}")
    if args.build_engines:
        print("TensorRT engine building is ENABLED.")
    else:
        print("TensorRT engine building is DISABLED (use --build-engines to enable).")

    setup_model_files(workspace_dir, config_path, args.build_engines)

if __name__ == "__main__":
    main()
