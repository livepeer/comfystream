import os
from pathlib import Path
import requests
from tqdm import tqdm
import yaml
import argparse
import subprocess
import sys
from rich import print

# Assuming utils is importable via standard mechanisms
from utils import get_config_path, load_model_config

# --- Constants ---
COMFYSTREAM_ROOT = Path(__file__).parents[3]
DEFAULT_WORKSPACE = os.environ.get('COMFY_UI_WORKSPACE', os.path.expanduser('~/comfyui'))

def parse_args():
    parser = argparse.ArgumentParser(description='Download ComfyUI models and optionally build TensorRT engines.')
    parser.add_argument('--workspace',
                       default=DEFAULT_WORKSPACE,
                       help=f'ComfyUI workspace directory (default: {DEFAULT_WORKSPACE} or $COMFY_UI_WORKSPACE)')
    parser.add_argument('--build-engines',
                        action='store_true',
                        help='Build TensorRT engines for configured models after downloading.')
    return parser.parse_args()

def download_file(url, destination, description=None):
    """Download a file with progress bar"""
    print(f"[blue]Attempting to download from {url} to {destination}[/blue]")
    try:
        response = requests.get(url, stream=True, timeout=30, allow_redirects=True)
        response.raise_for_status() # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))

        desc = description or Path(destination).name
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc, leave=False)

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, 'wb') as file:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                progress_bar.update(size)
        progress_bar.close()
        print(f"[blue]Successfully downloaded {desc}[/blue]")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[red]Error downloading {description or url}: {e}[/red]")
        if destination and Path(destination).exists():
             # Clean up partial download
             try:
                 Path(destination).unlink()
             except OSError:
                 pass
        return False
    except Exception as e:
        print(f"[red]An unexpected error occurred during download of {description or url}: {e}[/red]")
        if destination and Path(destination).exists():
             try:
                 Path(destination).unlink()
             except OSError:
                 pass
        return False


def build_tensorrt_engine(workspace_dir: Path, model_path: Path, engine_config: dict):
    """Builds a single TensorRT engine based on the provided config."""
    script_path_str = engine_config.get('script')
    engine_path_str = engine_config.get('engine_path')
    args_template = engine_config.get('args', [])

    assert script_path_str, f"[red]Error: Invalid engine config for model {model_path.name}. Missing 'script' field in model config.[/red]"
    assert engine_path_str, f"[red]Error: Invalid engine config for model {model_path.name}. Missing 'engine_path' in model config.[/red]"

    # Let's try resolving relative to workspace first, then comfystream root.
    script_path_ws = workspace_dir / script_path_str
    script_path_cs = COMFYSTREAM_ROOT / script_path_str
    script_path = None 

    if script_path_ws.exists():
        script_path = script_path_ws.resolve()
    elif script_path_cs.exists():
        script_path = script_path_cs.resolve()
    else:
        print(f"[red]Error: Build script not found at {script_path_ws} or {script_path_cs}[/red]")
        return False

    # Determine the correct working directory from the script's location
    absolute_cwd = script_path.parent
    if not absolute_cwd.is_dir():
         # This should theoretically not happen if the script exists, but good practice
         print(f"[red]Error: Determined CWD '{absolute_cwd}' is not a valid directory.[/red]")
         return False

    # Resolve engine path relative to workspace/models or output/
    if '/' in engine_path_str and Path(engine_path_str).parts[0] == 'output':
        engine_full_path = (workspace_dir / engine_path_str).resolve()
    else:
        engine_full_path = (workspace_dir / "models" / engine_path_str).resolve()

    # Ensure parent directory exists before creating engine file
    engine_full_path.parent.mkdir(parents=True, exist_ok=True)

    if engine_full_path.exists():
        print(f"[blue]Skipping build, engine already exists: {engine_full_path}[/blue]")
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
                )
            )
    except KeyError as e:
        print(f"[red]Error formatting arguments for {script_path.name}: Missing placeholder {e}[/red]")
        return False
    except Exception as e:
         print(f"[red]Error formatting arguments for {script_path.name}: {e}. Args: {args_template}[/red]")
         return False

    # Construct command
    command = [sys.executable, str(script_path)] + formatted_args
    print(f"\n[blue]Building engine: {' '.join(map(str, command))}[/blue]")
    # Always print the CWD being used now
    print(f"[blue]Running in directory: {absolute_cwd}[/blue]")

    try:
        # Use subprocess.run to execute the build script
        # Inherit environment, capture output
        # Ensure absolute_cwd is passed correctly
        result = subprocess.run(command, cwd=str(absolute_cwd), check=True, env=os.environ) 
        if not engine_full_path.exists():
             print(f"[red]Error: Build command completed but engine file not found at {engine_full_path}[/red]")
             return False
        print(f"[blue]Successfully built engine: {engine_full_path}[/blue]")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[red]Error building engine with script {script_path.name}:[/red]")
        print(f"[red]Command: {' '.join(map(str, e.cmd))}[/red]")
        print(f"[red]Return Code: {e.returncode}[/red]")
        return False
    except FileNotFoundError:
        print(f"[red]Error: Python executable '{sys.executable}' or script '{script_path}' not found.[/red]")
        return False
    except Exception as e:
        print(f"[red]An unexpected error occurred during engine build: {e}[/red]")
        return False


def setup_model_files(workspace_dir: Path, config_path: Path, build_engines_flag: bool):
    """Download and setup required model files based on configuration, optionally build engines."""
    try:
        config = load_model_config(config_path)
    except FileNotFoundError:
        print(f"[red]Error: Model config file not found at {config_path}[/red]")
        return
    except yaml.YAMLError as e:
        print(f"[red]Error parsing model config file: {e}[/red]")
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
        full_path = (models_path / path_str).resolve()

        model_name = model_info.get('name', model_key)
        downloaded_main_file = False

        # Ensure parent directory exists before downloading
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if not full_path.exists():
            print(f"\n[blue]Processing model: {model_name}[/blue]")
            if 'url' in model_info:
                 if download_file(model_info['url'], full_path, f"Downloading {model_name}"):
                     download_success_count += 1
                     downloaded_main_file = True
                 else:
                     download_fail_count += 1
            else:
                 print(f"[yellow]Warning: No URL specified for {model_name}, skipping download.[/yellow]")
        else:
            print(f"\n[blue]Model exists: {model_name} at {full_path}[/blue]")
            downloaded_main_file = True # Treat existing file as successfully "downloaded" for build logic

        # Handle any extra files (like configs)
        if 'extra_files' in model_info:
            for extra in model_info['extra_files']:
                extra_path_str = extra['path']
                extra_full_path = (models_path / extra_path_str).resolve()

                # Ensure parent directory exists before downloading extra file
                extra_full_path.parent.mkdir(parents=True, exist_ok=True)

                if not extra_full_path.exists():
                     if 'url' in extra:
                         print(f"[blue]Downloading extra file for {model_name}: {Path(extra_path_str).name}[/blue]")
                         if download_file(extra['url'], extra_full_path, f"Downloading {Path(extra_path_str).name}"):
                             download_success_count += 1
                         else:
                             download_fail_count += 1
                     else:
                          print(f"[yellow]Warning: No URL for extra file {extra_path_str}, skipping.[/yellow]")
                else:
                     print(f"[blue]Extra file exists: {extra_full_path}[/blue]")

        # --- Build TensorRT Engines ---
        if build_engines_flag and downloaded_main_file and 'tensorrt' in model_info:
            tensorrt_config = model_info['tensorrt']
            if tensorrt_config.get('build'):
                print(f"\n[blue]Attempting to build TensorRT engines for {model_name}...[/blue]")
                engines_to_build = tensorrt_config.get('engines', [])
                if not engines_to_build:
                     print(f"[yellow]Warning: 'tensorrt: build: true' but no 'engines' listed for {model_name}[/yellow]")
                     continue

                for engine_conf in engines_to_build:
                     if build_tensorrt_engine(workspace_dir, full_path, engine_conf):
                         build_success_count += 1
                     else:
                         build_fail_count += 1
            else:
                 print(f"[blue]Skipping TensorRT build for {model_name} (build flag is false in config)[/blue]")

    print("\n--- Summary ---")
    print(f"Model Downloads: {'[green]' + str(download_success_count) + '[/green]'} succeeded, {'[red]' + str(download_fail_count) + '[/red]'} failed.")
    if build_engines_flag:
        print(f"TensorRT Builds: {'[green]' + str(build_success_count) + '[/green]'} succeeded, {'[red]' + str(build_fail_count) + '[/red]'} failed.")
    print("[green]Setup process finished![/green]")


def main():
    args = parse_args()
    workspace_dir = Path(args.workspace).resolve() # Resolve workspace path

    # Create base workspace and output/tensorrt directories if they don't exist
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "output" / "tensorrt").mkdir(parents=True, exist_ok=True)

    # Determine config path
    config_path = COMFYSTREAM_ROOT / 'configs' / 'models.yaml'
    assert config_path.exists(), f"[red]Model config file not found at {config_path}. Please check the path.[/red]"

    print(f"[blue]Using workspace: {workspace_dir}[/blue]")
    print(f"[blue]Using model config: {config_path}[/blue]")
    if args.build_engines:
        print("[blue]TensorRT engine building is ENABLED.[/blue]")
    else:
        print("[blue]TensorRT engine building is DISABLED (use --build-engines to enable).[/blue]")

    setup_model_files(workspace_dir, config_path, args.build_engines)

if __name__ == "__main__":
    main()
