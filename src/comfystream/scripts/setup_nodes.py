import os
import subprocess
import sys
from pathlib import Path
import yaml
import argparse
from typing import Optional, Union
from utils import get_config_path, load_model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Setup ComfyUI nodes and models")
    parser.add_argument(
        "--workspace",
        default=os.environ.get("COMFY_UI_WORKSPACE", Path("~/comfyui").expanduser()),
        help="ComfyUI workspace directory (default: ~/comfyui or $COMFY_UI_WORKSPACE)",
    )
    return parser.parse_args()


def setup_environment(workspace_dir: Union[str, Path]) -> None:
    """Set up environment variables for ComfyUI workspace."""
    workspace_dir = Path(workspace_dir)
    os.environ["COMFY_UI_WORKSPACE"] = str(workspace_dir)
    os.environ["PYTHONPATH"] = str(workspace_dir)
    os.environ["CUSTOM_NODES_PATH"] = str(workspace_dir / "custom_nodes")


def setup_directories(workspace_dir: Union[str, Path]) -> None:
    """Create required directories in the workspace."""
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    custom_nodes_dir = workspace_dir / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)


def install_custom_nodes(workspace_dir: Union[str, Path], config_path: Optional[Union[str, Path]] = None) -> None:
    """Install custom nodes based on configuration.
    
    Args:
        workspace_dir: Path to ComfyUI workspace directory
        config_path: Optional path to nodes.yaml config file. If None, will use default config path.
    """
    workspace_dir = Path(workspace_dir)
    if config_path is None:
        config_path = get_config_path("nodes.yaml")
    config_path = Path(config_path)
    
    try:
        config = load_model_config(config_path)
    except FileNotFoundError:
        print(f"Error: Nodes config file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing nodes config file: {e}")
        return

    custom_nodes_path = workspace_dir / "custom_nodes"
    custom_nodes_path.mkdir(parents=True, exist_ok=True)
    os.chdir(custom_nodes_path)

    try:
        for _, node_info in config["nodes"].items():
            dir_name = node_info["url"].split("/")[-1].replace(".git", "")
            node_path = custom_nodes_path / dir_name

            print(f"Installing {node_info['name']}...")

            # Clone the repository if it doesn't already exist
            if not node_path.exists():
                cmd = ["git", "clone", node_info["url"]]
                if "branch" in node_info:
                    cmd.extend(["-b", node_info["branch"]])
                subprocess.run(cmd, check=True)
            else:
                print(f"{node_info['name']} already exists, skipping clone.")

            # Checkout specific commit if branch is a commit hash
            if (
                "branch" in node_info and len(node_info["branch"]) == 40
            ):  # SHA-1 hash length
                subprocess.run(
                    ["git", "-C", dir_name, "checkout", node_info["branch"]], check=True
                )

            # Install requirements if present
            requirements_file = node_path / "requirements.txt"
            if requirements_file.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(requirements_file),
                    ],
                    check=True,
                )

            # Install additional dependencies if specified
            if "dependencies" in node_info:
                for dep in node_info["dependencies"]:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", dep], check=True
                    )

            print(f"Installed {node_info['name']}")
    except Exception as e:
        print(f"Error installing {node_info['name']} {e}")
        raise e


def setup_nodes(workspace_dir: Optional[Union[str, Path]] = None, config_path: Optional[Union[str, Path]] = None) -> None:
    """Set up ComfyUI nodes and environment.
    
    Args:
        workspace_dir: Optional path to ComfyUI workspace directory. If None, will use default from env or ~/comfyui
        config_path: Optional path to nodes.yaml config file. If None, will use default config path.
    """
    if workspace_dir is None:
        workspace_dir = os.environ.get("COMFY_UI_WORKSPACE", Path("~/comfyui").expanduser())
    
    workspace_dir = Path(workspace_dir)
    
    setup_environment(workspace_dir)
    setup_directories(workspace_dir)
    install_custom_nodes(workspace_dir, config_path)


def main():
    """Entry point for command line usage."""
    args = parse_args()
    setup_nodes(args.workspace)


if __name__ == "__main__":
    main()
