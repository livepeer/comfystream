import os
import subprocess
import sys
import shutil
from pathlib import Path
import yaml
import argparse
from utils import get_config_path, load_model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Setup ComfyUI nodes and models")
    parser.add_argument(
        "--workspace",
        default=os.environ.get("COMFY_UI_WORKSPACE", Path("~/comfyui").expanduser()),
        help="ComfyUI workspace directory (default: ~/comfyui or $COMFY_UI_WORKSPACE)",
    )
    parser.add_argument(
        "--pull-branches",
        action="store_true",
        default=False,
        help="Update existing nodes to their specified branches",
    )
    return parser.parse_args()


def setup_environment(workspace_dir):
    os.environ["COMFY_UI_WORKSPACE"] = str(workspace_dir)
    os.environ["PYTHONPATH"] = str(workspace_dir)
    os.environ["CUSTOM_NODES_PATH"] = str(workspace_dir / "custom_nodes")


def is_comfy_cli_available():
    """Check if comfy-cli is available on the system"""
    return shutil.which("comfy") is not None


def get_package_manager():
    """Return the appropriate package manager (comfy-cli or pip)"""
    return "comfy-cli" if is_comfy_cli_available() else "pip"


def setup_directories(workspace_dir):
    """Create required directories in the workspace"""
    # Create base directories
    workspace_dir.mkdir(parents=True, exist_ok=True)
    custom_nodes_dir = workspace_dir / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)


def install_custom_nodes(workspace_dir, config_path=None, pull_branches=False):
    """Install custom nodes based on configuration"""
    if config_path is None:
        config_path = get_config_path("nodes.yaml")
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

    # Get the appropriate package manager
    package_manager = get_package_manager()
    print(f"Using package manager: {package_manager}")

    # Get the absolute path to constraints.txt
    constraints_path = Path(__file__).parent / "constraints.txt"
    if not constraints_path.exists():
        print(f"Warning: constraints.txt not found at {constraints_path}")
        constraints_path = None

    try:
        for _, node_info in config["nodes"].items():
            dir_name = node_info["url"].split("/")[-1].replace(".git", "")
            node_path = custom_nodes_path / dir_name

            print(f"Installing {node_info['name']}...")

            # Clone or update the repository
            if not node_path.exists():
                cmd = ["git", "clone", node_info["url"]]
                if "branch" in node_info:
                    cmd.extend(["-b", node_info["branch"]])
                subprocess.run(cmd, check=True)
            elif pull_branches and "branch" in node_info:
                print(f"Updating {node_info['name']} to latest {node_info['branch']}...")
                subprocess.run(["git", "-C", dir_name, "fetch", "origin"], check=True)
                subprocess.run(["git", "-C", dir_name, "checkout", node_info["branch"]], check=True)
                subprocess.run(["git", "-C", dir_name, "pull", "origin", node_info["branch"]], check=True)
            else:
                print(f"{node_info['name']} already exists, skipping clone.")

            # Checkout specific commit if branch is a commit hash
            if "branch" in node_info and len(node_info["branch"]) == 40:  # SHA-1 hash length
                print(f"Checking out specific commit {node_info['branch']}...")
                subprocess.run(["git", "-C", dir_name, "fetch", "origin"], check=True)
                subprocess.run(["git", "-C", dir_name, "checkout", node_info["branch"]], check=True)

            # Install the node using comfy-cli or pip
            if package_manager == "comfy-cli":
                if pull_branches and node_path.exists():
                    # Use local mode for existing nodes (pull-branches scenario)
                    print(f"Installing {node_info['name']} in local mode...")
                    install_cmd = ["comfy", "node", "install", "--fast-deps", "--mode", "local", str(node_path)]
                else:
                    # Use URL mode for new installations
                    print(f"Installing {node_info['name']} from URL...")
                    install_cmd = ["comfy", "node", "install", "--fast-deps", "--url", node_info["url"]]
                    if "branch" in node_info:
                        install_cmd.extend(["--branch", node_info["branch"]])
                
                subprocess.run(install_cmd, check=True)
            else:
                # Fallback to pip installation for requirements and dependencies
                # Install requirements if present
                requirements_file = node_path / "requirements.txt"
                if requirements_file.exists():
                    install_cmd = [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(requirements_file),
                    ]
                    if constraints_path and constraints_path.exists():
                        install_cmd.extend(["-c", str(constraints_path)])
                    subprocess.run(install_cmd, check=True)

                # Install additional dependencies if specified
                if "dependencies" in node_info:
                    for dep in node_info["dependencies"]:
                        install_cmd = [sys.executable, "-m", "pip", "install", dep]
                        if constraints_path and constraints_path.exists():
                            install_cmd.extend(["-c", str(constraints_path)])
                        subprocess.run(install_cmd, check=True)

            print(f"Installed {node_info['name']}")
    except Exception as e:
        print(f"Error installing {node_info['name']} {e}")
        raise e


def setup_nodes():
    args = parse_args()
    workspace_dir = Path(args.workspace)

    setup_environment(workspace_dir)
    setup_directories(workspace_dir)
    install_custom_nodes(workspace_dir, pull_branches=args.pull_branches)


if __name__ == "__main__":
    setup_nodes()
