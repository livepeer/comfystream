import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

CONSTRAINTS_PATH = Path(__file__).parent / "constraints.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Setup ComfyUI nodes and models")
    parser.add_argument(
        "--workspace",
        "--cwd",
        dest="workspace",
        default=os.environ.get("COMFYUI_CWD", Path("~/comfyui").expanduser()),
        help="ComfyUI workspace directory (default: ~/comfyui or $COMFYUI_CWD)",
    )
    parser.add_argument(
        "--pull-branches",
        action="store_true",
        default=False,
        help="Update existing nodes to their specified branches",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to custom nodes config file (default: configs/nodes.yaml). Can be a filename (searches in configs/), or an absolute/relative path.",
    )
    return parser.parse_args()


def setup_environment(workspace_dir):
    os.environ["COMFYUI_CWD"] = str(workspace_dir)
    os.environ["PYTHONPATH"] = str(workspace_dir)
    os.environ["CUSTOM_NODES_PATH"] = str(workspace_dir / "custom_nodes")


def setup_directories(workspace_dir):
    """Create required directories in the workspace"""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    custom_nodes_dir = workspace_dir / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)


def install_custom_nodes(workspace_dir, config_path=None, pull_branches=False):
    """Install custom nodes based on configuration"""
    if config_path is None:
        config_path = Path("configs") / "nodes.yaml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Nodes config file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing nodes config file: {e}")
        return

    custom_nodes_path = workspace_dir / "custom_nodes"
    custom_nodes_path.mkdir(parents=True, exist_ok=True)
    os.chdir(custom_nodes_path)

    failed_nodes = []

    # Build constraints args once, used for all pip installs
    constraints_args = ["-c", str(CONSTRAINTS_PATH)] if CONSTRAINTS_PATH.exists() else []

    for _, node_info in config["nodes"].items():
        try:
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
                subprocess.run(
                    ["git", "-C", dir_name, "pull", "origin", node_info["branch"]], check=True
                )
            else:
                print(f"{node_info['name']} already exists, skipping clone.")

            # Checkout specific commit if branch is a commit hash
            if "branch" in node_info and len(node_info["branch"]) == 40:  # SHA-1 hash length
                print(f"Checking out specific commit {node_info['branch']}...")
                subprocess.run(["git", "-C", dir_name, "fetch", "origin"], check=True)
                subprocess.run(["git", "-C", dir_name, "checkout", node_info["branch"]], check=True)

            # Install requirements if present
            requirements_file = node_path / "requirements.txt"
            if requirements_file.exists():
                print(f"Installing requirements from {requirements_file}")

                # Parse requirements file to extract --extra-index-url lines
                # uv doesn't support these directives in requirements files
                extra_index_urls = []
                package_lines = []

                with open(requirements_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        # Check if --extra-index-url is on its own line
                        if line.startswith("--extra-index-url"):
                            url = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
                            if url:
                                extra_index_urls.append(url)
                        # Check if --extra-index-url is inline with a package
                        elif "--extra-index-url" in line:
                            parts = line.split("--extra-index-url", 1)
                            package = parts[0].strip()
                            url = parts[1].strip() if len(parts) > 1 else ""
                            if package:
                                package_lines.append(package)
                            if url:
                                extra_index_urls.append(url)
                        else:
                            # Strip [all] extra from nvidia-modelopt to avoid
                            # onnxruntime-gpu version conflicts
                            if line.startswith("nvidia-modelopt"):
                                line = line.replace("[all]", "")
                            package_lines.append(line)

                # Create temp requirements file without --extra-index-url directives
                if extra_index_urls:
                    temp_req = node_path / "requirements.txt.tmp"
                    with open(temp_req, "w") as f:
                        f.write("\n".join(package_lines))

                    uv_cmd = ["uv", "pip", "install"]
                    for url in extra_index_urls:
                        uv_cmd.extend(["--extra-index-url", url])
                    uv_cmd.extend(["-r", str(temp_req)])
                    uv_cmd.extend(constraints_args)
                    subprocess.run(uv_cmd, check=True)
                    temp_req.unlink()
                else:
                    uv_cmd = ["uv", "pip", "install", "-r", str(requirements_file)]
                    uv_cmd.extend(constraints_args)
                    subprocess.run(uv_cmd, check=True)

            # Install additional dependencies if specified
            if "dependencies" in node_info:
                for dep in node_info["dependencies"]:
                    print(f"Installing dependency: {dep}")
                    uv_cmd = ["uv", "pip", "install"]
                    uv_cmd.extend(constraints_args)
                    uv_cmd.append(dep)
                    subprocess.run(uv_cmd, check=True)

            print(f"✓ Installed {node_info['name']}")
        except Exception as e:
            print(f"✗ Error installing {node_info['name']}: {e}")
            failed_nodes.append(node_info["name"])
            continue

    if failed_nodes:
        print(f"\nWarning: {len(failed_nodes)} node(s) failed to install:")
        for name in failed_nodes:
            print(f"  - {name}")


def setup_nodes():
    args = parse_args()
    workspace_dir = Path(args.workspace)

    # Resolve config path if provided
    config_path = None
    if args.config:
        config_path = Path(args.config)
        # If it's just a filename, look in configs directory
        if not config_path.is_absolute() and "/" not in str(config_path):
            config_path = Path("configs") / config_path
        if not config_path.exists():
            print(f"Error: Config file not found at {config_path}")
            sys.exit(1)

    setup_environment(workspace_dir)
    setup_directories(workspace_dir)
    install_custom_nodes(workspace_dir, config_path=config_path, pull_branches=args.pull_branches)


if __name__ == "__main__":
    setup_nodes()
