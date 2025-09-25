import yaml
import os
from pathlib import Path

def get_config_path(filename):
    """Get the absolute path to a config file with pattern matching support"""
    configs_dir = Path("configs")
    
    if not configs_dir.exists():
        print("  configs/ directory not found")
        raise FileNotFoundError("configs/ directory not found")
    
    # First try exact match
    config_path = configs_dir / filename
    if config_path.exists():
        return config_path
    
    # If no extension provided, try adding .yaml
    if not filename.endswith('.yaml'):
        config_path = configs_dir / f"{filename}.yaml"
        if config_path.exists():
            return config_path
    
    # Try pattern matching for nodes-* files
    if not filename.startswith('nodes-') and not filename == 'nodes.yaml':
        pattern_path = configs_dir / f"nodes-{filename}.yaml"
        if pattern_path.exists():
            return pattern_path
    
    # If still not found, show available files
    print(f"Warning: Config file matching '{filename}' not found")
    raise FileNotFoundError(f"Config file matching '{filename}' not found")

def load_model_config(config_path):
    """Load model configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_default_workspace():
    """Get the default workspace directory"""
    return os.environ.get("COMFYUI_WORKSPACE", Path("~/comfyui").expanduser())

def validate_and_prompt_workspace(workspace_path, script_name="script"):
    """
    Validate workspace directory exists and prompt user to create if it doesn't.
    
    Args:
        workspace_path: Path to workspace directory (str or Path object)
        script_name: Name of the calling script for better error messages
    
    Returns:
        Path: Validated workspace directory path
        
    Raises:
        SystemExit: If user cancels workspace creation
    """
    workspace_dir = Path(workspace_path)
    
    # Check if workspace exists, and prompt user if it doesn't
    if not workspace_dir.exists():
        print(f"Workspace directory '{workspace_dir}' does not exist.")
        
        # Check if this is the default workspace (user didn't specify one)
        default_workspace = get_default_workspace()
        if str(workspace_dir) == str(default_workspace):
            print("No workspace was specified and the default workspace doesn't exist.")
            
        try:
            response = input(f"Would you like to create '{workspace_dir}' and continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print(f"{script_name} cancelled.")
                raise SystemExit(0)
        except (KeyboardInterrupt, EOFError):
            print(f"\n{script_name} cancelled.")
            raise SystemExit(0)
    
    return workspace_dir

def setup_workspace_environment(workspace_dir):
    """Setup environment variables for workspace"""
    os.environ["COMFYUI_WORKSPACE"] = str(workspace_dir)
    os.environ["CUSTOM_NODES_PATH"] = str(workspace_dir / "custom_nodes")
