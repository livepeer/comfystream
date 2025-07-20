"""
ComfyStream Pre-Startup Script
This script only runs when ComfyStream is loaded as a vanilla custom node.
It will NOT run when ComfyStream is loaded as an installable package.

Workspace Detection:
- Supports both direct custom_nodes installation and symlinked custom nodes
- Automatically detects ComfyUI workspace location
- Handles various deployment scenarios (local, container, etc.)
- Uses multiple detection methods for robustness
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='[ComfyStream PreStartup] %(message)s'
)
logger = logging.getLogger(__name__)

def find_comfyui_workspace():
    """
    Find the ComfyUI workspace that contains the custom_nodes directory.
    
    Returns:
        tuple: (workspace_path, is_in_custom_nodes) where workspace_path is the Path to ComfyUI workspace
               and is_in_custom_nodes is a boolean indicating if we're in a custom_nodes context
    """
    try:
        # Get the current file's directory
        try:
            current_file = Path(__file__).resolve()
        except NameError:
            current_file = Path.cwd() / "prestartup_script.py"
        
        current_dir = current_file.parent
        
        # Method 1: Walk up the directory tree to find ComfyUI workspace
        for parent_dir in current_dir.parents:
            custom_nodes_path = parent_dir / "custom_nodes"
            if (custom_nodes_path.exists() and 
                ((parent_dir / "main.py").exists() or (parent_dir / "comfy").exists())):
                is_in_custom_nodes = "custom_nodes" in str(current_dir)
                return parent_dir, is_in_custom_nodes
        
        # Method 2: Alternative approach for symlinked custom nodes
        workspace_parent = current_dir.parent
        potential_comfyui = workspace_parent / "ComfyUI"
        if (potential_comfyui.exists() and 
            (potential_comfyui / "custom_nodes").exists() and
            ((potential_comfyui / "main.py").exists() or (potential_comfyui / "comfy").exists())):
            
            custom_nodes_path = potential_comfyui / "custom_nodes"
            is_in_custom_nodes = False
            
            # Check for symlinked custom node
            for item in custom_nodes_path.iterdir():
                if item.is_symlink():
                    try:
                        if item.resolve() == current_dir:
                            is_in_custom_nodes = True
                            break
                    except (OSError, RuntimeError):
                        continue
            
            # Also check if current directory name matches any custom node directory
            if not is_in_custom_nodes:
                current_dir_name = current_dir.name
                for item in custom_nodes_path.iterdir():
                    if item.name == current_dir_name:
                        is_in_custom_nodes = True
                        break
            
            return potential_comfyui, is_in_custom_nodes
        
        # Method 3: Check for common ComfyUI workspace locations
        common_locations = [
            Path("/workspace/ComfyUI"),
            Path("/opt/ComfyUI"),
            Path.cwd().parent / "ComfyUI",
            Path.cwd().parent.parent / "ComfyUI",
        ]
        
        for location in common_locations:
            if (location.exists() and 
                (location / "custom_nodes").exists() and
                ((location / "main.py").exists() or (location / "comfy").exists())):
                return location, True
        
        return None, False
        
    except Exception as e:
        logger.warning(f"Error finding ComfyUI workspace: {e}")
        return None, False

def detect_loading_method():
    """
    Detect whether ComfyStream is being loaded as a vanilla custom node or as an installable package.
    
    Returns:
        str: 'vanilla' if loaded as vanilla custom node, 'package' if loaded as installable package
    """
    try:
        workspace_path, is_in_custom_nodes = find_comfyui_workspace()
        
        if workspace_path is not None and is_in_custom_nodes:
            return "vanilla"
        else:
            return "package"
        
    except Exception:
        return "package"  # Default to package loading for safety

def setup_dynamic_module_loader(workspace_path):
    """
    Set up a dynamic module loader for ComfyUI that doesn't require persistent __init__.py files.
    """
    try:
        import importlib
        import importlib.util
        import importlib.machinery
        from importlib.abc import MetaPathFinder, Loader
        
        class ComfyUIModuleFinder(MetaPathFinder):
            """Custom meta path finder for ComfyUI modules"""
            
            def __init__(self, workspace_path):
                self.workspace_path = Path(workspace_path)
                self.module_paths = {
                    'comfy': self.workspace_path / "comfy",
                    'comfy_extras': self.workspace_path / "comfy_extras",
                    'comfy_execution': self.workspace_path / "comfy_execution"
                }
                
                # Redirection mapping for moved modules
                self.redirection_map = {
                    'comfy_execution.validation': 'comfy.validation',
                }
            
            def find_spec(self, fullname, path, target=None):
                """Find module spec for ComfyUI modules"""
                # Handle redirected modules
                if fullname in self.redirection_map:
                    target_module = self.redirection_map[fullname]
                    try:
                        importlib.import_module(target_module)
                        loader = ModuleRedirectLoader(target_module)
                        return importlib.machinery.ModuleSpec(fullname, loader)
                    except ImportError:
                        pass
                
                # Handle comfy.*, comfy_extras.*, comfy_execution.* modules
                for prefix, base_path in self.module_paths.items():
                    if fullname == prefix:
                        if base_path.exists():
                            loader = ComfyUIPackageLoader(str(base_path))
                            return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                    elif fullname.startswith(f'{prefix}.'):
                        # Skip base 'comfy' package to allow standard import
                        if prefix == 'comfy' and fullname == 'comfy':
                            continue
                            
                        spec = self._find_submodule_spec(fullname, prefix, base_path)
                        if spec:
                            return spec
                
                return None
            
            def _find_submodule_spec(self, fullname, prefix, base_path):
                """Find spec for a submodule within a ComfyUI package"""
                parts = fullname.split('.')
                submodule_parts = parts[1:]  # Remove prefix
                
                # Build the path to the submodule
                submodule_path = base_path
                for part in submodule_parts:
                    submodule_path = submodule_path / part
                
                # Try to find as a directory with __init__.py (package)
                if submodule_path.is_dir():
                    init_file = submodule_path / "__init__.py"
                    if init_file.exists():
                        loader = ComfyUIModuleLoader(str(init_file))
                        return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                    else:
                        loader = ComfyUIPackageLoader(str(submodule_path))
                        return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                
                # Try to find as a .py file
                py_file = submodule_path.with_suffix('.py')
                if py_file.is_file():
                    loader = ComfyUIModuleLoader(str(py_file))
                    return importlib.machinery.ModuleSpec(fullname, loader)
                
                return None
        
        class ModuleRedirectLoader(Loader):
            """Loader that redirects imports to a different module"""
            
            def __init__(self, target_module_name):
                self.target_module_name = target_module_name
            
            def create_module(self, spec):
                return None
            
            def exec_module(self, module):
                target_module = importlib.import_module(self.target_module_name)
                for attr_name in dir(target_module):
                    if not attr_name.startswith('_'):
                        setattr(module, attr_name, getattr(target_module, attr_name))
                module.__file__ = getattr(target_module, '__file__', None)
                module.__package__ = module.__name__.rpartition('.')[0]

        class ComfyUIPackageLoader(Loader):
            """Loader for ComfyUI packages (directories)"""
            
            def __init__(self, path):
                self.path = path
            
            def create_module(self, spec):
                return None
            
            def exec_module(self, module):
                module.__path__ = [self.path]
                module.__package__ = module.__name__
        
        class ComfyUIModuleLoader(Loader):
            """Loader for ComfyUI modules (.py files)"""
            
            def __init__(self, path):
                self.path = path
            
            def create_module(self, spec):
                return None
            
            def exec_module(self, module):
                module.__file__ = self.path
                
                if self.path.endswith('__init__.py'):
                    module.__path__ = [str(Path(self.path).parent)]
                    module.__package__ = module.__name__
                
                spec = importlib.util.spec_from_file_location(module.__name__, self.path)
                if spec and spec.loader:
                    spec.loader.exec_module(module)
        
        # Install the custom finder
        finder = ComfyUIModuleFinder(workspace_path)
        if finder not in sys.meta_path:
            sys.meta_path.insert(0, finder)
        
    except Exception as e:
        logger.error(f"Error setting up dynamic module loader: {e}")
        # Fallback to traditional approach if dynamic loading fails
        ensure_comfyui_init_files_fallback(workspace_path)

def ensure_comfyui_init_files_fallback(workspace_path):
    """
    Fallback method: Create temporary __init__.py files for ComfyUI directories.
    """
    try:
        base_dirs = ['comfy', 'comfy_extras']
        temp_marker = "# TEMPORARY FILE - Created by ComfyStream dynamic loader\n"
        
        for base_dir in base_dirs:
            base_path = workspace_path / base_dir
            if not base_path.exists():
                continue
                
            # Create __init__.py in the root of base_dir first
            root_init = base_path / "__init__.py"
            if not root_init.exists():
                root_init.write_text(temp_marker + "# This file ensures comfy is treated as a package\n")
                
            # Then walk subdirectories
            for root, dirs, files in os.walk(base_path):
                init_path = Path(root) / "__init__.py"
                if not init_path.exists():
                    init_path.write_text(temp_marker + "# This file ensures the directory is treated as a package\n")
                
    except Exception as e:
        logger.error(f"Error creating temporary __init__.py files: {e}")

def setup_environment(workspace_path):
    """
    Set up environment variables and paths for ComfyUI workspace.
    """
    try:
        # Add ComfyUI workspace to Python path
        workspace_str = str(workspace_path)
        if workspace_str not in sys.path:
            sys.path.insert(0, workspace_str)
        
        # Set environment variables
        os.environ['COMFY_UI_WORKSPACE'] = workspace_str
        os.environ['PYTHONPATH'] = workspace_str
        os.environ['CUSTOM_NODES_PATH'] = str(workspace_path / "custom_nodes")
        
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")

def setup_comfyui_modules(workspace_path):
    """
    Set up ComfyUI execution modules for package loading.
    """
    try:
        workspace_path = Path(workspace_path)
        comfyui_modules = ['comfy_execution', 'comfy_extras']
        
        for module_name in comfyui_modules:
            module_path = workspace_path / module_name
            
            if module_path.exists() and module_path.is_dir():
                # Add the module directory to Python path
                module_str = str(module_path)
                if module_str not in sys.path:
                    sys.path.insert(0, module_str)
                
                # Ensure the module has an __init__.py file for proper package recognition
                init_file = module_path / "__init__.py"
                if not init_file.exists():
                    try:
                        init_file.write_text("# Package initialization file for ComfyUI module\n")
                    except Exception:
                        pass
                
                # Also add the parent workspace to sys.path if not already there
                parent_path = str(workspace_path)
                if parent_path not in sys.path:
                    sys.path.insert(0, parent_path)
        
    except Exception as e:
        logger.error(f"Error setting up ComfyUI execution modules: {e}")

def cleanup_temporary_files(workspace_path):
    """
    Clean up temporary __init__.py files created during fallback loading.
    """
    try:
        temp_markers = [
            "# TEMPORARY FILE - Created by ComfyStream dynamic loader",
            "# This file ensures comfy is treated as a package",
            "# This file ensures the directory is treated as a package"
        ]
        base_dirs = ['comfy', 'comfy_extras']
        
        for base_dir in base_dirs:
            base_path = workspace_path / base_dir
            if not base_path.exists():
                continue
            
            for root, dirs, files in os.walk(base_path):
                init_path = Path(root) / "__init__.py"
                if init_path.exists():
                    try:
                        content = init_path.read_text()
                        if any(marker in content for marker in temp_markers):
                            init_path.unlink()
                    except Exception:
                        pass
        
    except Exception:
        pass

def run_vanilla_initialization():
    """
    Run initialization tasks specific to vanilla custom node loading.
    """
    try:
        comfyui_workspace, is_in_custom_nodes = find_comfyui_workspace()
        
        if comfyui_workspace is None or not is_in_custom_nodes:
            logger.warning("Could not find ComfyUI workspace for vanilla initialization")
            return
        
        setup_environment(comfyui_workspace)
        setup_dynamic_module_loader(comfyui_workspace)
        
    except Exception as e:
        logger.error(f"Error during vanilla initialization: {e}")

def run_package_initialization():
    """
    Run initialization tasks specific to package loading.
    """
    try:
        comfyui_workspace, _ = find_comfyui_workspace()
        
        if comfyui_workspace is not None:
            setup_environment(comfyui_workspace)
            setup_comfyui_modules(comfyui_workspace)
        else:
            logger.warning("Could not find ComfyUI workspace for package initialization")
        
    except Exception as e:
        logger.error(f"Error during package initialization: {e}")

def main():
    """
    Main function that runs the pre-startup script.
    """
    try:
        # Clean up any leftover temporary files from previous runs
        workspace_path, _ = find_comfyui_workspace()
        if workspace_path:
            cleanup_temporary_files(workspace_path)
        
        # Detect the loading method and run appropriate initialization
        loading_method = detect_loading_method()
        
        if loading_method == "vanilla":
            run_vanilla_initialization()
        else:
            run_package_initialization()
    
    except Exception as e:
        logger.error(f"Error during pre-startup script execution: {e}")
    
    finally:
        # Clean up any temporary files created during this run
        try:
            workspace_path, _ = find_comfyui_workspace()
            if workspace_path:
                cleanup_temporary_files(workspace_path)
        except Exception:
            pass

# Run the main function when this script is executed
if __name__ == "__main__":
    main()
else:
    # When imported as a module, still run the main function
    main()
