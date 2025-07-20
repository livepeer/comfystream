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
    level=logging.INFO,
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
            # __file__ is not defined when executed with exec()
            current_file = Path.cwd() / "prestartup_script.py"
        
        current_dir = current_file.parent
        logger.debug(f"Current directory: {current_dir}")
        
        # Method 1: Walk up the directory tree to find ComfyUI workspace
        for parent_dir in current_dir.parents:
            custom_nodes_path = parent_dir / "custom_nodes"
            if (custom_nodes_path.exists() and 
                ((parent_dir / "main.py").exists() or (parent_dir / "comfy").exists())):
                # Found ComfyUI workspace, check if we're in custom_nodes context
                is_in_custom_nodes = "custom_nodes" in str(current_dir)
                logger.debug(f"Found ComfyUI workspace via parent walk: {parent_dir}, in custom_nodes: {is_in_custom_nodes}")
                return parent_dir, is_in_custom_nodes
        
        # Method 2: Alternative approach for symlinked custom nodes
        # Look for ComfyUI workspace in the same parent directory as current workspace
        workspace_parent = current_dir.parent
        potential_comfyui = workspace_parent / "ComfyUI2"
        if (potential_comfyui.exists() and 
            (potential_comfyui / "custom_nodes").exists() and
            ((potential_comfyui / "main.py").exists() or (potential_comfyui / "comfy").exists())):
            # Found ComfyUI workspace via alternative path
            # Check if we're in a custom_nodes context by looking for our directory in custom_nodes
            custom_nodes_path = potential_comfyui / "custom_nodes"
            is_in_custom_nodes = False
            
            # Check for symlinked custom node
            for item in custom_nodes_path.iterdir():
                if item.is_symlink():
                    try:
                        if item.resolve() == current_dir:
                            is_in_custom_nodes = True
                            logger.debug(f"Found symlinked custom node: {item} -> {item.resolve()}")
                            break
                    except (OSError, RuntimeError) as e:
                        # Handle broken symlinks
                        logger.debug(f"Skipping broken symlink {item}: {e}")
                        continue
            
            # Also check if current directory name matches any custom node directory
            if not is_in_custom_nodes:
                current_dir_name = current_dir.name
                for item in custom_nodes_path.iterdir():
                    if item.name == current_dir_name:
                        is_in_custom_nodes = True
                        logger.debug(f"Found custom node by name match: {item}")
                        break
            
            logger.debug(f"Found ComfyUI workspace via alternative path: {potential_comfyui}, in custom_nodes: {is_in_custom_nodes}")
            return potential_comfyui, is_in_custom_nodes
        
        # Method 3: Check for common ComfyUI workspace locations
        common_locations = [
            Path("/workspace/ComfyUI"),
            Path("/workspace/ComfyUI2"),
            Path("/opt/ComfyUI"),
            Path.cwd().parent / "ComfyUI",
            Path.cwd().parent.parent / "ComfyUI",
        ]
        
        for location in common_locations:
            if (location.exists() and 
                (location / "custom_nodes").exists() and
                ((location / "main.py").exists() or (location / "comfy").exists())):
                logger.debug(f"Found ComfyUI workspace in common location: {location}")
                # For common locations, assume we're in custom_nodes context if we're in the right workspace
                is_in_custom_nodes = True
                return location, is_in_custom_nodes
        
        # Not found
        logger.debug("Could not find ComfyUI workspace")
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
        
        if workspace_path is not None:
            if is_in_custom_nodes:
                logger.info("Detected vanilla custom node loading method")
                return "vanilla"
            else:
                logger.info("Detected package loading method")
                return "package"
        else:
            # If we can't find ComfyUI workspace with custom_nodes, assume package loading
            logger.info("Could not detect ComfyUI workspace with custom_nodes, assuming package loading")
            return "package"
        
    except Exception as e:
        logger.warning(f"Error detecting loading method: {e}")
        return "package"  # Default to package loading for safety

def run_vanilla_initialization():
    """
    Run initialization tasks specific to vanilla custom node loading.
    This function only runs when ComfyStream is loaded as a vanilla custom node.
    """
    logger.info("Running vanilla custom node initialization...")
    
    try:
        # Get the ComfyUI workspace path using the reusable function
        comfyui_workspace, is_in_custom_nodes = find_comfyui_workspace()
        
        if comfyui_workspace is None:
            logger.warning("Could not find ComfyUI workspace for vanilla initialization")
            return
        
        if not is_in_custom_nodes:
            logger.warning("Not in custom_nodes context, skipping vanilla initialization")
            return
        
        logger.info(f"ComfyUI workspace found at: {comfyui_workspace}")
        
        # Set up environment variables for vanilla loading
        setup_vanilla_environment(comfyui_workspace)
        
        # Set up dynamic module loader for ComfyUI
        setup_dynamic_module_loader(comfyui_workspace)
        
        # Initialize any vanilla-specific components
        # initialize_vanilla_components()
        
        logger.info("Vanilla custom node initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error during vanilla initialization: {e}")

def setup_dynamic_module_loader(workspace_path):
    """
    Set up a dynamic module loader for ComfyUI that doesn't require persistent __init__.py files.
    This creates a custom meta path finder that can import ComfyUI modules on-demand.
    
    NOTE: This function is ONLY called during vanilla custom node loading. During package
    loading, the standard import system is used without any module redirection.
    
    Args:
        workspace_path (Path): Path to ComfyUI workspace
    """
    logger.info("Setting up dynamic module loader for ComfyUI...")
    
    try:
        import importlib
        import importlib.util
        import importlib.machinery
        from importlib.abc import MetaPathFinder, Loader
        
        class ComfyUIModuleFinder(MetaPathFinder):
            """Custom meta path finder for ComfyUI modules"""
            
            def __init__(self, workspace_path):
                self.workspace_path = Path(workspace_path)
                self.comfy_path = self.workspace_path / "comfy"
                self.comfy_extras_path = self.workspace_path / "comfy_extras"
                self.comfy_execution_path = self.workspace_path / "comfy_execution"
            
            def find_spec(self, fullname, path, target=None):
                """Find module spec for ComfyUI modules"""
                
                # Handle comfy.* modules
                if fullname.startswith('comfy.') or fullname == 'comfy':
                    spec = self._find_comfy_spec(fullname)
                    # Only return a spec if we found the module in the workspace
                    if spec is not None:
                        return spec
                
                # Handle comfy_extras.* modules
                if fullname.startswith('comfy_extras.') or fullname == 'comfy_extras':
                    spec = self._find_comfy_extras_spec(fullname)
                    # Only return a spec if we found the module in the workspace
                    if spec is not None:
                        return spec
                
                # Handle comfy_execution.* modules
                if fullname.startswith('comfy_execution.') or fullname == 'comfy_execution':
                    spec = self._find_comfy_execution_spec(fullname)
                    # Only return a spec if we found the module in the workspace
                    if spec is not None:
                        return spec
                
                # Let other imports fall through to the standard import system
                return None
            
            def _find_comfy_spec(self, fullname):
                """Find spec for comfy.* modules"""
                # Only handle submodules, not the base 'comfy' package
                # This allows the installed 'comfy' package to be imported normally
                if fullname == 'comfy':
                    return None  # Let the standard import system handle this
                
                elif fullname.startswith('comfy.'):
                    parts = fullname.split('.')
                    submodule_parts = parts[1:]  # Remove 'comfy' prefix
                    
                    # Build the path to the submodule
                    submodule_path = self.comfy_path
                    for part in submodule_parts:
                        submodule_path = submodule_path / part
                    
                    # Only handle modules that exist in the ComfyUI workspace
                    # Try to find as a directory with __init__.py (package)
                    if submodule_path.is_dir():
                        init_file = submodule_path / "__init__.py"
                        if init_file.exists():
                            # This is a package with __init__.py
                            try:
                                loader = ComfyUIModuleLoader(str(init_file))
                                if loader is not None:
                                    return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                                else:
                                    logger.warning(f"Failed to create loader for {fullname}")
                            except Exception as e:
                                logger.warning(f"Error creating loader for {fullname}: {e}")
                        else:
                            # This is a package directory without __init__.py
                            try:
                                loader = ComfyUIPackageLoader(str(submodule_path))
                                if loader is not None:
                                    return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                                else:
                                    logger.warning(f"Failed to create loader for {fullname}")
                            except Exception as e:
                                logger.warning(f"Error creating loader for {fullname}: {e}")
                    
                    # Try to find as a .py file
                    py_file = submodule_path.with_suffix('.py')
                    if py_file.is_file():
                        try:
                            loader = ComfyUIModuleLoader(str(py_file))
                            if loader is not None:
                                return importlib.machinery.ModuleSpec(fullname, loader)
                            else:
                                logger.warning(f"Failed to create loader for {fullname}")
                        except Exception as e:
                            logger.warning(f"Error creating loader for {fullname}: {e}")
                
                # Return None if module not found in workspace - let standard import handle it
                return None
            
            def _find_comfy_execution_spec(self, fullname):
                """Find spec for comfy_execution.* modules"""
                
                # Redirection mapping for comfy_execution.* to comfy.*
                # Only include modules that have actually been moved to the comfy package
                redirection_map = {
                    'comfy_execution.validation': 'comfy.validation',
                    # Add other modules here only if they have been moved to comfy package
                    # Most comfy_execution modules should remain in comfy_execution
                }
                
                # Check if this module needs redirection
                if fullname in redirection_map:
                    target_module = redirection_map[fullname]
                    try:
                        # Try to import from the target location
                        importlib.import_module(target_module)
                        logger.info(f"Redirecting {fullname} to {target_module}")
                        loader = ModuleRedirectLoader(target_module)
                        if loader is not None:
                            return importlib.machinery.ModuleSpec(fullname, loader)
                        else:
                            logger.warning(f"Failed to create redirect loader for {fullname}")
                    except ImportError as e:
                        logger.debug(f"Target module {target_module} not available: {e}")
                        # Fall through to normal handling
                        pass
                    except Exception as e:
                        logger.warning(f"Error creating redirect loader for {fullname}: {e}")
                        # Fall through to normal handling
                        pass
                
                # Handle the base comfy_execution package
                if fullname == 'comfy_execution':
                    if self.comfy_execution_path.exists():
                        try:
                            loader = ComfyUIPackageLoader(str(self.comfy_execution_path))
                            if loader is not None:
                                return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                            else:
                                logger.warning(f"Failed to create loader for {fullname}")
                        except Exception as e:
                            logger.warning(f"Error creating loader for {fullname}: {e}")
                
                elif fullname.startswith('comfy_execution.'):
                    parts = fullname.split('.')
                    submodule_parts = parts[1:]  # Remove 'comfy_execution' prefix
                    
                    # Build the path to the submodule
                    submodule_path = self.comfy_execution_path
                    for part in submodule_parts:
                        submodule_path = submodule_path / part
                    
                    # Only handle modules that exist in the ComfyUI workspace
                    # Try to find as a directory with __init__.py (package)
                    if submodule_path.is_dir():
                        init_file = submodule_path / "__init__.py"
                        if init_file.exists():
                            # This is a package with __init__.py
                            try:
                                loader = ComfyUIModuleLoader(str(init_file))
                                if loader is not None:
                                    return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                                else:
                                    logger.warning(f"Failed to create loader for {fullname}")
                            except Exception as e:
                                logger.warning(f"Error creating loader for {fullname}: {e}")
                        else:
                            # This is a package directory without __init__.py
                            try:
                                loader = ComfyUIPackageLoader(str(submodule_path))
                                if loader is not None:
                                    return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                                else:
                                    logger.warning(f"Failed to create loader for {fullname}")
                            except Exception as e:
                                logger.warning(f"Error creating loader for {fullname}: {e}")
                    
                    # Try to find as a .py file
                    py_file = submodule_path.with_suffix('.py')
                    if py_file.is_file():
                        try:
                            loader = ComfyUIModuleLoader(str(py_file))
                            if loader is not None:
                                return importlib.machinery.ModuleSpec(fullname, loader)
                            else:
                                logger.warning(f"Failed to create loader for {fullname}")
                        except Exception as e:
                            logger.warning(f"Error creating loader for {fullname}: {e}")
                
                # Return None if module not found in workspace - let standard import handle it
                return None
            
            def _find_comfy_extras_spec(self, fullname):
                """Find spec for comfy_extras.* modules"""
                if fullname == 'comfy_extras':
                    if self.comfy_extras_path.exists():
                        try:
                            loader = ComfyUIPackageLoader(str(self.comfy_extras_path))
                            if loader is not None:
                                return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                            else:
                                logger.warning(f"Failed to create loader for {fullname}")
                        except Exception as e:
                            logger.warning(f"Error creating loader for {fullname}: {e}")
                
                elif fullname.startswith('comfy_extras.'):
                    parts = fullname.split('.')
                    submodule_parts = parts[1:]  # Remove 'comfy_extras' prefix
                    
                    # Build the path to the submodule
                    submodule_path = self.comfy_extras_path
                    for part in submodule_parts:
                        submodule_path = submodule_path / part
                    
                    # Try to find as a directory with __init__.py (package)
                    if submodule_path.is_dir():
                        init_file = submodule_path / "__init__.py"
                        if init_file.exists():
                            # This is a package with __init__.py
                            try:
                                loader = ComfyUIModuleLoader(str(init_file))
                                if loader is not None:
                                    return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                                else:
                                    logger.warning(f"Failed to create loader for {fullname}")
                            except Exception as e:
                                logger.warning(f"Error creating loader for {fullname}: {e}")
                        else:
                            # This is a package directory without __init__.py
                            try:
                                loader = ComfyUIPackageLoader(str(submodule_path))
                                if loader is not None:
                                    return importlib.machinery.ModuleSpec(fullname, loader, is_package=True)
                                else:
                                    logger.warning(f"Failed to create loader for {fullname}")
                            except Exception as e:
                                logger.warning(f"Error creating loader for {fullname}: {e}")
                    
                    # Try to find as a .py file
                    py_file = submodule_path.with_suffix('.py')
                    if py_file.is_file():
                        try:
                            loader = ComfyUIModuleLoader(str(py_file))
                            if loader is not None:
                                return importlib.machinery.ModuleSpec(fullname, loader)
                            else:
                                logger.warning(f"Failed to create loader for {fullname}")
                        except Exception as e:
                            logger.warning(f"Error creating loader for {fullname}: {e}")
                
                return None
        
        class ModuleRedirectLoader(Loader):
            """Loader that redirects imports to a different module"""
            
            def __init__(self, target_module_name):
                self.target_module_name = target_module_name
            
            def create_module(self, spec):
                """Create a new module for the given spec"""
                return None  # Use default module creation
            
            def exec_module(self, module):
                """Execute the module by importing the target and copying its contents"""
                # Import the target module
                target_module = importlib.import_module(self.target_module_name)
                
                # Copy all attributes from the target module to this module
                for attr_name in dir(target_module):
                    if not attr_name.startswith('_'):
                        setattr(module, attr_name, getattr(target_module, attr_name))
                
                # Set module metadata
                module.__file__ = getattr(target_module, '__file__', None)
                module.__package__ = module.__name__.rpartition('.')[0]

        class ComfyUIPackageLoader(Loader):
            """Loader for ComfyUI packages (directories)"""
            
            def __init__(self, path):
                self.path = path
            
            def create_module(self, spec):
                """Create a new module for the given spec"""
                return None  # Use default module creation
            
            def exec_module(self, module):
                """Execute the module"""
                module.__path__ = [self.path]
                module.__package__ = module.__name__
        
        class ComfyUIModuleLoader(Loader):
            """Loader for ComfyUI modules (.py files)"""
            
            def __init__(self, path):
                self.path = path
            
            def create_module(self, spec):
                """Create a new module for the given spec"""
                return None  # Use default module creation
            
            def exec_module(self, module):
                """Execute the module"""
                # Set __file__ attribute for the module
                module.__file__ = self.path
                
                # If this is a package (loading __init__.py), set up __path__
                if self.path.endswith('__init__.py'):
                    module.__path__ = [str(Path(self.path).parent)]
                    module.__package__ = module.__name__
                
                # Load and execute the module
                spec = importlib.util.spec_from_file_location(module.__name__, self.path)
                if spec and spec.loader:
                    spec.loader.exec_module(module)
        
        # Install the custom finder
        finder = ComfyUIModuleFinder(workspace_path)
        if finder not in sys.meta_path:
            sys.meta_path.insert(0, finder)
            logger.info("Dynamic module loader installed successfully")
        
    except Exception as e:
        logger.error(f"Error setting up dynamic module loader: {e}")
        # Fallback to traditional approach if dynamic loading fails
        logger.info("Falling back to traditional __init__.py approach")
        ensure_comfyui_init_files_fallback(workspace_path)

def ensure_comfyui_init_files_fallback(workspace_path):
    """
    Fallback method: Create temporary __init__.py files for ComfyUI directories.
    These are marked as temporary and can be cleaned up later.
    
    Args:
        workspace_path (Path): Path to ComfyUI workspace
    """
    logger.info("Creating temporary __init__.py files for ComfyUI directories...")
    
    try:
        # Directories that need __init__.py files
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
                logger.info(f"Created temporary __init__.py at {root_init}")
                
            # Then walk subdirectories
            for root, dirs, files in os.walk(base_path):
                init_path = Path(root) / "__init__.py"
                if not init_path.exists():
                    init_path.write_text(temp_marker + "# This file ensures the directory is treated as a package\n")
                    logger.debug(f"Created temporary __init__.py at {init_path}")
                
    except Exception as e:
        logger.error(f"Error creating temporary __init__.py files: {e}")

def setup_vanilla_environment(workspace_path):
    """
    Set up environment variables and paths for vanilla custom node loading.
    
    Args:
        workspace_path (Path): Path to ComfyUI workspace
    """
    logger.info("Setting up environment for vanilla custom node loading...")
    
    try:
        # Add ComfyUI workspace to Python path
        workspace_str = str(workspace_path)
        if workspace_str not in sys.path:
            sys.path.insert(0, workspace_str)
            logger.info(f"Added {workspace_str} to Python path")
        
        # Set environment variables
        os.environ['COMFY_UI_WORKSPACE'] = workspace_str
        os.environ['PYTHONPATH'] = workspace_str
        os.environ['CUSTOM_NODES_PATH'] = str(workspace_path / "custom_nodes")
        
        logger.info("Environment variables set for vanilla loading")
        
    except Exception as e:
        logger.error(f"Error setting up vanilla environment: {e}")

def cleanup_temporary_files(workspace_path):
    """
    Clean up temporary __init__.py files created during fallback loading.
    
    Args:
        workspace_path (Path): Path to ComfyUI workspace
    """
    logger.info("Cleaning up temporary files...")
    
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
            
            # Walk all directories and remove temporary __init__.py files
            for root, dirs, files in os.walk(base_path):
                init_path = Path(root) / "__init__.py"
                if init_path.exists():
                    try:
                        content = init_path.read_text()
                        # Check if this is a temporary file created by ComfyStream
                        if any(marker in content for marker in temp_markers):
                            init_path.unlink()
                            logger.debug(f"Removed temporary __init__.py at {init_path}")
                    except Exception as e:
                        logger.debug(f"Could not remove {init_path}: {e}")
        
        logger.info("Temporary file cleanup completed")
        
    except Exception as e:
        logger.debug(f"Error during cleanup: {e}")

def initialize_vanilla_components():
    """
    Initialize any components specific to vanilla custom node loading.
    """
    logger.info("Initializing vanilla-specific components...")
    
    try:
        # Import and initialize any vanilla-specific modules
        # This is where you would add any initialization code specific to vanilla loading
        
        # Example: Initialize tensor cache for vanilla loading
        try:
            from comfystream.tensor_cache import image_inputs, image_outputs, audio_inputs, audio_outputs
            logger.info("Tensor cache components initialized for vanilla loading")
        except ImportError as e:
            logger.warning(f"Could not import tensor cache components: {e}")
        
        # Example: Set up any vanilla-specific logging
        logger.info("Vanilla-specific components initialized")
        
    except Exception as e:
        logger.error(f"Error initializing vanilla components: {e}")

def setup_package_comfyui_modules(workspace_path):
    """
    Set up ComfyUI execution modules for package loading.
    This ensures comfy_execution and other workspace modules are available during package loading.
    
    Args:
        workspace_path (Path): Path to ComfyUI workspace
    """
    logger.info("Setting up ComfyUI execution modules for package loading...")
    
    try:
        workspace_path = Path(workspace_path)
        
        # List of ComfyUI module directories that need to be available
        comfyui_modules = ['comfy_execution', 'comfy_extras']
        
        for module_name in comfyui_modules:
            module_path = workspace_path / module_name
            
            if module_path.exists() and module_path.is_dir():
                # Add the module directory to Python path
                module_str = str(module_path)
                if module_str not in sys.path:
                    sys.path.insert(0, module_str)
                    logger.info(f"Added {module_name} module to Python path: {module_str}")
                
                # Ensure the module has an __init__.py file for proper package recognition
                init_file = module_path / "__init__.py"
                if not init_file.exists():
                    try:
                        init_file.write_text("# Package initialization file for ComfyUI module\n")
                        logger.info(f"Created __init__.py for {module_name} module")
                    except Exception as e:
                        logger.warning(f"Could not create __init__.py for {module_name}: {e}")
                
                # Also add the parent workspace to sys.path if not already there
                # This ensures imports like "from comfy_execution.validation import ..." work
                parent_path = str(workspace_path)
                if parent_path not in sys.path:
                    sys.path.insert(0, parent_path)
        
        logger.info("ComfyUI execution modules setup completed for package loading")
        
    except Exception as e:
        logger.error(f"Error setting up ComfyUI execution modules: {e}")

def run_package_initialization():
    """
    Run initialization tasks specific to package loading.
    This function runs when ComfyStream is loaded as an installable package.
    """
    logger.info("Running package initialization...")
    
    try:
        # Get the ComfyUI workspace path using the reusable function
        comfyui_workspace, is_in_custom_nodes = find_comfyui_workspace()
        
        if comfyui_workspace is not None:
            logger.info(f"ComfyUI workspace found at: {comfyui_workspace}")
            
            # Add ComfyUI workspace to Python path for package loading
            workspace_str = str(comfyui_workspace)
            if workspace_str not in sys.path:
                sys.path.insert(0, workspace_str)
                logger.info(f"Added {workspace_str} to Python path for package loading")
            
            # Set up ComfyUI execution modules for package loading
            setup_package_comfyui_modules(comfyui_workspace)
            
            # Set environment variables
            os.environ['COMFY_UI_WORKSPACE'] = workspace_str
            os.environ['PYTHONPATH'] = workspace_str
            
            logger.info("Environment set up for package loading")
        else:
            logger.warning("Could not find ComfyUI workspace for package initialization")
        
    except Exception as e:
        logger.error(f"Error during package initialization: {e}")

def main():
    """
    Main function that runs the pre-startup script.
    This function executes initialization tasks for both loading methods.
    """
    logger.info("ComfyStream pre-startup script starting...")
    
    try:
        # Clean up any leftover temporary files from previous runs
        workspace_path, _ = find_comfyui_workspace()
        if workspace_path:
            cleanup_temporary_files(workspace_path)
        
        # Detect the loading method
        loading_method = detect_loading_method()
        
        if loading_method == "vanilla":
            logger.info("ComfyStream detected as vanilla custom node - running initialization")
            run_vanilla_initialization()
        else:
            logger.info("ComfyStream detected as installable package - running package initialization")
            run_package_initialization()
    
    except Exception as e:
        logger.error(f"Error during pre-startup script execution: {e}")
    
    finally:
        # Clean up any temporary files created during this run
        try:
            workspace_path, _ = find_comfyui_workspace()
            if workspace_path:
                cleanup_temporary_files(workspace_path)
        except Exception as e:
            logger.debug(f"Error during final cleanup: {e}")
    
    logger.info("ComfyStream pre-startup script completed")

# Run the main function when this script is executed
if __name__ == "__main__":
    main()
else:
    # When imported as a module, still run the main function
    main()
