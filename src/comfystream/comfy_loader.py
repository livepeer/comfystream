"""
Dedicated ComfyUI module loader with advanced features
"""
import sys
import importlib
import importlib.util
import pkgutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ComfyModuleLoader:
    """Advanced ComfyUI module loader with caching and override support"""
    
    def __init__(self, comfyui_path: Optional[str] = None, override_utils_path: Optional[str] = None):
        self.comfyui_path = Path(comfyui_path) if comfyui_path else self._get_default_comfyui_path()
        self.override_utils_path = Path(override_utils_path) if override_utils_path else None
        self._loaded_modules: Dict[str, Any] = {}
        self._discovered_modules: List[str] = []
        
    def _get_default_comfyui_path(self) -> Path:
        """Get default ComfyUI path"""
        # Get the path relative to this comfy_loader.py file
        # src/comfystream/comfy_loader.py -> ../../external
        loader_file = Path(__file__).resolve()
        comfystream_root = loader_file.parent.parent.parent  # Go up 3 levels: comfystream/ <- src/ <- comfystream/
        external_path = comfystream_root / "external"
        logger.debug(f"Computed external path: {external_path}")
        return external_path
    
    def ensure_comfyui_available(self) -> bool:
        """Ensure ComfyUI is available (clone if needed)"""
        comfy_path = self.comfyui_path / "comfy"
        if not comfy_path.exists():
            logger.warning(f"ComfyUI comfy path does not exist: {comfy_path}")
            return False
        else: 
            logger.debug(f"ComfyUI comfy path exists: {comfy_path}")
            return True
    
    def setup_python_path(self) -> None:
        """Setup Python path for ComfyUI imports"""
        if not self.ensure_comfyui_available():
            raise ImportError(f"ComfyUI not available at {self.comfyui_path}")
        
        # Add the external directory to Python path so 'import comfy' works
        external_path = str(self.comfyui_path)
        if external_path not in sys.path:
            sys.path.insert(0, external_path)
            logger.debug(f"Added {external_path} to Python path")
        
        # Verify that comfy can be imported
        try:
            # Use importlib to test the import without triggering linting errors
            comfy_spec = importlib.util.find_spec("comfy")
            if comfy_spec is None:
                raise ImportError("comfy module spec not found")
            logger.debug(f"Successfully verified comfy module spec from {external_path}")
        except ImportError as e:
            logger.error(f"Cannot import comfy from {external_path}: {e}")
            # Try to add some diagnostic info
            comfy_path = self.comfyui_path / "comfy"
            if comfy_path.exists():
                init_file = comfy_path / "__init__.py"
                logger.debug(f"Comfy directory exists at {comfy_path}")
                logger.debug(f"__init__.py exists: {init_file.exists()}")
            raise ImportError(f"Failed to import comfy module from {external_path}: {e}")
    
    def discover_comfy_modules(self) -> List[str]:
        """Dynamically discover all available comfy modules"""
        if self._discovered_modules:
            return self._discovered_modules
        
        self.setup_python_path()
        
        discovered = []
        comfy_path = self.comfyui_path / "comfy"
        
        if not comfy_path.exists():
            logger.error(f"Comfy path does not exist: {comfy_path}")
            return discovered
        
        # Add the main comfy module
        discovered.append("comfy")
        
        try:
            import os
            # Walk through the comfy directory to find all modules
            for root, dirs, files in os.walk(comfy_path):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                
                root_path = Path(root)
                
                # Check if this directory has an __init__.py (making it a package)
                if (root_path / "__init__.py").exists() and root_path != comfy_path:
                    # Convert path to module name
                    relative_path = root_path.relative_to(comfy_path.parent)
                    module_name = ".".join(relative_path.parts)
                    discovered.append(module_name)
                
                # Check for individual .py files
                for file in files:
                    if file.endswith(".py") and file != "__init__.py":
                        file_path = root_path / file
                        relative_path = file_path.relative_to(comfy_path.parent)
                        # Remove .py extension and convert to module name
                        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
                        module_name = ".".join(module_parts)
                        discovered.append(module_name)
            
        except Exception as e:
            logger.error(f"Error discovering comfy modules: {e}")
        
        self._discovered_modules = list(set(discovered))  # Remove duplicates
        logger.info(f"Discovered {len(self._discovered_modules)} comfy modules")
        return self._discovered_modules
    
    def load_comfy_module(self, module_name: str = "comfy") -> Any:
        """Load a specific comfy module"""
        if module_name in self._loaded_modules:
            return self._loaded_modules[module_name]
        
        self.setup_python_path()
        
        try:
            # Clear any cached failed imports
            if module_name in sys.modules:
                logger.debug(f"Module {module_name} already in sys.modules")
            
            module = importlib.import_module(module_name)
            self._loaded_modules[module_name] = module
            logger.debug(f"Loaded module: {module_name}")
            return module
        except ImportError as e:
            logger.error(f"Failed to load module {module_name}: {e}")
            # Add diagnostic information
            logger.error(f"Python path includes: {sys.path[:3]}...")  # Show first 3 paths
            logger.error(f"ComfyUI path: {self.comfyui_path}")
            raise ImportError(f"Cannot load {module_name}: {e}") from e
    
    def load_modules_by_pattern(self, pattern: str) -> Dict[str, Any]:
        """Load all modules matching a pattern"""
        available_modules = self.discover_comfy_modules()
        matching_modules = {}
        
        for module_name in available_modules:
            if pattern in module_name:
                try:
                    module = self.load_comfy_module(module_name)
                    matching_modules[module_name] = module
                except ImportError as e:
                    logger.warning(f"Could not load matching module {module_name}: {e}")
        
        return matching_modules
    
    def load_all_available_modules(self, max_modules: Optional[int] = None) -> Dict[str, Any]:
        """Load all discovered comfy modules (with optional limit)"""
        available_modules = self.discover_comfy_modules()
        
        if max_modules:
            available_modules = available_modules[:max_modules]
        
        loaded_modules = {}
        for module_name in available_modules:
            try:
                module = self.load_comfy_module(module_name)
                loaded_modules[module_name] = module
            except ImportError as e:
                logger.warning(f"Could not load module {module_name}: {e}")
        
        return loaded_modules
    
    def load_complete_namespace(self) -> Any:
        """Load the complete comfy namespace with all submodules"""
        # Load main comfy module
        comfy = self.load_comfy_module("comfy")
        
        # Get all available modules
        available_modules = self.discover_comfy_modules()
        
        # Load essential core modules first
        core_modules = [
            "comfy.utils",
            "comfy.model_management", 
            "comfy.api",
            "comfy.cli_args_types",
            "comfy.client",
            "comfy.sample",
            "comfy.samplers",
            "comfy.sd",
            "comfy.model_base",
            "comfy.graph",
            "comfy.nodes"
        ]
        
        # Load core modules
        for module_name in core_modules:
            if module_name in available_modules:
                try:
                    self.load_comfy_module(module_name)
                except ImportError as e:
                    logger.warning(f"Could not preload core module {module_name}: {e}")
        
        logger.info(f"Loaded comfy namespace with {len(self._loaded_modules)} modules")
        return comfy
    
    def load_essential_modules_only(self) -> Any:
        """Load only the most essential comfy modules for basic functionality"""
        comfy = self.load_comfy_module("comfy")
        
        essential_modules = [
            "comfy.utils",
            "comfy.model_management",
            "comfy.sample",
            "comfy.samplers"
        ]
        
        for module_name in essential_modules:
            try:
                self.load_comfy_module(module_name)
            except ImportError as e:
                logger.warning(f"Could not load essential module {module_name}: {e}")
        
        return comfy
    
    def get_loaded_modules(self) -> Dict[str, Any]:
        """Get all currently loaded modules"""
        return self._loaded_modules.copy()
    
    def get_available_modules(self) -> List[str]:
        """Get list of all available modules"""
        return self.discover_comfy_modules()
    
# Create a default loader instance
default_loader = ComfyModuleLoader()

def get_comfy_namespace() -> Any:
    """Get the complete comfy namespace using the default loader"""
    return default_loader.load_complete_namespace()

def get_essential_comfy() -> Any:
    """Get essential comfy modules only"""
    return default_loader.load_essential_modules_only()

def discover_available_modules() -> List[str]:
    """Discover all available comfy modules"""
    return default_loader.discover_comfy_modules()

def load_specific_module(module_name: str) -> Any:
    """Load a specific comfy module"""
    return default_loader.load_comfy_module(module_name)

def load_modules_matching(pattern: str) -> Dict[str, Any]:
    """Load all modules matching a pattern"""
    return default_loader.load_modules_by_pattern(pattern)