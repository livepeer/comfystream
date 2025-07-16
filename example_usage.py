"""
Example usage of the enhanced ComfyUI module loader
"""

# Import the loader functions
from comfystream.comfy_loader import (
    get_comfy_namespace,
    get_essential_comfy,
    discover_available_modules,
    load_specific_module,
    load_modules_matching,
    default_loader,
    ComfyModuleLoader
)

# Example 1: Quick start - Load essential modules only
def quick_start_example():
    """Load just the essential modules for basic functionality"""
    comfy = get_essential_comfy()
    
    # Now you can use comfy.utils, comfy.model_management, etc.
    # These modules are automatically loaded and available
    return comfy

# Example 2: Full namespace - Load everything available
def full_namespace_example():
    """Load the complete comfy namespace with all discovered modules"""
    comfy = get_comfy_namespace()
    
    # This loads the main comfy module plus all core modules
    # Additional modules can be loaded on-demand
    return comfy

# Example 3: Discover what's available
def discovery_example():
    """Discover all available modules before loading"""
    # Get list of all available modules
    available_modules = discover_available_modules()
    print(f"Available modules: {len(available_modules)}")
    
    # Filter for specific types
    api_modules = [m for m in available_modules if 'api' in m]
    model_modules = [m for m in available_modules if 'model' in m]
    
    print(f"API-related modules: {api_modules}")
    print(f"Model-related modules: {model_modules}")
    
    return available_modules

# Example 4: Load specific modules on demand
def on_demand_loading_example():
    """Load specific modules only when needed"""
    # Load just the modules you need
    utils_module = load_specific_module("comfy.utils")
    model_mgmt = load_specific_module("comfy.model_management")
    
    # Use the modules
    # utils_module.some_function()
    # model_mgmt.some_other_function()
    
    return utils_module, model_mgmt

# Example 5: Pattern-based loading
def pattern_loading_example():
    """Load modules matching specific patterns"""
    # Load all API-related modules
    api_modules = load_modules_matching("api")
    
    # Load all sampling-related modules
    sampling_modules = load_modules_matching("sample")
    
    # Load all model-related modules
    model_modules = load_modules_matching("model")
    
    return api_modules, sampling_modules, model_modules

# Example 6: Custom loader with different path
def custom_loader_example():
    """Create a custom loader with different settings"""
    # Create a custom loader for a different ComfyUI installation
    custom_loader = ComfyModuleLoader(
        comfyui_path="/custom/path/to/comfyui"
    )
    
    # Use the custom loader
    modules = custom_loader.discover_comfy_modules()
    comfy = custom_loader.load_complete_namespace()
    
    return custom_loader, comfy

# Example 7: Error handling and debugging
def robust_loading_example():
    """Demonstrate robust loading with error handling"""
    try:
        # Attempt to load complete namespace
        comfy = get_comfy_namespace()
        
        # Check what modules were actually loaded
        loaded_modules = default_loader.get_loaded_modules()
        print(f"Successfully loaded {len(loaded_modules)} modules")
        
        # Try to access specific functionality
        if 'comfy.utils' in loaded_modules:
            utils = loaded_modules['comfy.utils']
            print("Utils module available")
        
        return comfy
        
    except ImportError as e:
        print(f"Import error: {e}")
        # Fallback to essential modules only
        return get_essential_comfy()
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Example 8: Integration in your application
class MyComfyApplication:
    """Example application using the ComfyUI loader"""
    
    def __init__(self):
        self.comfy_loader = default_loader
        self.comfy = None
        self.loaded_modules = {}
    
    def initialize(self):
        """Initialize the ComfyUI modules"""
        try:
            # Discover available modules
            available = self.comfy_loader.discover_comfy_modules()
            print(f"Found {len(available)} available ComfyUI modules")
            
            # Load essential modules
            self.comfy = get_essential_comfy()
            
            # Load additional modules as needed
            self._load_additional_modules()
            
            print("ComfyUI initialization complete")
            return True
            
        except Exception as e:
            print(f"Failed to initialize ComfyUI: {e}")
            return False
    
    def _load_additional_modules(self):
        """Load additional modules based on application needs"""
        additional_modules = [
            "comfy.api",
            "comfy.graph",
            "comfy.nodes"
        ]
        
        for module_name in additional_modules:
            try:
                module = load_specific_module(module_name)
                self.loaded_modules[module_name] = module
                print(f"Loaded additional module: {module_name}")
            except ImportError:
                print(f"Optional module not available: {module_name}")
    
    def get_module(self, module_name: str):
        """Get a specific loaded module"""
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        
        # Try to load it on demand
        try:
            module = load_specific_module(module_name)
            self.loaded_modules[module_name] = module
            return module
        except ImportError:
            return None

if __name__ == "__main__":
    # Run examples
    print("=== ComfyUI Loader Examples ===")
    
    # Example usage
    app = MyComfyApplication()
    if app.initialize():
        print("Application initialized successfully!")
        
        # Use the loaded modules
        utils_module = app.get_module("comfy.utils")
        if utils_module:
            print("Utils module is available for use")
