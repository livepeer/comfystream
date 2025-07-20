import os
import sys
import importlib.abc
import importlib.machinery
import importlib.util

# Point to the directory containing our web files
WEB_DIRECTORY = "./nodes/web/js"

# Set up comfy_execution redirection immediately
def setup_comfy_execution_redirection():
    """Set up import redirection for comfy_execution modules to comfy modules"""
    
    # Check if we're in vanilla custom node mode
    # Look for multiple indicators of vanilla loading
    current_file = os.path.abspath(__file__)
    is_vanilla_mode = (
        'custom_nodes' in current_file or 
        'custom_nodes' in os.getcwd() or
        os.environ.get('COMFY_UI_WORKSPACE') is not None
    )
    
    # Always set up redirection for safety, but log the mode
    if is_vanilla_mode:
        print("[ComfyStream] Setting up comfy_execution redirection for vanilla custom node mode")
    else:
        print("[ComfyStream] Setting up comfy_execution redirection for package mode")
    
    class ComfyExecutionRedirectFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            # Redirection mapping for comfy_execution.* to comfy.*
            # Only include modules that have actually been moved to the comfy package
            redirection_map = {
                'comfy_execution.validation': 'comfy.validation',
                # Add other modules here only if they have been moved to comfy package
                # Most comfy_execution modules should remain in comfy_execution
            }
            
            if fullname in redirection_map:
                target_module = redirection_map[fullname]
                try:
                    # Import the target module first
                    target_module_obj = importlib.import_module(target_module)
                    print(f"[ComfyStream] Redirecting {fullname} to {target_module}")
                    
                    # Create a simple loader that returns the target module
                    class RedirectLoader(importlib.abc.Loader):
                        def exec_module(self, module):
                            # Copy all attributes from the target module
                            for attr_name in dir(target_module_obj):
                                if not attr_name.startswith('_'):
                                    setattr(module, attr_name, getattr(target_module_obj, attr_name))
                            # Set the module's __name__ to the expected name
                            module.__name__ = fullname
                            module.__file__ = target_module_obj.__file__
                            module.__package__ = fullname.rsplit('.', 1)[0]
                    
                    return importlib.machinery.ModuleSpec(
                        fullname, 
                        RedirectLoader(),
                        origin=target_module_obj.__file__
                    )
                except Exception as e:
                    print(f"[ComfyStream] Error redirecting {fullname}: {e}")
                    pass
            
            return None
    
    # Install the finder if not already installed
    finder = ComfyExecutionRedirectFinder()
    if finder not in sys.meta_path:
        sys.meta_path.insert(0, finder)

# Set up the redirection immediately
setup_comfy_execution_redirection()

# Import node mappings from all node modules
from .nodes.tensor_utils import NODE_CLASS_MAPPINGS as tensor_utils_mappings
from .nodes.audio_utils import NODE_CLASS_MAPPINGS as audio_utils_mappings
from .nodes.video_stream_utils import NODE_CLASS_MAPPINGS as video_stream_utils_mappings
from .nodes.web import NODE_CLASS_MAPPINGS as web_mappings

# Import API module (this sets up the web routes)
from .nodes import api

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(tensor_utils_mappings)
NODE_CLASS_MAPPINGS.update(audio_utils_mappings)
NODE_CLASS_MAPPINGS.update(video_stream_utils_mappings)
NODE_CLASS_MAPPINGS.update(web_mappings)

# Import display name mappings
from .nodes.tensor_utils import NODE_DISPLAY_NAME_MAPPINGS as tensor_utils_display_mappings
from .nodes.audio_utils import NODE_DISPLAY_NAME_MAPPINGS as audio_utils_display_mappings
from .nodes.video_stream_utils import NODE_DISPLAY_NAME_MAPPINGS as video_stream_utils_display_mappings
from .nodes.web import NODE_DISPLAY_NAME_MAPPINGS as web_display_mappings

# Combine all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(tensor_utils_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(audio_utils_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(video_stream_utils_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(web_display_mappings)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']