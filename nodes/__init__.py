"""ComfyStream nodes package"""

# Import comfy_loader first to ensure proper namespace setup
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from comfystream import comfy_loader

# Import node modules (comfy namespace should already be set up)
try:
    from .audio_utils import *
    from .tensor_utils import *
    from .video_stream_utils import *
    from .api import *
    from .web import *
except ImportError as e:
    # If imports fail, provide empty mappings
    print(f"Warning: Failed to import comfystream node modules: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Collect all NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS from submodules
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import and update mappings from submodules
for module in [audio_utils, tensor_utils, video_stream_utils, api, web]:
    if hasattr(module, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

# Web directory for UI components
import os
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

NODE_DISPLAY_NAME_MAPPINGS["ComfyStreamLauncher"] = "Launch ComfyStream 🚀"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
