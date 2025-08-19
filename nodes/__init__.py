"""ComfyStream nodes package"""

from .audio_utils import *
from .tensor_utils import *
from .video_stream_utils import *
from .api import *
from .web import *

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

# Validate node classes to prevent JSON serialization issues
try:
    import sys
    if 'comfystream.node_validation' not in sys.modules:
        from comfystream.node_validation import validate_all_node_classes
        validate_all_node_classes(NODE_CLASS_MAPPINGS)
except ImportError:
    # node_validation module not available, skip validation
    pass
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Node validation failed: {e}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
