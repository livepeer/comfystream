"""ComfyStream nodes package"""

# With the new ComfyUI entry point system, NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
# are automatically discovered from individual .py files, so we don't need to aggregate them here.

# The old import pattern is kept for compatibility with any remaining dependencies
from .audio_utils import *
from .tensor_utils import *
from .video_stream_utils import *
# from .api import *  # Temporarily disabled due to ComfyUI dependency issues
from .web import *

# Web directory for UI components
import os
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")
