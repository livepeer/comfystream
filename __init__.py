import os
import pathlib
success = True

import sys
from pathlib import Path
root = Path(__file__).resolve().parent

sys.path.insert(0, str(root / 'src'))
import nodes
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
# Point to the directory containing our web files
WEB_DIRECTORY = "./nodes/web/js"

# Import and expose node classes
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

from comfystream.tensor_cache import (
    image_inputs,
    image_outputs, 
    audio_inputs,
    audio_outputs
)

# Or import the entire module
from . import tensor_cache

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 
           'image_inputs', 'image_outputs', 
           'audio_inputs', 'audio_outputs',
           'tensor_cache']