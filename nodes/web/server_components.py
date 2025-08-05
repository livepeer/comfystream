"""Server components for ComfyStream that can be imported when needed"""

import os
import sys
import asyncio
import logging
from typing import Optional

# Add src to path for importing comfystream components
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import comfy_loader for context detection
from comfystream import comfy_loader

def get_server_components():
    """Get server components if available"""
    # Only try to import server components if not in vanilla custom node context
    if comfy_loader.is_vanilla_custom_node_context():
        logging.info("Skipping server component import in vanilla custom node context")
        return {}
    
    try:
        from comfystream import ComfyStreamClient, Pipeline
        return {
            'ComfyStreamClient': ComfyStreamClient,
            'Pipeline': Pipeline
        }
    except ImportError as e:
        logging.warning(f"Server components not available: {e}")
        return {}

def get_pipeline_class():
    """Get the Pipeline class if available"""
    components = get_server_components()
    return components.get('Pipeline')

def get_client_class():
    """Get the ComfyStreamClient class if available"""
    components = get_server_components()
    return components.get('ComfyStreamClient') 