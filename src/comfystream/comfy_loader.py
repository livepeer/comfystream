"""ComfyUI namespace loader for hiddenswitch branch v0.3.40

This module provides importlib-based namespace override for ComfyUI hiddenswitch branch,
ensuring proper isolation between vanilla custom nodes and comfystream server components.
"""

import os
import sys
import importlib
import importlib.util
import tempfile
import tarfile
import shutil
from pathlib import Path
from typing import Optional

# Package data path for ComfyUI hiddenswitch
COMFYUI_PACKAGE_DATA = "comfyui_hiddenswitch_v0.3.40"

def extract_comfyui_package_data():
    """Extract ComfyUI hiddenswitch v0.3.40 from package data"""
    try:
        import importlib.resources as resources
        package_data_path = resources.files('comfystream').joinpath(COMFYUI_PACKAGE_DATA)
        
        if package_data_path.exists():
            return str(package_data_path)
    except ImportError:
        pass
    
    # Fallback: check if package data exists in development environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dev_package_data = os.path.join(current_dir, COMFYUI_PACKAGE_DATA)
    
    if os.path.exists(dev_package_data):
        return dev_package_data
    
    # If package data doesn't exist, download and extract
    return download_and_extract_comfyui()

def download_and_extract_comfyui():
    """Download and extract ComfyUI hiddenswitch v0.3.40"""
    import urllib.request
    
    url = "https://github.com/hiddenswitch/ComfyUI/archive/refs/tags/v0.3.40.tar.gz"
    temp_dir = tempfile.mkdtemp(prefix="comfyui_hiddenswitch_")
    
    try:
        # Download the tar.gz
        tar_path = os.path.join(temp_dir, "comfyui_hiddenswitch_v0.3.40.tar.gz")
        print(f"Downloading ComfyUI hiddenswitch v0.3.40 from {url}")
        urllib.request.urlretrieve(url, tar_path)
        
        # Extract the tar.gz
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(temp_dir)
        
        # Find the extracted directory
        extracted_dir = None
        for item in os.listdir(temp_dir):
            if item.startswith("ComfyUI-"):
                extracted_dir = os.path.join(temp_dir, item)
                break
        
        if not extracted_dir:
            raise RuntimeError("Failed to find extracted ComfyUI directory")
        
        return extracted_dir
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to download/extract ComfyUI: {e}")

def setup_comfy_namespace():
    """Setup the comfy namespace using importlib override"""
    comfyui_path = extract_comfyui_package_data()
    
    # Add ComfyUI path to sys.path
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)
    
    # Ensure comfy and comfy_extras directories have __init__.py files
    for subdir in ['comfy', 'comfy_extras']:
        init_path = os.path.join(comfyui_path, subdir, '__init__.py')
        if not os.path.exists(init_path):
            os.makedirs(os.path.dirname(init_path), exist_ok=True)
            with open(init_path, 'w') as f:
                f.write("# Auto-generated __init__.py for comfy namespace\n")
    
    # Override the comfy module using importlib
    try:
        # Import comfy modules
        import comfy
        import comfy_extras
        
        # Store original modules for potential restoration
        if 'comfy' in sys.modules:
            sys.modules['_original_comfy'] = sys.modules['comfy']
        if 'comfy_extras' in sys.modules:
            sys.modules['_original_comfy_extras'] = sys.modules['comfy_extras']
        
        return True
    except ImportError as e:
        print(f"Warning: Failed to import comfy modules: {e}")
        return False

def is_vanilla_custom_node_context():
    """Check if we're running in a vanilla custom node context"""
    # Check if we're being imported by ComfyUI's custom node loader
    import inspect
    frame = inspect.currentframe()
    
    while frame:
        if frame.f_code.co_name == 'load_custom_node':
            return True
        frame = frame.f_back
    
    # Also check if we're in a custom_nodes directory
    import os
    current_path = os.path.abspath(__file__)
    if 'custom_nodes' in current_path:
        return True
    
    # Check if we're being imported from a custom_nodes directory
    try:
        import inspect
        frame = inspect.currentframe()
        while frame:
            if hasattr(frame, 'f_globals'):
                module_name = frame.f_globals.get('__name__', '')
                if 'custom_nodes' in module_name:
                    return True
            frame = frame.f_back
    except:
        pass
    
    return False

def should_load_comfystream_components():
    """Determine if comfystream server components should be loaded"""
    # Don't load server components in vanilla custom node context
    if is_vanilla_custom_node_context():
        return False
    
    # Check environment variables
    if os.environ.get('COMFYSTREAM_DISABLE_SERVER', '0') == '1':
        return False
    
    return True

# Initialize comfy namespace when module is imported
# Always setup namespace, but control component loading separately
setup_comfy_namespace() 