"""
Workflow loading utilities for ComfyStream server.

This module provides functions to load default workflows from JSON files
instead of hardcoded Python constants.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import importlib.resources

logger = logging.getLogger(__name__)

def find_workflow_file(filename: str) -> Optional[Path]:
    """Find a workflow file, trying package data first, then file system."""
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Try package data first (for installed packages)
    try:
        workflow_file = importlib.resources.files('comfystream') / 'workflows' / 'comfystream' / filename
        if workflow_file.is_file():
            return Path(str(workflow_file))
    except (FileNotFoundError, ModuleNotFoundError):
        pass
    
    # Fallback to file system (for development)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent
    
    # Handle different input path formats
    if filename.startswith('workflows/'):
        # If filename already includes workflows/ prefix, use it directly
        workflow_path = project_root / filename
    else:
        # Otherwise, assume it's just the filename and add the default path
        workflow_path = project_root / "workflows" / "comfystream" / filename
    
    return workflow_path if workflow_path.exists() else None

def load_workflow(filename: str) -> Dict[str, Any]:
    """Load a workflow from a JSON file.
    
    Args:
        filename: Name of the workflow file (with or without .json extension)
        
    Returns:
        Dictionary containing the workflow, or None if file not found
    """
    workflow_path = find_workflow_file(filename)
    
    if workflow_path is None:
        logger.warning(f"Workflow file not found: {filename}")
        logger.warning("Using default workflow as fallback")
        return get_default_workflow()
    
    try:
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        logger.info(f"Loaded workflow from: {workflow_path}")
        return workflow
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in workflow file {workflow_path}: {e}")
    except Exception as e:
        logger.error(f"Error loading workflow {workflow_path}: {e}")
    
    logger.warning("Using default workflow as fallback")
    return get_default_workflow()

def get_default_workflow() -> Dict[str, Any]:
    """Get the default workflow for the pipeline.
    
    Uses a simple passthrough workflow: LoadTensor -> SaveTensor
    This is the most basic workflow that should always work.
    """
    workflow = {
        "1": {
            "inputs": {
                "images": ["2", 0]
            },
            "class_type": "SaveTensor",
            "_meta": {
                "title": "SaveTensor"
            }
        },
        "2": {
            "inputs": {},
            "class_type": "LoadTensor",
            "_meta": {
                "title": "LoadTensor"
            }
        }
    }
    return workflow

def get_inverted_workflow() -> Dict[str, Any]:
    """Get the inverted prompt workflow for image inversion."""
    # Try to load from file, fallback to hardcoded
    try:
        return load_workflow("tensor-utils-example-api.json")
    except:
        return {
            "1": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                    "title": "Empty Latent Image"
                }
            },
            "2": {
                "inputs": {
                    "samples": ["1", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "3": {
                "inputs": {
                    "images": ["2", 0]
                },
                "class_type": "SaveTensor",
                "_meta": {
                    "title": "SaveTensor"
                }
            }
        }
    
def get_default_sd_workflow() -> Dict[str, Any]:
    """Get the default Stable Diffusion prompt workflow."""
    return load_workflow("sd15-tensorrt-api.json")
