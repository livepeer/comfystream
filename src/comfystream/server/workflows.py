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

logger = logging.getLogger(__name__)

def get_workflows_dir() -> Path:
    """Get the path to the workflows directory."""
    # Note: Workflow files are installed by pyproject.toml as data for comfystream.server package
    # Get the root directory of the project (where workflows/ is located)
    current_file = Path(__file__)
    # Go up from src/comfystream/server/workflows.py to the project root
    project_root = current_file.parent.parent.parent.parent
    return project_root / "workflows" / "comfystream"

def load_workflow(filename: str) -> Dict[str, Any]:
    """Load a workflow from a JSON file.
    
    Args:
        filename: Name of the workflow file (with or without .json extension)
        
    Returns:
        Dictionary containing the workflow, or None if file not found
    """
    if not filename.endswith('.json'):
        filename += '.json'
    
    workflows_dir = get_workflows_dir()
    workflow_path = workflows_dir / filename
    workflow = None
    try:
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        logger.info(f"Loaded workflow from {workflow_path}")
    except FileNotFoundError:
        logger.warning(f"Workflow file not found: {workflow_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in workflow file {workflow_path}: {e}")
    except Exception as e:
        logger.error(f"Error loading workflow {workflow_path}: {e}")
    finally:
        if workflow:
            return workflow
        else:
            logger.warning("Using default workflow as fallback")
            return get_default_workflow()

def get_default_workflow() -> Dict[str, Any]:
    """Get the default workflow for the pipeline."""
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
    return load_workflow("inverted-color-api.json")
    
def get_default_sd_workflow() -> Dict[str, Any]:
    """Get the default Stable Diffusion prompt workflow."""
    return load_workflow("sd15-tensorrt-api.json") 