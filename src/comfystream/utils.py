import copy
import json
import os
import logging
import importlib
from typing import Dict, Any, List, Tuple, Optional, Union
from pytrickle.api import StreamParamsUpdateRequest
from comfy.api.components.schema.prompt import Prompt, PromptDictInput
from .modalities import (
    get_node_counts_by_type,
    get_convertible_node_keys,
)

def create_load_tensor_node():
    return {
        "inputs": {},
        "class_type": "LoadTensor",
        "_meta": {"title": "LoadTensor"},
    }

def create_save_tensor_node(inputs: Dict[Any, Any]):
    return {
        "inputs": inputs,
        "class_type": "SaveTensor",
        "_meta": {"title": "SaveTensor"},
    }

def _validate_prompt_constraints(counts: Dict[str, int]) -> None:
    """Validate that the prompt meets the required constraints."""
    if counts["primary_inputs"] > 1:
        raise Exception("too many primary inputs in prompt")

    if counts["primary_inputs"] == 0 and counts["inputs"] > 2:
        raise Exception("too many inputs in prompt")

    if counts["outputs"] > 3:
        raise Exception("too many outputs in prompt")

    if counts["primary_inputs"] + counts["inputs"] == 0:
        raise Exception("missing input")

    if counts["outputs"] == 0:
        raise Exception("missing output")

def convert_prompt(prompt: PromptDictInput, return_dict: bool = False) -> Prompt:
    """Convert a prompt by replacing specific node types with tensor equivalents."""
    try:
        # Note: lazy import is necessary to prevent KeyError during validation
        importlib.import_module("comfy.api.components.schema.prompt_node")
    except Exception:
        pass
    
    """Convert and validate a ComfyUI workflow prompt."""
    Prompt.validate(prompt)
    prompt = copy.deepcopy(prompt)

    # Count nodes and validate constraints
    counts = get_node_counts_by_type(prompt)
    _validate_prompt_constraints(counts)
    
    # Collect nodes that need conversion
    convertible_keys = get_convertible_node_keys(prompt)

    # Replace nodes based on their conversion type
    for key in convertible_keys["PrimaryInputLoadImage"]:
        prompt[key] = create_load_tensor_node()

    # Conditional replacement: only if no primary input and exactly one LoadImage
    if counts["primary_inputs"] == 0 and len(convertible_keys["LoadImage"]) == 1:
        prompt[convertible_keys["LoadImage"][0]] = create_load_tensor_node()

    # Replace output nodes
    for key in convertible_keys["PreviewImage"] + convertible_keys["SaveImage"]:
        node = prompt[key]
        prompt[key] = create_save_tensor_node(node["inputs"])

    # Return dict if requested (for downstream components that expect plain dicts)
    if return_dict:
        return prompt  # Already a plain dict at this point
    
    # Validate the processed prompt and return Pydantic object
    return Prompt.validate(prompt)


def load_prompt_from_file(path: str) -> PromptDictInput:
    """Load a prompt from a JSON file."""
    provided = path.lstrip("./")
    
    if ".." in provided or os.path.isabs(provided):
        raise ValueError("Invalid path format")
    
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    if provided.startswith("workflows/comfystream/"):
        resolved = os.path.normpath(os.path.join(repo_root, provided))
    else:
        resolved = os.path.normpath(os.path.join(repo_root, "workflows", "comfystream", provided))
    
    workflows_dir = os.path.normpath(os.path.join(repo_root, "workflows", "comfystream"))
    if not resolved.startswith(workflows_dir):
        raise ValueError("Path is outside workflows directory")

    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)

    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected JSON file to contain a dictionary")
    
    return convert_prompt(data, return_dict=True)



class ComfyStreamParamsUpdateRequest(StreamParamsUpdateRequest if StreamParamsUpdateRequest else object):
    """ComfyStream parameter validation."""
    
    def __init__(self, **data):
        # Handle prompts parameter
        if "prompts" in data:
            prompts = data["prompts"]
            
            # Parse JSON string if needed
            if isinstance(prompts, str) and prompts.strip():
                try:
                    prompts = json.loads(prompts)
                except json.JSONDecodeError:
                    data.pop("prompts")
            
            # Handle list - use first valid dict
            elif isinstance(prompts, list):
                prompts = next((p for p in prompts if isinstance(p, dict)), None)
                if not prompts:
                    data.pop("prompts")
            
            # Validate prompts
            if "prompts" in data and isinstance(prompts, dict):
                try:
                    data["prompts"] = convert_prompt(prompts, return_dict=True)
                except Exception:
                    data.pop("prompts")
        
        # Call parent constructor
        if StreamParamsUpdateRequest:
            super().__init__(**data)
        else:
            for key, value in data.items():
                setattr(self, key, value)
    
    @classmethod
    def model_validate(cls, obj):
        return cls(**obj)
    
    def model_dump(self):
        if StreamParamsUpdateRequest:
            return super().model_dump()
        else:
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def get_default_workflow() -> dict:
    """Return the default workflow as a dictionary for warmup.
    
    Returns:
        dict: Default workflow dictionary
    """
    return {
        "1": {
            "inputs": {
                "images": [
                    "2",
                    0
                ]
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

