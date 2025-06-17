"""
ComfyStream utilities for prompt processing and ComfyUI integration
"""
import sys
import copy
import asyncio
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Import ComfyStream modules
from comfystream import tensor_cache
from comfystream.comfy_loader import get_comfy_namespace, load_specific_module

# Setup comfy modules with complete namespace
logger.info("Loading ComfyUI namespace...")
try:
    comfy = get_comfy_namespace()
    logger.info("ComfyUI namespace loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ComfyUI namespace: {e}")
    raise

# Import specific ComfyUI components
try:
    # Load specific modules we need
    schema_module = load_specific_module("comfy.api.components.schema.prompt")
    cli_args_module = load_specific_module("comfy.cli_args_types") 
    client_module = load_specific_module("comfy.client.embedded_comfy_client")
    
    # Import the classes we need
    PromptDictInput = schema_module.PromptDictInput
    Configuration = cli_args_module.Configuration
    Comfy = client_module.Comfy
    
    logger.info("ComfyUI components imported successfully")
except Exception as e:
    logger.error(f"Failed to import ComfyUI components: {e}")
    raise

# Define Prompt class for validation (if not available from ComfyUI)
class Prompt(dict):
    """Prompt wrapper with validation"""
    
    @classmethod
    def validate(cls, prompt_data):
        """Validate prompt data structure"""
        if not isinstance(prompt_data, dict):
            raise ValueError("Prompt must be a dictionary")
        
        for node_id, node_data in prompt_data.items():
            if not isinstance(node_data, dict):
                raise ValueError(f"Node {node_id} must be a dictionary")
            
            if "class_type" not in node_data:
                raise ValueError(f"Node {node_id} missing class_type")
        
        return cls(prompt_data)

def convert_prompt(prompt: PromptDictInput) -> Prompt:
    # Validate the schema
    Prompt.validate(prompt)

    prompt = copy.deepcopy(prompt)

    num_primary_inputs = 0
    num_inputs = 0
    num_outputs = 0

    keys = {
        "PrimaryInputLoadImage": [],
        "LoadImage": [],
        "PreviewImage": [],
        "SaveImage": [],
    }
    
    for key, node in prompt.items():
        class_type = node.get("class_type")

        # Collect keys for nodes that might need to be replaced
        if class_type in keys:
            keys[class_type].append(key)

        # Count inputs and outputs
        if class_type == "PrimaryInputLoadImage":
            num_primary_inputs += 1
        elif class_type in ["LoadImage", "LoadTensor", "LoadAudioTensor"]:
            num_inputs += 1
        elif class_type in ["PreviewImage", "SaveImage", "SaveTensor", "SaveAudioTensor"]:
            num_outputs += 1

    # Only handle single primary input
    if num_primary_inputs > 1:
        raise Exception("too many primary inputs in prompt")

    # If there are no primary inputs, only handle single input
    if num_primary_inputs == 0 and num_inputs > 1:
        raise Exception("too many inputs in prompt")

    # Only handle single output for now
    if num_outputs > 1:
        raise Exception("too many outputs in prompt")

    if num_primary_inputs + num_inputs == 0:
        raise Exception("missing input")

    if num_outputs == 0:
        raise Exception("missing output")

    # Replace nodes
    for key in keys["PrimaryInputLoadImage"]:
        prompt[key] = create_load_tensor_node()

    if num_primary_inputs == 0 and len(keys["LoadImage"]) == 1:
        prompt[keys["LoadImage"][0]] = create_load_tensor_node()

    for key in keys["PreviewImage"] + keys["SaveImage"]:
        node = prompt[key]
        prompt[key] = create_save_tensor_node(node["inputs"])

    # Validate the processed prompt input
    prompt = Prompt.validate(prompt)

    return prompt

# Helper functions for prompt processing
def create_load_tensor_node():
    """Create a LoadTensor node for input processing"""
    return {
        "class_type": "LoadTensor",
        "inputs": {}
    }

def create_save_tensor_node(original_inputs: Dict[str, Any]):
    """Create a SaveTensor node for output processing"""
    return {
        "class_type": "SaveTensor", 
        "inputs": original_inputs.copy() if original_inputs else {}
    }

def create_load_audio_tensor_node():
    """Create a LoadAudioTensor node for audio input processing"""
    return {
        "class_type": "LoadAudioTensor",
        "inputs": {}
    }

def create_save_audio_tensor_node(original_inputs: Dict[str, Any]):
    """Create a SaveAudioTensor node for audio output processing"""
    return {
        "class_type": "SaveAudioTensor",
        "inputs": original_inputs.copy() if original_inputs else {}
    }
