import copy
import json
import logging

from typing import Dict, Any, List, Union
from comfy.api.components.schema.prompt import Prompt, PromptDictInput

logger = logging.getLogger(__name__)


# Input node types
INPUT_NODES_PRIMARY = ["PrimaryInputLoadImage"]
INPUT_NODES_GENERAL = ["LoadImage", "LoadTensor", "LoadAudioTensor"]
INPUT_NODES_GENERATION = ["EmptyLatentImage"]

# Output node types
OUTPUT_NODES_PREVIEW = ["PreviewImage"]
OUTPUT_NODES_SAVE = ["SaveImage", "SaveTensor", "SaveAudioTensor"]

# Audio-focused node types
AUDIO_INPUT_NODES = ["LoadAudioTensor"]
AUDIO_OUTPUT_NODES = ["SaveAudioTensor"]

# Video/Image-focused node types
VIDEO_INPUT_NODES = ["LoadTensor", "EmptyLatentImage", "LoadImage", "PrimaryInputLoadImage"]
VIDEO_OUTPUT_NODES = ["PreviewImage", "SaveImage", "SaveTensor"]

# Node type collections for workflow processing
REPLACEABLE_NODE_TYPES = INPUT_NODES_PRIMARY + INPUT_NODES_GENERAL + OUTPUT_NODES_PREVIEW + OUTPUT_NODES_SAVE

# All input types (for counting and validation)
ALL_INPUT_NODES = INPUT_NODES_PRIMARY + INPUT_NODES_GENERAL + INPUT_NODES_GENERATION

# All output types (for counting and validation)
ALL_OUTPUT_NODES = OUTPUT_NODES_PREVIEW + OUTPUT_NODES_SAVE

# Utility node types that don't require input/output nodes
UTILITY_NODE_TYPES = [
    "UnloadAllModels", "UnloadModel", "FreeMemory", "ClearLatentCache",
    "ClearCacheNode", "GarbageCollectNode", "SystemInfoNode"
]


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


def convert_prompt(prompt: PromptDictInput) -> Prompt:
    """
    Convert and validate a ComfyUI workflow prompt.
    
    """
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
        if class_type in INPUT_NODES_PRIMARY:
            num_primary_inputs += 1
        elif class_type in INPUT_NODES_GENERAL:
            num_inputs += 1
        elif class_type in ALL_OUTPUT_NODES:
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


def is_audio_focused_workflow(prompt: Dict[Any, Any]) -> bool:
    """
    Detect if a workflow is audio-focused by checking the input node type.
    
    This function should be called AFTER convert_prompt() has been run to ensure
    consistent node types. It checks for:
    - LoadAudioTensor = audio workflow
    - LoadTensor, EmptyLatentImage, LoadImage = video workflow
    
    Args:
        prompt: The workflow prompt dictionary (must be processed by convert_prompt first)
        
    Returns:
        True if the workflow is audio-focused, False otherwise
    """
    for node in prompt.values():
        class_type = node.get("class_type", "")
        
        # Check for audio input node
        if class_type in AUDIO_INPUT_NODES:
            return True
            
        # Check for video/image input nodes
        elif class_type in VIDEO_INPUT_NODES:
            return False
    
    # Default to video workflow if no clear input node found
    return False

def parse_prompt_data(prompt_data: Union[str, Dict, List[Dict]]) -> List[Dict[str, Any]]:
    """Parse prompt data from various formats into a standardized list of dictionaries.
    
    Args:
        prompt_data: Can be:
            - A JSON string containing a dict or list of dicts
            - A single prompt dictionary
            - A list of prompt dictionaries
            
    Returns:
        List of prompt dictionaries
        
    Raises:
        ValueError: If the prompt data format is invalid
        json.JSONDecodeError: If JSON string is malformed
    """
    # Handle JSON string input (common in control messages)
    if isinstance(prompt_data, str):
        try:
            prompt_data = json.loads(prompt_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in prompts: {e}")
            raise ValueError(f"Invalid JSON in prompts: {e}")
    
    # Handle dict or list input
    if isinstance(prompt_data, dict):
        return [prompt_data]
    elif isinstance(prompt_data, list):
        if not all(isinstance(prompt, dict) for prompt in prompt_data):
            raise ValueError("All prompts in list must be dictionaries")
        return prompt_data
    else:
        raise ValueError(f"Prompts must be dict, list, or JSON string, got {type(prompt_data)}")
