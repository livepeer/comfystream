import copy

from typing import Dict, Any
from comfy.api.components.schema.prompt import Prompt, PromptDictInput


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
        elif class_type in ["PreviewImage", "SaveImage", "SaveTensor", "SaveAudioTensor", "SaveTextTensor"]:
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
    Detect if a workflow is audio-focused by checking for audio processing nodes.
    
    A workflow is considered audio-focused if:
    - It contains LoadAudioTensor or SaveAudioTensor nodes
    - It doesn't contain video processing nodes (LoadTensor, SaveTensor)
    
    Args:
        prompt: The workflow prompt dictionary
        
    Returns:
        True if the workflow is audio-focused, False otherwise
    """
    has_audio_nodes = False
    has_video_nodes = False
    
    for node in prompt.values():
        class_type = node.get("class_type", "")
        
        # Check for audio processing nodes
        if class_type in ["LoadAudioTensor", "SaveAudioTensor"]:
            has_audio_nodes = True
            
        # Check for video processing nodes
        elif class_type in ["LoadTensor", "SaveTensor", "LoadImage", "SaveImage", "PreviewImage", "PrimaryInputLoadImage"]:
            has_video_nodes = True
    
    # Audio-focused if it has audio nodes and no video nodes
    return has_audio_nodes and not has_video_nodes
