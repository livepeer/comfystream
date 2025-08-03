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


# Node type to frame type mapping - extensible and centralized
NODE_TO_FRAME_TYPE_MAPPING = {
    # Audio input/output nodes require AudioFrame
    "LoadAudioTensor": "AudioFrame",
    "SaveAudioTensor": "AudioFrame",
    
    # Video/Image input/output nodes require VideoFrame  
    "LoadTensor": "VideoFrame",
    "SaveTensor": "VideoFrame", 
    "LoadImage": "VideoFrame",
    "SaveImage": "VideoFrame",
    "PreviewImage": "VideoFrame",
    "PrimaryInputLoadImage": "VideoFrame",
    
    # Text nodes could work with any frame type but don't require specific input frames
    "SaveTextTensor": "TextFrame",
    
    # Processing nodes that indicate workflow type
    "PitchShifter": "AudioFrame",  # Audio processing indicates audio workflow
}

# Nodes that modify audio output (vs just analyzing audio)
AUDIO_MODIFICATION_NODES = {
    "SaveAudioTensor",  # Outputs modified audio
    "PitchShifter",     # Modifies audio pitch
    # Add other audio modification nodes here
}

# Nodes that only analyze audio for other outputs (text, etc.)
AUDIO_ANALYSIS_NODES = {
    "LoadAudioTensor",  # Reads audio for analysis
    # Add other audio analysis nodes here
}

def analyze_workflow_frame_requirements(prompt: Dict[Any, Any]) -> Dict[str, bool]:
    """
    Analyze a workflow to determine what frame types are required.
    
    Args:
        prompt: The workflow prompt dictionary
        
    Returns:
        Dictionary with frame types as keys and boolean values indicating if required
        e.g., {"AudioFrame": True, "VideoFrame": False, "TextFrame": False}
    """
    frame_requirements = {
        "AudioFrame": False,
        "VideoFrame": False, 
        "TextFrame": False
    }
    
    for node in prompt.values():
        class_type = node.get("class_type", "")
        required_frame_type = NODE_TO_FRAME_TYPE_MAPPING.get(class_type)
        
        if required_frame_type and required_frame_type in frame_requirements:
            frame_requirements[required_frame_type] = True
    
    return frame_requirements

def has_audio_modification_nodes(prompt: Dict[Any, Any]) -> bool:
    """
    Check if a workflow has nodes that modify audio output.
    
    Args:
        prompt: The workflow prompt dictionary
        
    Returns:
        True if the workflow has audio modification nodes, False otherwise
    """
    for node in prompt.values():
        class_type = node.get("class_type", "")
        if class_type in AUDIO_MODIFICATION_NODES:
            return True
    return False

def has_audio_analysis_nodes(prompt: Dict[Any, Any]) -> bool:
    """
    Check if a workflow has nodes that analyze audio for other outputs.
    
    Args:
        prompt: The workflow prompt dictionary
        
    Returns:
        True if the workflow has audio analysis nodes, False otherwise
    """
    for node in prompt.values():
        class_type = node.get("class_type", "")
        if class_type in AUDIO_ANALYSIS_NODES:
            return True
    return False

def is_audio_modification_workflow(prompt: Dict[Any, Any]) -> bool:
    """
    Detect if a workflow modifies audio (vs just analyzing it).
    
    A workflow is considered audio-modification if:
    - It has audio modification nodes (SaveAudioTensor, PitchShifter, etc.)
    
    Args:
        prompt: The workflow prompt dictionary
        
    Returns:
        True if the workflow modifies audio, False otherwise
    """
    return has_audio_modification_nodes(prompt)

def is_audio_analysis_workflow(prompt: Dict[Any, Any]) -> bool:
    """
    Detect if a workflow only analyzes audio without modifying it.
    
    A workflow is considered audio-analysis if:
    - It has audio analysis nodes (LoadAudioTensor)
    - It doesn't have audio modification nodes (SaveAudioTensor, PitchShifter, etc.)
    
    Args:
        prompt: The workflow prompt dictionary
        
    Returns:
        True if the workflow only analyzes audio, False otherwise
    """
    return has_audio_analysis_nodes(prompt) and not has_audio_modification_nodes(prompt)

def is_audio_focused_workflow(prompt: Dict[Any, Any]) -> bool:
    """
    Detect if a workflow is audio-focused by checking for audio processing nodes.
    
    A workflow is considered audio-focused if:
    - It contains audio processing nodes (requiring AudioFrame)
    - It doesn't contain video processing nodes (requiring VideoFrame)
    
    Args:
        prompt: The workflow prompt dictionary
        
    Returns:
        True if the workflow is audio-focused, False otherwise
    """
    frame_requirements = analyze_workflow_frame_requirements(prompt)
    
    # Audio-focused if it requires audio frames and doesn't require video frames
    return frame_requirements["AudioFrame"] and not frame_requirements["VideoFrame"]
