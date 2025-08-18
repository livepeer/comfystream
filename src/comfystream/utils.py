import copy
import json
import logging

from typing import Dict, Any, List, Union
from comfy.api.components.schema.prompt import Prompt, PromptDictInput

logger = logging.getLogger(__name__)


# Input node types
INPUT_NODES_PRIMARY = ["PrimaryInputLoadImage"]
INPUT_NODES_GENERAL = ["LoadImage", "LoadTensor", "LoadAudioTensor", "LoadAudioTensorStream"]
INPUT_NODES_GENERATION = ["EmptyLatentImage"]

# Output node types
OUTPUT_NODES_PREVIEW = ["PreviewImage"]
OUTPUT_NODES_SAVE = ["SaveImage", "SaveTensor", "SaveAudioTensor", "SaveTextTensor"]

# Audio-focused node types
AUDIO_INPUT_NODES = ["LoadAudioTensor", "LoadAudioTensorStream"]
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
    try:
        prompt = Prompt.validate(prompt)
    except Exception as e:
        logger.warning(f"Prompt validation failed: {e}, returning unvalidated prompt")
        # Return the prompt as-is if validation fails
        pass

    return prompt


# Node type to frame type mapping - extensible and centralized
NODE_TO_FRAME_TYPE_MAPPING = {
    # Audio input/output nodes require AudioFrame
    "LoadAudioTensor": "AudioFrame",
    "SaveAudioTensor": "AudioFrame",
    "LoadAudioTensorStream": "AudioFrame",
    "AudioTranscriptionNode": "AudioFrame",
    "SRTGeneratorNode": "AudioFrame",
    
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
    "LoadAudioTensorStream",  # Reads audio for analysis
    "AudioTranscriptionNode",  # Transcribes audio to text
    "SRTGeneratorNode",  # Generates SRT from transcription
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

def analyze_workflow_output_types(prompt: Dict[Any, Any]) -> Dict[str, bool]:
    """
    Analyze a workflow to determine what output types it produces.
    
    Args:
        prompt: The workflow prompt dictionary
        
    Returns:
        Dictionary with output types as keys and boolean values indicating if produced
        e.g., {"audio_output": True, "video_output": False, "text_output": True}
    """
    output_types = {
        "audio_output": False,
        "video_output": False,
        "text_output": False
    }
    
    # Define nodes that produce specific output types
    audio_output_nodes = {"SaveAudioTensor"}
    video_output_nodes = {"SaveImageTensor", "SaveTensor"}  
    text_output_nodes = {"SaveTextTensor", "SRTGeneratorNode", "AudioTranscriptionNode"}
    
    for node in prompt.values():
        class_type = node.get("class_type", "")
        
        if class_type in audio_output_nodes:
            output_types["audio_output"] = True
        elif class_type in video_output_nodes:
            output_types["video_output"] = True
        elif class_type in text_output_nodes:
            output_types["text_output"] = True
    
    return output_types

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

def enable_warmup_mode(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enable warmup mode by setting debug_info=True on SaveTextTensor nodes.
    AudioTranscriptionNode automatically handles warmup sentinels based on model loading state.
    
    Args:
        workflow: The workflow dictionary to modify
        
    Returns:
        Modified workflow with warmup mode enabled for SaveTextTensor nodes
    """
    warmup_workflow = copy.deepcopy(workflow)
    
    modified_nodes = 0
    for node_id, node_data in warmup_workflow.items():
        if isinstance(node_data, dict):
            class_type = node_data.get("class_type")
            if class_type == "SaveTextTensor":
                # Enable debug_info for warmup sentinels (AudioTranscriptionNode now handles this automatically)
                if "inputs" not in node_data:
                    node_data["inputs"] = {}
                node_data["inputs"]["debug_info"] = True
                modified_nodes += 1
                logger.info(f"Enabled warmup mode for {class_type} node {node_id}")
    
    if modified_nodes > 0:
        logger.info(f"Enabled warmup mode for {modified_nodes} SaveTextTensor node(s)")
    else:
        logger.debug("No SaveTextTensor nodes found to enable warmup mode")
        
    return warmup_workflow

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
