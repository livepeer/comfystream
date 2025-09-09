import copy

from typing import Dict, Any, Set, Union, List
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

    # # Only handle single output for now
    # if num_outputs > 1:
    #     raise Exception("too many outputs in prompt")
    
    if num_outputs == 0:
        raise Exception("missing output")

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


def detect_prompt_modalities(prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]) -> Set[str]:
    """Detect modalities (video, audio, text) required by a workflow.
    
    The modality detection follows passthrough logic:
    - Video: Only processes if SaveTensor/SaveImage/PreviewImage nodes are present, otherwise passthrough
    - Audio: Only processes if SaveAudioTensor nodes are present, otherwise passthrough  
    - Text: Processes if SaveTextTensor nodes are present
    
    Args:
        prompts: Single prompt dict or list of prompt dicts
        
    Returns:
        Set of modality strings: {'video', 'audio', 'text'}
    """
    if isinstance(prompts, list):
        # For multiple prompts, detect modalities across all
        all_modalities = set()
        for prompt in prompts:
            all_modalities.update(detect_prompt_modalities(prompt))
        return all_modalities
    
    modalities = set()
    prompt = prompts
    
    # Video modality detection - only if there are output nodes that require processing
    video_output_nodes = {
        "SaveTensor", "PreviewImage", "SaveImage"
    }
    
    # Audio modality detection - only if there are output nodes that require processing
    audio_output_nodes = {
        "SaveAudioTensor"
    }
    
    # Text modality detection
    text_output_nodes = {
        "SaveTextTensor"
    }
    
    # Track what we find
    has_video_output = False
    has_audio_output = False
    has_text_output = False
    has_video_input = False
    has_audio_input = False
    
    video_input_nodes = {
        "LoadTensor", "PrimaryInputLoadImage", "LoadImage"
    }
    audio_input_nodes = {
        "LoadAudioTensor", "LoadAudioTensorStream"
    }
    
    for node_id, node_data in prompt.items():
        class_type = node_data.get("class_type", "")
        
        # Check for input nodes
        if class_type in video_input_nodes:
            has_video_input = True
        if class_type in audio_input_nodes:
            has_audio_input = True
            
        # Check for output nodes that require processing
        if class_type in video_output_nodes:
            has_video_output = True
        if class_type in audio_output_nodes:
            has_audio_output = True
        if class_type in text_output_nodes:
            has_text_output = True
    
    # Only add modalities if there are both inputs and outputs that require processing
    # This implements the passthrough logic - if there's no SaveTensor node, video passes through
    if has_video_input and has_video_output:
        modalities.add("video")
    if has_audio_input and has_audio_output:
        modalities.add("audio")  
    if has_text_output:  # Text output doesn't require input
        modalities.add("text")
    
    return modalities
