import copy
import json
import os
import logging

from typing import Dict, Any, List, Tuple, Optional, Union
from comfy.api.components.schema.prompt import Prompt, PromptDictInput
from pytrickle.api import StreamParamsUpdateRequest


logger = logging.getLogger(__name__)


def analyze_prompt_io(prompt: Union[PromptDictInput, Prompt]) -> Dict[str, List[Tuple[str, str]]]:
    """Analyze a prompt and determine input/output nodes by domain."""
    result = {
        "audio_inputs": [], "audio_outputs": [],
        "video_inputs": [], "video_outputs": [],
        "text_inputs": [], "text_outputs": []
    }

    items_iter = prompt.items() if hasattr(prompt, 'items') else dict(prompt).items()

    for node_id, node in items_iter:
        class_type = node.get("class_type") if isinstance(node, dict) else getattr(node, "class_type", None)
        
        # Classify nodes by type
        if class_type in ["LoadImage", "LoadTensor", "PrimaryInputLoadImage"]:
            result["video_inputs"].append((node_id, class_type))
        elif class_type in ["PreviewImage", "SaveImage", "SaveTensor"]:
            result["video_outputs"].append((node_id, class_type))
        elif class_type in ["LoadAudioTensor", "LoadAudioTensorStream"]:
            result["audio_inputs"].append((node_id, class_type))
        elif class_type in ["SaveAudioTensor"]:
            result["audio_outputs"].append((node_id, class_type))
        elif class_type in ["SaveTextTensor"]:
            result["text_outputs"].append((node_id, class_type))

    return result


def detect_prompt_modalities(prompts: List[Union[PromptDictInput, Prompt]]) -> Dict[str, Dict[str, bool]]:
    """Detect which modalities are present across prompts based on node analysis.

    Returns a compact summary suitable for warmup decisions.

    Example return value:
        {
            "audio": {"input": True, "output": True},
            "video": {"input": True, "output": False},
            "text": {"input": False, "output": True},
        }
    """
    modalities = {
        "audio": {"input": False, "output": False},
        "video": {"input": False, "output": False},
        "text": {"input": False, "output": False},
    }

    for prompt in prompts:
        io = analyze_prompt_io(prompt)
        if io["audio_inputs"]:
            modalities["audio"]["input"] = True
        if io["audio_outputs"]:
            modalities["audio"]["output"] = True
        if io["video_inputs"]:
            modalities["video"]["input"] = True
        if io["video_outputs"]:
            modalities["video"]["output"] = True
        if io["text_inputs"]:
            modalities["text"]["input"] = True
        if io["text_outputs"]:
            modalities["text"]["output"] = True

    return modalities


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

def convert_prompt(prompt: PromptDictInput, return_dict: bool = False) -> Union[Prompt, dict]:
    """Convert and validate a ComfyUI workflow prompt."""
    Prompt.validate(prompt)
    prompt = copy.deepcopy(prompt)

    # Count node types
    primary_inputs = sum(1 for node in prompt.values() if node.get("class_type") == "PrimaryInputLoadImage")
    inputs = sum(1 for node in prompt.values() if node.get("class_type") in ["LoadImage", "LoadTensor", "LoadAudioTensor", "LoadAudioTensorStream"])
    outputs = sum(1 for node in prompt.values() if node.get("class_type") in ["PreviewImage", "SaveImage", "SaveTensor", "SaveAudioTensor", "SaveTextTensor"])

    # Validate counts
    if primary_inputs > 1 or (primary_inputs == 0 and inputs > 1):
        raise Exception("too many inputs")
    if outputs > 1:
        raise Exception("too many outputs")
    if primary_inputs + inputs == 0:
        raise Exception("missing input")
    if outputs == 0:
        raise Exception("missing output")

    # Replace nodes
    for key, node in prompt.items():
        class_type = node.get("class_type")
        if class_type == "PrimaryInputLoadImage" or (primary_inputs == 0 and class_type == "LoadImage"):
            prompt[key] = create_load_tensor_node()
        elif class_type in ["PreviewImage", "SaveImage"]:
            prompt[key] = create_save_tensor_node(node["inputs"])

    validated_prompt = Prompt.validate(prompt)
    return prompt if return_dict else validated_prompt


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

