import copy
import json
import os
import logging

from typing import Dict, Any, List, Tuple, Optional, Union
from comfy.api.components.schema.prompt import Prompt, PromptDictInput
from pytrickle.api import StreamParamsUpdateRequest


logger = logging.getLogger(__name__)


def analyze_prompt_io(prompt: Union[PromptDictInput, Prompt]) -> Dict[str, List[Tuple[str, str]]]:
    """Analyze a prompt and determine input/output nodes by domain.

    Classification rules:
      - If a node's class lives under audio_utils => audio node
      - If a node's class lives under tensor_utils => video node
      - If a node's class lives under text_utils => text node
      - Nodes with OUTPUT_NODE=True => output nodes, others => input nodes

    Args:
        prompt: The prompt dictionary (converted or not) to analyze.

    Returns:
        Dict with keys "audio_inputs", "audio_outputs", "video_inputs", "video_outputs",
        "text_inputs", "text_outputs", each a list of (node_id, class_type) tuples.
    """
    class_info = {} #_load_node_class_info()

    audio_inputs: List[Tuple[str, str]] = []
    audio_outputs: List[Tuple[str, str]] = []
    video_inputs: List[Tuple[str, str]] = []
    video_outputs: List[Tuple[str, str]] = []
    text_inputs: List[Tuple[str, str]] = []
    text_outputs: List[Tuple[str, str]] = []

    # Iterate without deepcopy to support Comfy Prompt model instances
    if hasattr(prompt, 'items'):
        items_iter = prompt.items()  # type: ignore[attr-defined]
    else:
        # Fallback: attempt to coerce to dict
        items_iter = dict(prompt).items()  # type: ignore[arg-type]

    for node_id, node in items_iter:
        # Support both plain dicts and model objects
        if isinstance(node, dict):
            class_type = node.get("class_type")
        else:
            class_type = getattr(node, "class_type", None)
        info = class_info.get(class_type)

        # Handle core ComfyUI nodes with hardcoded classifications
        if class_type in ["LoadImage", "LoadTensor", "PrimaryInputLoadImage"]:
            video_inputs.append((node_id, class_type))
            continue
        elif class_type in ["PreviewImage", "SaveImage", "SaveTensor"]:
            video_outputs.append((node_id, class_type))
            continue
        elif class_type in ["LoadAudioTensor", "LoadAudioTensorStream"]:
            audio_inputs.append((node_id, class_type))
            continue
        elif class_type in ["SaveAudioTensor"]:
            audio_outputs.append((node_id, class_type))
            continue
        elif class_type in ["SaveTextTensor"]:
            text_outputs.append((node_id, class_type))
            continue

        # Then check node class metadata for custom nodes
        if not info or info.get("domain") not in {"audio", "video", "text"}:
            continue

        is_output = bool(info.get("is_output", False))
        domain = info["domain"]

        if domain == "audio":
            if is_output:
                audio_outputs.append((node_id, class_type))
            else:
                audio_inputs.append((node_id, class_type))
        elif domain == "video":
            if is_output:
                video_outputs.append((node_id, class_type))
            else:
                video_inputs.append((node_id, class_type))
        elif domain == "text":
            if is_output:
                text_outputs.append((node_id, class_type))
            else:
                text_inputs.append((node_id, class_type))

    return {
        "audio_inputs": audio_inputs,
        "audio_outputs": audio_outputs,
        "video_inputs": video_inputs,
        "video_outputs": video_outputs,
        "text_inputs": text_inputs,
        "text_outputs": text_outputs,
    }


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
    """Load a prompt from a JSON file for warmup.

    Expects a single prompt dictionary mapping node IDs to node definitions.
    Each node must have 'class_type' and 'inputs' fields.
    """
    # Resolve under repository root workflows/comfystream by default
    provided = path.lstrip("./")
    
    # Validate path format
    if ".." in provided or os.path.isabs(provided):
        logger.error(f"Invalid path format: {path}")
        raise ValueError("Invalid path format")
    
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    if provided.startswith("workflows/comfystream/"):
        resolved = os.path.normpath(os.path.join(repo_root, provided))
    else:
        resolved = os.path.normpath(os.path.join(repo_root, "workflows", "comfystream", provided))
    
    # Ensure path is within expected directory structure
    workflows_dir = os.path.normpath(os.path.join(repo_root, "workflows", "comfystream"))
    if not resolved.startswith(workflows_dir):
        logger.error(f"Path {resolved} is outside workflows directory")
        raise ValueError("Path is outside workflows directory")

    if not os.path.isfile(resolved):
        logger.error(f"Warmup workflow not found at {resolved}")
        raise FileNotFoundError(resolved)

    logger.info(f"Using warmup workflow: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate that data is a prompt mapping
    if not isinstance(data, dict):
        raise ValueError("Expected JSON file to contain a dictionary mapping node IDs to node definitions")
    
    # Validate each node has required structure
    for node_id, node in data.items():
        if not isinstance(node_id, str):
            raise ValueError(f"Node ID must be a string, got {type(node_id)}")
        if not isinstance(node, dict):
            raise ValueError(f"Node {node_id} must be a dictionary, got {type(node)}")
        if "class_type" not in node:
            raise ValueError(f"Node {node_id} missing required 'class_type' field")
        if "inputs" not in node:
            raise ValueError(f"Node {node_id} missing required 'inputs' field")

    # Convert the prompt to ensure PreviewImage nodes become SaveTensor nodes for warmup
    logger.info(f"Original prompt nodes: {list(data.keys())}")
    logger.info(f"LoadImage nodes: {[k for k, v in data.items() if v.get('class_type') == 'LoadImage']}")
    logger.info(f"PreviewImage nodes: {[k for k, v in data.items() if v.get('class_type') == 'PreviewImage']}")
    
    converted_prompt = convert_prompt(data, return_dict=True)
    logger.info(f"Converted warmup prompt: {len(converted_prompt)} nodes")
    logger.info(f"LoadTensor nodes: {[k for k, v in converted_prompt.items() if v.get('class_type') == 'LoadTensor']}")
    logger.info(f"SaveTensor nodes: {[k for k, v in converted_prompt.items() if v.get('class_type') == 'SaveTensor']}")
    
    return converted_prompt


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
    """
    Convert and validate a ComfyUI workflow prompt.
    
    Args:
        prompt: The prompt dictionary to convert and validate
        return_dict: If True, return a plain dictionary. If False, return a Pydantic Prompt object.
        
    Returns:
        Either a Pydantic Prompt object (default) or a plain dictionary (if return_dict=True)
    """
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
    
    class_info = {} #_load_node_class_info()

    for key, node in prompt.items():
        class_type = node.get("class_type")

        # Collect keys for nodes that might need to be replaced
        if class_type in keys:
            keys[class_type].append(key)

        # Count inputs and outputs using node class metadata where available
        if class_type == "PrimaryInputLoadImage":
            num_primary_inputs += 1
            continue

        # Always check fallback heuristics first for core ComfyUI nodes
        if class_type in [
            "LoadImage",
            "LoadTensor",
            "LoadAudioTensor",
            "LoadAudioTensorStream",
        ]:
            num_inputs += 1
            continue
        elif class_type in [
            "PreviewImage",
            "SaveImage",
            "SaveTensor",
            "SaveAudioTensor",
            "SaveTextTensor",
        ]:
            num_outputs += 1
            continue

        # Then check node class metadata for custom nodes
        info = class_info.get(class_type)
        if info:
            if info.get("is_output", False):
                num_outputs += 1
            elif info.get("domain") in {"audio", "video"}:
                num_inputs += 1

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
    validated_prompt = Prompt.validate(prompt)

    # Return based on requested format
    if return_dict:
        # Skip final Pydantic validation and return the plain dict directly
        # This avoids any Pydantic object creation that could cause issues
        logger.debug(f"Returning plain dict directly: type={type(prompt)}, keys={list(prompt.keys())[:5]}")
        return prompt
    else:
        # Return Pydantic object (original behavior)
        logger.debug(f"Returning Pydantic object: type={type(validated_prompt)}")
        return validated_prompt


class ComfyStreamParamsUpdateRequest(StreamParamsUpdateRequest if StreamParamsUpdateRequest else object):
    """
    ComfyStream-specific parameter validation that extends pytrickle's StreamParamsUpdateRequest.
    
    Adds validation for ComfyUI workflow prompts while preserving all pytrickle parameter validation
    (width/height conversion, framerate limits, etc.).
    """
    
    def __init__(self, **data):
        """Initialize with prompt validation."""
        # Handle prompts parameter if present
        if "prompts" in data:
            prompts = data["prompts"]
            
            # Parse JSON string or list of JSON strings if needed
            if isinstance(prompts, str):
                if not prompts or prompts.strip() == "":
                    logger.info("Removing empty prompts string")
                    data.pop("prompts")
                else:
                    try:
                        parsed_prompts = json.loads(prompts)
                        if isinstance(parsed_prompts, dict):
                            data["prompts"] = parsed_prompts
                            logger.info(f"✅ Parsed JSON prompts string: {len(parsed_prompts)} nodes")
                        else:
                            logger.warning(f"Parsed JSON is not a dict: type={type(parsed_prompts)}")
                            data.pop("prompts")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse prompts JSON string: {e}")
                        data.pop("prompts")
            elif isinstance(prompts, list):
                # Handle list of JSON strings - use the first valid one
                logger.info(f"Processing list of {len(prompts)} prompt entries")
                parsed_prompt = None
                for i, prompt_item in enumerate(prompts):
                    if isinstance(prompt_item, str) and prompt_item.strip():
                        try:
                            parsed_item = json.loads(prompt_item)
                            if isinstance(parsed_item, dict):
                                parsed_prompt = parsed_item
                                logger.info(f"✅ Using prompt entry {i}: {len(parsed_item)} nodes")
                                break
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse prompt entry {i}: {e}")
                            continue
                    elif isinstance(prompt_item, dict):
                        parsed_prompt = prompt_item
                        logger.info(f"✅ Using dict prompt entry {i}: {len(prompt_item)} nodes")
                        break
                
                if parsed_prompt:
                    data["prompts"] = parsed_prompt
                else:
                    logger.warning("No valid prompts found in list, removing prompts field")
                    data.pop("prompts")
            
            # Validate prompts with ComfyStream if we have them
            if "prompts" in data:
                try:
                    # Ensure we pass a plain dict to convert_prompt
                    prompts_dict = data["prompts"]
                    if not isinstance(prompts_dict, dict):
                        logger.warning(f"Expected dict for prompts validation, got {type(prompts_dict)}")
                        data.pop("prompts")
                    else:
                        # Use convert_prompt with return_dict=True for validation and node replacement
                        validated_prompt_dict = convert_prompt(prompts_dict, return_dict=True)
                        data["prompts"] = validated_prompt_dict
                        logger.info(f"✅ ComfyUI workflow validated: {len(data['prompts'])} nodes")
                        logger.debug(f"Final prompts type in Pydantic class: {type(data['prompts'])}")
                        if data["prompts"]:
                            first_node = next(iter(data["prompts"].values()))
                            logger.debug(f"First node type: {type(first_node)}")
                except Exception as e:
                    logger.error(f"❌ ComfyUI workflow validation failed: {e}")
                    # Remove invalid prompts rather than failing entire request
                    data.pop("prompts")
        
        # Call parent constructor if available
        if StreamParamsUpdateRequest:
            super().__init__(**data)
        else:
            # Fallback if pytrickle not available
            for key, value in data.items():
                setattr(self, key, value)
    
    @classmethod
    def model_validate(cls, obj):
        """Custom validation that handles both pytrickle and ComfyStream parameters."""
        if StreamParamsUpdateRequest:
            # Create instance which will trigger validation
            instance = cls(**obj)
            return instance
        else:
            # Fallback validation
            return cls(**obj)
    
    def model_dump(self):
        """Return validated parameters as dictionary."""
        if StreamParamsUpdateRequest:
            result = super().model_dump()
            logger.debug(f"Pydantic model_dump result: type={type(result)}")
            if "prompts" in result:
                logger.debug(f"Prompts in model_dump: type={type(result['prompts'])}")
                if result["prompts"]:
                    first_node = next(iter(result["prompts"].values())) if isinstance(result["prompts"], dict) else None
                    if first_node:
                        logger.debug(f"First node in model_dump: type={type(first_node)}")
            return result
        else:
            # Fallback - return all attributes
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

