import copy
import json
import os
import logging

from typing import Dict, Any, List, Tuple, Optional, Union
from comfy.api.components.schema.prompt import Prompt, PromptDictInput

logger = logging.getLogger(__name__)


def analyze_prompt_io(prompt: Union[PromptDictInput, Prompt]) -> Dict[str, List[Tuple[str, str]]]:
    """Analyze a prompt and determine input/output nodes by domain.

    Classification rules:
      - If a node's class lives under audio_utils => audio node
      - If a node's class lives under tensor_utils => video node
      - Nodes with OUTPUT_NODE=True => output nodes, others => input nodes

    Args:
        prompt: The prompt dictionary (converted or not) to analyze.

    Returns:
        Dict with keys "audio_inputs", "audio_outputs", "video_inputs", "video_outputs",
        each a list of (node_id, class_type) tuples.
    """
    class_info = {} #_load_node_class_info()

    audio_inputs: List[Tuple[str, str]] = []
    audio_outputs: List[Tuple[str, str]] = []
    video_inputs: List[Tuple[str, str]] = []
    video_outputs: List[Tuple[str, str]] = []

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
        elif class_type in ["SaveAudioTensor", "SaveTextTensor"]:
            audio_outputs.append((node_id, class_type))
            continue

        # Then check node class metadata for custom nodes
        if not info or info.get("domain") not in {"audio", "video"}:
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

    return {
        "audio_inputs": audio_inputs,
        "audio_outputs": audio_outputs,
        "video_inputs": video_inputs,
        "video_outputs": video_outputs,
    }


def detect_prompt_modalities(prompts: List[Union[PromptDictInput, Prompt]]) -> Dict[str, Dict[str, bool]]:
    """Detect which modalities are present across prompts based on node analysis.

    Returns a compact summary suitable for warmup decisions.

    Example return value:
        {
            "audio": {"input": True, "output": True},
            "video": {"input": True, "output": False},
        }
    """
    modalities = {
        "audio": {"input": False, "output": False},
        "video": {"input": False, "output": False},
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

    return modalities


def load_prompt_from_file(path: str) -> PromptDictInput:
    """Load prompts from a JSON file for warmup.

    Accepts files that are:
      - a single prompt dict
      - a list of prompt dicts
      - an object with a top-level 'prompts' list

    Returns a single prompt dictionary suitable for convert_prompt.
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

    def is_prompt_mapping(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        # Heuristic: keys look like node ids and values look like node dicts
        for key, val in obj.items():
            if not isinstance(key, str) or not isinstance(val, dict):
                return False
            if "class_type" not in val or "inputs" not in val:
                return False
        return True

    def list_nodes_to_mapping(nodes_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        mapping: Dict[str, Any] = {}
        for node in nodes_list:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id")
            if node_id is None:
                # Skip nodes without id
                continue
            class_type = node.get("class_type") or node.get("type")
            inputs = node.get("inputs", {})
            if class_type is None:
                continue
            mapping[str(node_id)] = {
                "class_type": class_type,
                "inputs": inputs,
                "_meta": node.get("_meta", {}),
            }
        return mapping

    # Normalize common wrapper formats
    prompt_candidate: Any = data
    if isinstance(data, dict):
        if "prompts" in data and isinstance(data["prompts"], list) and data["prompts"]:
            prompt_candidate = data["prompts"][0]
        elif "prompt" in data and isinstance(data["prompt"], dict):
            prompt_candidate = data["prompt"]
        elif "workflow" in data and isinstance(data["workflow"], dict):
            prompt_candidate = data["workflow"]
        elif "nodes" in data:
            if isinstance(data["nodes"], dict):
                prompt_candidate = data["nodes"]
            elif isinstance(data["nodes"], list):
                prompt_candidate = list_nodes_to_mapping(data["nodes"])
    elif isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            # Recurse into the single dict
            inner = data[0]
            if "nodes" in inner and isinstance(inner["nodes"], list):
                prompt_candidate = list_nodes_to_mapping(inner["nodes"]) 
            else:
                prompt_candidate = inner
        else:
            # Attempt to coerce list of nodes to mapping
            prompt_candidate = list_nodes_to_mapping([n for n in data if isinstance(n, dict)])

    if not is_prompt_mapping(prompt_candidate):
        raise ValueError("Unsupported warmup workflow format; expected mapping of node_id to node with class_type and inputs")

    return prompt_candidate


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
    prompt = Prompt.validate(prompt)

    return prompt
