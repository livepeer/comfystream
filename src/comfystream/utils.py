import copy
import importlib
from typing import Dict, Any, Set, Union, List, TypedDict
from comfy.api.components.schema.prompt import Prompt, PromptDictInput


class ModalityIO(TypedDict):
    """Input/output capabilities for a single modality."""
    input: bool
    output: bool


class WorkflowModality(TypedDict):
    """Workflow modality detection result mapping modalities to their I/O capabilities."""
    video: ModalityIO
    audio: ModalityIO
    text: ModalityIO


# Centralized node type definitions
NODE_TYPES = {
    # Video nodes
    "video_input": {"LoadTensor", "PrimaryInputLoadImage", "LoadImage"},
    "video_output": {"SaveTensor", "PreviewImage", "SaveImage"},
    
    # Audio nodes
    "audio_input": {"LoadAudioTensor"},
    "audio_output": {"SaveAudioTensor"},
    
    # Text nodes
    "text_input": set(),  # No text input nodes currently
    "text_output": {"SaveTextTensor"},
}

# Modality mappings derived from NODE_TYPES
MODALITY_MAPPINGS = {
    "video": {
        "input": NODE_TYPES["video_input"],
        "output": NODE_TYPES["video_output"]
    },
    "audio": {
        "input": NODE_TYPES["audio_input"], 
        "output": NODE_TYPES["audio_output"]
    },
    "text": {
        "input": NODE_TYPES["text_input"],
        "output": NODE_TYPES["text_output"]
    }
}

# Node types that need special handling in convert_prompt
CONVERTIBLE_NODES = {
    "PrimaryInputLoadImage": "input_replacement",
    "LoadImage": "conditional_input_replacement", 
    "PreviewImage": "output_replacement",
    "SaveImage": "output_replacement",
}


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


def _count_nodes_by_type(prompt: Dict[Any, Any]) -> Dict[str, int]:
    """Count nodes by their functional types (primary inputs, inputs, outputs)."""
    counts = {"primary_inputs": 0, "inputs": 0, "outputs": 0}
    
    # Flatten all input and output node types for easier checking
    all_input_nodes = NODE_TYPES["video_input"] | NODE_TYPES["audio_input"] | NODE_TYPES["text_input"]
    all_output_nodes = NODE_TYPES["video_output"] | NODE_TYPES["audio_output"] | NODE_TYPES["text_output"]
    
    for node in prompt.values():
        class_type = node.get("class_type")
        
        if class_type == "PrimaryInputLoadImage":
            counts["primary_inputs"] += 1
        elif class_type in all_input_nodes:
            counts["inputs"] += 1
        elif class_type in all_output_nodes:
            counts["outputs"] += 1
            
    return counts


def _collect_convertible_node_keys(prompt: Dict[Any, Any]) -> Dict[str, List[str]]:
    """Collect keys of nodes that need conversion, organized by node type."""
    keys = {node_type: [] for node_type in CONVERTIBLE_NODES.keys()}
    
    for key, node in prompt.items():
        class_type = node.get("class_type")
        if class_type in keys:
            keys[class_type].append(key)
            
    return keys


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


def convert_prompt(prompt: PromptDictInput) -> Prompt:
    """Convert a prompt by replacing specific node types with tensor equivalents."""
    try:
        # Note: lazy import is necessary to prevent KeyError during validation
        importlib.import_module("comfy.api.components.schema.prompt_node")
    except Exception:
        pass
    
    # Validate the schema
    Prompt.validate(prompt)
    prompt = copy.deepcopy(prompt)

    # Count nodes and validate constraints
    counts = _count_nodes_by_type(prompt)
    _validate_prompt_constraints(counts)
    
    # Collect nodes that need conversion
    convertible_keys = _collect_convertible_node_keys(prompt)

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

    # Validate the processed prompt
    prompt = Prompt.validate(prompt)
    return prompt

def _create_empty_workflow_modality() -> WorkflowModality:
    """Create an empty WorkflowModality with all capabilities set to False."""
    return {
        "video": {"input": False, "output": False},
        "audio": {"input": False, "output": False},
        "text":  {"input": False, "output": False},
    }


def _merge_workflow_modalities(base: WorkflowModality, other: WorkflowModality) -> WorkflowModality:
    """Merge two WorkflowModality objects using logical OR for all capabilities."""
    for modality in base:
        for direction in base[modality]:
            base[modality][direction] = base[modality][direction] or other[modality][direction]
    return base


def detect_io_points(prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]) -> WorkflowModality:
    """Detect input/output presence per modality for a workflow.

    Returns a WorkflowModality mapping each modality to its I/O capabilities.
    This is independent from modality decisions and is used to route frames
    into/out of the pipeline and to decide passthrough behavior.
    """
    if isinstance(prompts, list):
        merged = _create_empty_workflow_modality()
        for prompt in prompts:
            modality = detect_io_points(prompt)
            merged = _merge_workflow_modalities(merged, modality)
        return merged

    # Initialize result
    result = _create_empty_workflow_modality()

    # Scan nodes and detect modality I/O points using centralized mappings
    for node in prompts.values():
        class_type = node.get("class_type", "")
        
        for modality, directions in MODALITY_MAPPINGS.items():
            if class_type in directions["input"]:
                result[modality]["input"] = True
            if class_type in directions["output"]:
                result[modality]["output"] = True

    return result


def detect_prompt_modalities(prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]) -> Set[str]:
    """Detect which modalities are used by a workflow.
    
    Returns a set of modality names that have either input or output nodes.
    This is used by the pipeline to determine which modalities need processing.
    """
    io_points = detect_io_points(prompts)
    modalities = set()
    
    for modality, capabilities in io_points.items():
        if capabilities["input"] or capabilities["output"]:
            modalities.add(modality)
    
    return modalities
