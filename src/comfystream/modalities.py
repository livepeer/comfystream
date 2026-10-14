from typing import Any, Dict, List, Set, TypedDict, Union


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
    "text_output": {"SaveTextTensor", "SaveFalResult"},
}

# Flatten all input and output node types for easier checking
all_input_nodes = NODE_TYPES["video_input"] | NODE_TYPES["audio_input"] | NODE_TYPES["text_input"]
all_output_nodes = (
    NODE_TYPES["video_output"] | NODE_TYPES["audio_output"] | NODE_TYPES["text_output"]
)

# Modality mappings derived from NODE_TYPES
MODALITY_MAPPINGS = {
    "video": {
        "input": NODE_TYPES["video_input"],
        "output": NODE_TYPES["video_output"],
    },
    "audio": {
        "input": NODE_TYPES["audio_input"],
        "output": NODE_TYPES["audio_output"],
    },
    "text": {
        "input": NODE_TYPES["text_input"],
        "output": NODE_TYPES["text_output"],
    },
}

# Node types that need special handling in convert_prompt
CONVERTIBLE_NODES = {
    "PrimaryInputLoadImage": "input_replacement",
    "LoadImage": "conditional_input_replacement",
    "PreviewImage": "output_replacement",
    "SaveImage": "output_replacement",
}


def get_node_counts_by_type(prompt: Dict[Any, Any]) -> Dict[str, int]:
    """Count nodes by their functional types (primary inputs, inputs, outputs)."""
    counts = {"primary_inputs": 0, "inputs": 0, "outputs": 0}

    for node in prompt.values():
        class_type = node.get("class_type")

        if class_type == "PrimaryInputLoadImage":
            counts["primary_inputs"] += 1
        elif class_type in all_input_nodes:
            counts["inputs"] += 1
        elif class_type in all_output_nodes:
            counts["outputs"] += 1

    return counts


def get_convertible_node_keys(prompt: Dict[Any, Any]) -> Dict[str, List[str]]:
    """Collect keys of nodes that need conversion, organized by node type."""
    keys = {node_type: [] for node_type in CONVERTIBLE_NODES.keys()}

    for key, node in prompt.items():
        class_type = node.get("class_type")
        if class_type in keys:
            keys[class_type].append(key)

    return keys


def create_empty_workflow_modality() -> WorkflowModality:
    """Create an empty WorkflowModality with all capabilities set to False."""
    return {
        "video": {"input": False, "output": False},
        "audio": {"input": False, "output": False},
        "text": {"input": False, "output": False},
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
        merged = create_empty_workflow_modality()
        for prompt in prompts:
            modality = detect_io_points(prompt)
            merged = _merge_workflow_modalities(merged, modality)
        return merged

    # Initialize result
    result = create_empty_workflow_modality()

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


class CapabilityModality(TypedDict):
    """Batch capability I/O derived from a fal endpoint schema, not Comfy nodes."""

    image: ModalityIO
    video: ModalityIO
    audio: ModalityIO
    text: ModalityIO
    mesh: ModalityIO


_IMAGE_KEYS = {
    "image",
    "images",
    "image_url",
    "image_urls",
    "start_image_url",
    "end_image_url",
    "mask_url",
    "mask_image_url",
    "thumbnail",
    "texture_image_url",
}
_VIDEO_KEYS = {
    "video",
    "videos",
    "video_url",
    "video_urls",
}
_AUDIO_KEYS = {
    "audio",
    "audio_url",
    "audio_file",
    "audio_path",
}
_TEXT_KEYS = {
    "text",
    "prompt",
    "transcription",
    "caption",
    "subtitles",
}
_MESH_KEYS = {
    "model_glb",
    "model_urls",
    "glb",
    "mesh",
    "model_obj",
    "model_fbx",
}


def create_empty_capability_modality() -> CapabilityModality:
    return {
        "image": {"input": False, "output": False},
        "video": {"input": False, "output": False},
        "audio": {"input": False, "output": False},
        "text": {"input": False, "output": False},
        "mesh": {"input": False, "output": False},
    }


def _schema_components(schema_document: Dict[str, Any]) -> Dict[str, Any]:
    components = schema_document.get("components")
    if not isinstance(components, dict):
        return {}
    schemas = components.get("schemas")
    return schemas if isinstance(schemas, dict) else {}


def _resolve_schema(schema: Any, components: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
        name = ref.rsplit("/", 1)[-1]
        resolved = components.get(name)
        if isinstance(resolved, dict):
            return resolved
    return schema


def _property_names(schema: Any, components: Dict[str, Any], depth: int = 0) -> Set[str]:
    if depth > 4 or not isinstance(schema, dict):
        return set()
    resolved = _resolve_schema(schema, components)
    names: Set[str] = set()
    properties = resolved.get("properties")
    if isinstance(properties, dict):
        for key, spec in properties.items():
            if isinstance(key, str):
                names.add(key)
                names.update(_property_names(spec, components, depth + 1))
    items = resolved.get("items")
    if isinstance(items, dict):
        names.update(_property_names(items, components, depth + 1))
    for option_key in ("anyOf", "oneOf", "allOf"):
        options = resolved.get(option_key)
        if isinstance(options, list):
            for option in options:
                names.update(_property_names(option, components, depth + 1))
    return names


def _match_keys(names: Set[str], needles: Set[str]) -> bool:
    lowered = {name.lower() for name in names}
    if lowered & needles:
        return True
    for name in lowered:
        for needle in needles:
            if needle in name:
                return True
    return False


def _endpoint_hints(endpoint_id: str) -> CapabilityModality:
    result = create_empty_capability_modality()
    lowered = endpoint_id.lower()
    if any(token in lowered for token in ("image-to-video", "/i2v", "-i2v", "text-to-video", "/t2v", "-t2v", "video-to-video", "/v2v", "-v2v", "image-to-video")):
        result["video"]["output"] = True
    if any(token in lowered for token in ("text-to-image", "image-to-image", "/edit", "flux", "ideogram", "recraft", "seedream", "grok-imagine-image", "gpt-image")):
        if "video" not in lowered:
            result["image"]["output"] = True
    if any(token in lowered for token in ("whisper", "asr", "transcribe")):
        result["audio"]["input"] = True
        result["text"]["output"] = True
    if any(token in lowered for token in ("tts", "music", "sfx", "audio")):
        result["audio"]["output"] = True
    if any(token in lowered for token in ("3d", "tripo", "meshy")):
        result["mesh"]["output"] = True
    return result


def detect_capability_modality(
    schema_document: Dict[str, Any],
    endpoint_id: str = "",
) -> CapabilityModality:
    """Detect image/video/audio/text/3d I/O from a fal endpoint schema document."""

    result = create_empty_capability_modality()
    if not isinstance(schema_document, dict):
        return result

    components = _schema_components(schema_document)
    input_names = _property_names(schema_document.get("input_schema"), components)
    output_names = _property_names(schema_document.get("output_schema"), components)

    pairs = (
        ("image", _IMAGE_KEYS),
        ("video", _VIDEO_KEYS),
        ("audio", _AUDIO_KEYS),
        ("text", _TEXT_KEYS),
        ("mesh", _MESH_KEYS),
    )
    for modality, keys in pairs:
        if _match_keys(input_names, keys):
            result[modality]["input"] = True
        if _match_keys(output_names, keys):
            result[modality]["output"] = True

    hints = _endpoint_hints(endpoint_id or str(schema_document.get("endpoint_id") or ""))
    for modality, directions in hints.items():
        for direction, enabled in directions.items():
            result[modality][direction] = result[modality][direction] or enabled
    return result

