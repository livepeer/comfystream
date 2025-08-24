import copy
import random

from typing import Dict, Any

import logging
logger = logging.getLogger(__name__)

def create_load_tensor_node():
    return {
        "inputs": {
            "tensor_data": ""  # Empty tensor data that will be filled at runtime
        },
        "class_type": "LoadTensorAPI",
        "_meta": {"title": "Load Tensor (API)"},
    }

def create_load_image_base64_node():
    return {
        "inputs": {
            "image": ""  # Should be "image" not "image_data" to match LoadImageBase64
        },
        "class_type": "LoadImageBase64",
        "_meta": {"title": "Load Image Base64 (ComfyStream)"},
    }

def create_save_tensor_node(inputs: Dict[Any, Any]):
    """Create a SaveTensorAPI node with proper input formatting"""
    # Make sure images input is properly formatted [node_id, output_index]
    images_input = inputs.get("images")
    
    # If images input is not properly formatted as [node_id, output_index]
    if not isinstance(images_input, list) or len(images_input) != 2:
        print(f"Warning: Invalid images input format: {images_input}, using default")
        images_input = ["", 0]  # Default empty value
    
    return {
        "inputs": {
            "images": images_input,  # Should be [node_id, output_index]
            "format": "png",  # Better default than JPG for quality
            "quality": 95
        },
        "class_type": "SaveTensorAPI", 
        "_meta": {"title": "Save Tensor (API)"},
    }

def create_send_image_websocket_node(inputs: Dict[Any, Any]):
    # Get the correct image input reference
    images_input = inputs.get("images", inputs.get("image"))
    
    # If not properly formatted, use default
    if not images_input:
        images_input = ["", 0]  # Default empty value
    
    return {
        "inputs": {
            "images": images_input,
            "format": "PNG"  # Default format
        },
        "class_type": "SendImageWebsocket",    
        "_meta": {"title": "Send Image Websocket (ComfyStream)"},
    }

def create_send_tensor_websocket_node(inputs: Dict[Any, Any]):
    # Get the correct image input reference
    tensor_input = inputs.get("images", inputs.get("tensor"))
    
    if not tensor_input:
        logging.warning("No valid tensor input found for SendTensorWebSocket node")
        tensor_input = ["", 0]  # Default empty value
    
    return {
        "inputs": {
            "tensor": tensor_input
        },
        "class_type": "SendTensorWebSocket",
        "_meta": {"title": "Save Tensor WebSocket (ComfyStream)"},
    }

def convert_prompt(prompt):
    logging.info("Converting prompt: %s", prompt)

    # Initialize counters
    num_primary_inputs = 0
    num_inputs = 0
    num_outputs = 0

    keys = {
        "PrimaryInputLoadImage": [],
        "LoadImage": [],
        "PreviewImage": [],
        "SaveImage": [],
    }

    # Set random seeds for any seed nodes
    for key, node in prompt.items():
        if not isinstance(node, dict) or "inputs" not in node:
            continue
            
        # Check if this node has a seed input directly
        if "seed" in node.get("inputs", {}):
            # Generate a random seed (same range as JavaScript's Math.random() * 18446744073709552000)
            random_seed = random.randint(0, 18446744073709551615)
            node["inputs"]["seed"] = random_seed
            logger.debug(f"Set random seed {random_seed} for node {key}")
    
    for key, node in prompt.items():
        class_type = node.get("class_type")

        # Collect keys for nodes that might need to be replaced
        if class_type in keys:
            keys[class_type].append(key)

        # Count inputs and outputs
        if class_type == "PrimaryInputLoadImage":
            num_primary_inputs += 1
        elif class_type in ["LoadImage", "LoadImageBase64"]:
            num_inputs += 1
        elif class_type in ["PreviewImage", "SaveImage", "SendImageWebsocket", "SendTensorWebSocket"]:
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

    # Replace nodes with proper implementations
    for key in keys["PrimaryInputLoadImage"]:
        prompt[key] = create_load_image_base64_node()

    if num_primary_inputs == 0 and len(keys["LoadImage"]) == 1:
        prompt[keys["LoadImage"][0]] = create_load_image_base64_node()

    for key in keys["PreviewImage"] + keys["SaveImage"]:
        node = prompt[key]
        # prompt[key] = create_save_image_node(node["inputs"]) 
        prompt[key] = create_send_image_websocket_node(node["inputs"]) # TESTING

    # TODO: Validate the processed prompt input
            
    return prompt
