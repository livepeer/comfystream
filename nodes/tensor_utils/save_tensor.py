import torch
import numpy as np
import base64
import json # Import json for shape serialization

class SaveTensor:
    CATEGORY = "tensor_utils"
    # Output the Base64 string, dtype string, and shape string (JSON)
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("bytes_b64", "dtype", "shape",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",), # Input is still a ComfyUI IMAGE (tensor)
            }
        }

    @classmethod
    def IS_CHANGED(s, images):
        # Simple change detection based on tensor hash or similar could go here
        # For now, returning nan to always re-execute
        return float("nan")

    def execute(self, images: torch.Tensor):
        # Assuming images is [B, H, W, C], often B=1 from ComfyUI nodes
        if images.dim() == 4 and images.shape[0] == 1:
            tensor_to_process = images[0] # Use the single tensor in the batch
        else:
            # Handle other cases if necessary (e.g., direct [H, W, C] or batch > 1)
            tensor_to_process = images

        # Convert to NumPy array
        np_array = tensor_to_process.cpu().numpy()

        # Get data type and shape
        dtype_str = str(np_array.dtype)
        shape_json = json.dumps(np_array.shape)

        # Serialize to bytes
        raw_bytes = np_array.tobytes()

        # Encode bytes using Base64
        b64_encoded_str = base64.b64encode(raw_bytes).decode('utf-8')

        # Package the output data for the UI/API
        ui_output = {
            "bytes_b64": [b64_encoded_str], # Wrap in list as is typical for UI outputs
            "dtype": [dtype_str],
            "shape": [shape_json]
        }

        # Return the dictionary structured for UI/API retrieval
        # Note: The 'result' key can be omitted if this node's output
        # isn't directly connected to another node's input within the graph.
        # If it IS connected, you might need return {"ui": ui_output, "result": result_data}
        # where result_data is the tuple: (b64_encoded_str, dtype_str, shape_json,)
        return {"ui": ui_output}
