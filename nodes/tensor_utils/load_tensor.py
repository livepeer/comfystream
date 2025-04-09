import os
import torch
import numpy as np
import base64
import json

class LoadTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",) # Standard ComfyUI name for tensor output
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bytes_b64": ("STRING", {"multiline": True}), # Input Base64 string
                "dtype": ("STRING", {"multiline": False}),    # Input dtype string
                "shape": ("STRING", {"multiline": False}),    # Input shape string (JSON)
            }
        }

    @classmethod
    def IS_CHANGED(s, bytes_b64, dtype, shape):
        # Can implement change detection based on inputs if needed
        return float("nan")

    def execute(self, bytes_b64: str, dtype: str, shape: str):
        print("LoadTensor PID", os.getpid())
        try:
            # Decode Base64
            raw_bytes = base64.b64decode(bytes_b64)

            # Parse shape from JSON string
            shape_tuple = tuple(json.loads(shape))

            # Determine numpy dtype
            np_dtype = np.dtype(dtype)

            # Reconstruct NumPy array - add .copy() here
            np_array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape_tuple).copy()

            # Convert to PyTorch tensor
            tensor = torch.from_numpy(np_array)

            # Add batch dimension (ComfyUI expects [B, H, W, C])
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

            return (tensor,)
        except Exception as e:
            print(f"Error in LoadTensor: {e}")
            # Return a dummy tensor or raise error? Returning None might break workflows.
            # Let's return a small black tensor as a fallback.
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
