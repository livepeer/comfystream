"""Tensor utility nodes for ComfyStream"""

from .load_tensor import LoadTensor, NODE_DISPLAY_NAME_MAPPINGS as load_tensor_display_mappings
from .save_tensor import SaveTensor, NODE_DISPLAY_NAME_MAPPINGS as save_tensor_display_mappings

NODE_CLASS_MAPPINGS = {"LoadTensor": LoadTensor, "SaveTensor": SaveTensor}

# Combine display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(load_tensor_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(save_tensor_display_mappings)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
