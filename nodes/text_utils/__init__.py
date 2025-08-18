"""Text utility nodes for ComfyStream"""

from .save_text_tensor import SaveTextTensor

NODE_CLASS_MAPPINGS = {"SaveTextTensor": SaveTextTensor}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveTextTensor": "Save Text Tensor"}


__all__ = ["NODE_CLASS_MAPPINGS"]