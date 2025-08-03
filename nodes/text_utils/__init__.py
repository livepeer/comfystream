"""Text utility nodes for ComfyStream"""

from .save_text_tensor import SaveTextTensor
from .srt_generator_node import SRTGeneratorNode

NODE_CLASS_MAPPINGS = {"SaveTextTensor": SaveTextTensor, "SRTGeneratorNode": SRTGeneratorNode}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveTextTensor": "Save Text Tensor", "SRTGeneratorNode": "SRT Generator"}


__all__ = ["NODE_CLASS_MAPPINGS"]