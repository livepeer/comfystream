"""Tensor utility nodes for ComfyStream"""

from .load_tensor import LoadTensor
from .save_tensor import SaveTensor
from .save_text_tensor import SaveTextTensor
from .performance_nodes import PerformanceTimerNode, StartPerformanceTimerNode

NODE_CLASS_MAPPINGS = {
    "LoadTensor": LoadTensor,
    "SaveTensor": SaveTensor,
    "SaveTextTensor": SaveTextTensor,
    "PerformanceTimerNode": PerformanceTimerNode,
    "StartPerformanceTimerNode": StartPerformanceTimerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS"]
