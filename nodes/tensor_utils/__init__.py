"""Tensor utility nodes for ComfyStream"""

from .flux_klein_stream import FluxKleinStream
from .invert_image import InvertImage
from .load_tensor import LoadTensor
from .save_fal_result import SaveFalResult
from .save_tensor import SaveTensor
from .save_text_tensor import SaveTextTensor

NODE_CLASS_MAPPINGS = {
    "LoadTensor": LoadTensor,
    "SaveTensor": SaveTensor,
    "SaveTextTensor": SaveTextTensor,
    "SaveFalResult": SaveFalResult,
    "InvertImage": InvertImage,
    "FluxKleinStream": FluxKleinStream,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InvertImage": "Invert Image",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
