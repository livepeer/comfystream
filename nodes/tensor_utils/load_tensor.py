import torch
import queue
import logging
from typing import Tuple
from comfystream import tensor_cache

logger = logging.getLogger(__name__)


class LoadTensor:
    CATEGORY = "ComfyStream/Loaders"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    DESCRIPTION = "Load image tensor from ComfyStream input with configurable timeout. Raises exception if no input available within timeout period."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "timeout_seconds": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 10.0, 
                    "step": 0.1,
                    "tooltip": "Maximum time to wait for image frames before raising an error"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, timeout_seconds: float = 1.0) -> Tuple[torch.Tensor]:
        try:
            frame = tensor_cache.image_inputs.get(block=True, timeout=timeout_seconds)
            frame.side_data.skipped = False
            return (frame.side_data.input,)
        except queue.Empty:
            error_msg = f"No image frames available in tensor cache after {timeout_seconds}s timeout. ComfyStream may not be receiving input or the workflow may not have image input nodes."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
