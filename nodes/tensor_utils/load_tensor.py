import torch
import queue
import logging
from typing import Tuple
from comfystream import tensor_cache
from comfystream.exceptions import ComfyStreamInputTimeoutError

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
                    "default": 3.0, 
                    "min": 0.1, 
                    "max": 30.0,  # Increased max for warmup scenarios
                    "step": 0.1,
                    "tooltip": "Maximum time to wait for image frames before raising an error. Use higher values (10-30s) for warmup scenarios."
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
            raise ComfyStreamInputTimeoutError("video", timeout_seconds, "ComfyStream may not be receiving input or workflow may not have video input nodes")
