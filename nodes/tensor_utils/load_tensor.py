import torch
import queue
from comfystream import tensor_cache


class LoadTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self):
        try:
            frame = tensor_cache.image_inputs.get(block=True, timeout=1.0)
            frame.side_data.skipped = False
            return (frame.side_data.input,)
        except queue.Empty:
            # No image input available - return a black 512x512 image
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (black_image,)
