import queue

import torch

from comfystream import tensor_cache
from comfystream.exceptions import ComfyStreamInputTimeoutError


class LoadTensor:
    CATEGORY = "ComfyStream/Loaders"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    DESCRIPTION = "Load image tensor from ComfyStream input with timeout."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "timeout_seconds": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 30.0,
                        "step": 0.1,
                        "tooltip": "Timeout in seconds",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Number of frames to stack into a single batch",
                    },
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, timeout_seconds: float = 1.0, batch_size: int = 1):
        frames = []
        for _ in range(batch_size):
            try:
                frame = tensor_cache.image_inputs.get(block=True, timeout=timeout_seconds)
            except queue.Empty:
                raise ComfyStreamInputTimeoutError("video", timeout_seconds)
            frame.side_data.skipped = False
            frames.append(frame.side_data.input)

        if len(frames) == 1:
            return (frames[0],)
        return (torch.cat(frames, dim=0),)
