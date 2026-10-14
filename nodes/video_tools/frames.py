"""Per-frame nodes used by the resize and rotate workflows."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from comfystream.tools.frame_slot import (
    get_input_frame,
    set_output_frame,
)


class LoadVideoFrame:
    """Return the frame the worker staged for this prompt."""

    CATEGORY = "video_tools"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def execute(self):
        array = np.asarray(get_input_frame())
        tensor = torch.from_numpy(array.astype("float32") / 255.0).unsqueeze(0)
        return (tensor,)


class SaveVideoFrame:
    """Keep the workflow output for the worker to encode."""

    CATEGORY = "video_tools"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    def execute(self, images: torch.Tensor):
        array = images[0].detach().clamp(0, 1).cpu().numpy()
        set_output_frame((array * 255.0).astype("uint8"))
        return (images,)


class RotateFlipImage:
    """Rotate clockwise, then flip. Matches the rotate tool workflow."""

    CATEGORY = "video_tools"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "rotation": ([0, 90, 180, 270], {"default": 0}),
                "flip": (["none", "horizontal", "vertical"], {"default": "none"}),
            }
        }

    def execute(self, images: torch.Tensor, rotation: int, flip: str):
        image = Image.fromarray(
            (images[0].detach().clamp(0, 1).cpu().numpy() * 255.0).astype("uint8")
        )
        if int(rotation):
            image = image.rotate(-int(rotation), expand=True)
        if flip == "horizontal":
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif flip == "vertical":
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        array = np.asarray(image.convert("RGB")).astype("float32") / 255.0
        return (torch.from_numpy(array).unsqueeze(0),)
