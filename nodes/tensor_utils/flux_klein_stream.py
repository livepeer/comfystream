"""Comfy node that runs the FLUX.2 Klein feedback loop on a stream frame.

Diffusers is imported on first use so other pipelines can load this package
without a FLUX.2 build of diffusers.
"""

from __future__ import annotations

import numpy as np
import torch

_MODEL = None


def reset_session() -> None:
    """Drop the previous customer's frame. The loaded weights stay resident."""
    if _MODEL is not None:
        _MODEL.reset()


class FluxKleinStream:
    CATEGORY = "ComfyStream"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    DESCRIPTION = "Realtime FLUX.2 Klein. Weights load on the first frame and stay loaded."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "a psychedelic landscape, vivid colors, intricate details", "multiline": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "input_blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 8}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1}),
                "feedback": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "width": ("INT", {"default": 384, "min": 64, "max": 1024, "step": 16}),
                "height": ("INT", {"default": 384, "min": 64, "max": 1024, "step": 16}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    @classmethod
    def reset_session(cls) -> None:
        reset_session()

    def execute(
        self,
        image: torch.Tensor,
        prompt: str,
        seed: int,
        input_blend: float,
        steps: int,
        guidance: float,
        feedback: float,
        width: int,
        height: int,
    ):
        model = _model(width, height, steps, guidance, feedback, seed, input_blend, prompt)
        model.update_prompt(str(prompt))
        model.update_seed(int(seed))
        model.update_input_blend(float(input_blend))
        frame = image[0].detach().float().clamp(0, 1).mul(255).byte().cpu().numpy()
        out = model.process(np.ascontiguousarray(frame))
        tensor = torch.from_numpy(out).float().div(255.0).unsqueeze(0)
        return (tensor,)


def _model(width, height, steps, guidance, feedback, seed, input_blend, prompt):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from comfystream.realtime.flux_klein import DEFAULT_MODEL, FluxKleinModel

    model = FluxKleinModel()
    model.load(
        model=DEFAULT_MODEL,
        prompt=str(prompt),
        width=int(width),
        height=int(height),
        steps=int(steps),
        guidance=float(guidance),
        feedback_strength=float(feedback),
        seed=int(seed),
        input_blend=float(input_blend),
    )
    _MODEL = model
    return model
