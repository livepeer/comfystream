"""FLUX.2-klein-4B realtime backend (diffusers feedback loop)."""

from __future__ import annotations

import asyncio
from typing import Any

import av

from comfystream.realtime.flux_klein import (
    DEFAULT_FEEDBACK,
    DEFAULT_GUIDANCE,
    DEFAULT_HEIGHT,
    DEFAULT_INPUT_BLEND,
    DEFAULT_MAX_TEXT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    FluxKleinModel,
)
from comfystream.realtime.spec import RealtimePipelineSpec


class FluxKleinBackend:
    def __init__(self, spec: RealtimePipelineSpec):
        options = spec.options
        self.model_id = str(options.get("model", DEFAULT_MODEL))
        self.width = int(options.get("width", DEFAULT_WIDTH))
        self.height = int(options.get("height", DEFAULT_HEIGHT))
        self.steps = int(options.get("steps", DEFAULT_STEPS))
        self.guidance = float(options.get("guidance", DEFAULT_GUIDANCE))
        self.feedback = float(options.get("feedback", DEFAULT_FEEDBACK))
        self.default_seed = int(options.get("seed", DEFAULT_SEED))
        self.default_input_blend = float(options.get("input_blend", DEFAULT_INPUT_BLEND))
        self.batch = max(1, int(options.get("batch", 1)))
        self.cpu_offload = bool(options.get("cpu_offload", False))
        self.compile_mode = options.get("compile") or None
        self.default_prompt = str(options.get("prompt", DEFAULT_PROMPT))
        self.model = FluxKleinModel()

    async def load(self) -> None:
        await asyncio.to_thread(
            self.model.load,
            model=self.model_id,
            prompt=self.default_prompt,
            width=self.width,
            height=self.height,
            steps=self.steps,
            guidance=self.guidance,
            feedback_strength=self.feedback,
            seed=self.default_seed,
            input_blend=self.default_input_blend,
            batch=self.batch,
            enable_cpu_offload=self.cpu_offload,
            compile_mode=self.compile_mode,
        )

    def defaults(self) -> dict[str, Any]:
        return {
            "prompt": self.default_prompt,
            "seed": self.default_seed,
            "input_blend": self.default_input_blend,
        }

    async def idle(self) -> None:
        return None

    def reset(self) -> None:
        self.model.reset()
        self.model.update_prompt(self.default_prompt)
        self.model.update_seed(self.default_seed)
        self.model.update_input_blend(self.default_input_blend)

    async def scrub(self, session_id: str = "") -> None:
        """Restore the default prompt. The loaded weights stay resident."""
        self.reset()
        await self.apply(self.defaults(), session_id)

    async def apply(self, params: dict[str, Any], session_id: str = "") -> None:
        if "prompt" in params:
            self.model.update_prompt(str(params["prompt"]))
        if "seed" in params and int(params["seed"]) != self.model.seed:
            self.model.update_seed(int(params["seed"]))
        if "input_blend" in params:
            self.model.update_input_blend(float(params["input_blend"]))

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "resolution": f"{self.width}x{self.height}",
            "steps": self.steps,
            "feedback_strength": self.feedback,
            "refine_steps": min(int(self.steps * self.feedback), self.steps),
            "max_text_tokens": DEFAULT_MAX_TEXT_TOKENS,
            "batch": self.batch,
            "compile": self.compile_mode,
        }

    def _transform(self, frames: list[av.VideoFrame]) -> list[av.VideoFrame]:
        rgbs = [frame.to_ndarray(format="rgb24") for frame in frames]
        outs = []
        for src, out_rgb in zip(frames, self.model.process_batch(rgbs)):
            out = av.VideoFrame.from_ndarray(out_rgb, format="rgb24")
            out.pts = src.pts
            out.time_base = src.time_base
            outs.append(out)
        return outs

    async def process(
        self, frames: list[av.VideoFrame], session_id: str = ""
    ) -> list[av.VideoFrame]:
        return await asyncio.to_thread(self._transform, frames)
