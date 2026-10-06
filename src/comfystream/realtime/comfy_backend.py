"""ComfyStream workflow realtime backend (e.g. StreamDiffusion TensorRT sd-turbo)."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, cast

import av
import numpy as np
import torch

from comfystream.pipeline import Pipeline
from comfystream.realtime.spec import RealtimePipelineSpec

_REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PARAMS = ("prompt", "negative_prompt")
TEXT_PARAMS = {"prompt", "negative_prompt"}
# The first frame loads models and may build TensorRT engines for a new GPU arch.
FIRST_FRAME_TIMEOUT_S = 1800.0


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO_ROOT / candidate


def engine_dir_for_device(engine_root: str) -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return str(Path(engine_root) / f"sm{major}{minor}")


class ComfyWorkflowBackend:
    def __init__(self, spec: RealtimePipelineSpec):
        options = spec.options
        if "workflow" not in options:
            raise ValueError(f"{spec.name}: comfy_workflow backend requires options.workflow")
        self.workflow_path = _resolve(str(options["workflow"]))
        self.width = int(options.get("width", 512))
        self.height = int(options.get("height", 512))
        self.workspace = str(options.get("workspace", "/workspace/ComfyUI"))
        self.engine_root = str(options.get("engine_root", ""))
        self.prompt_nodes = [str(node) for node in options.get("prompt_nodes", [])]
        declared = options.get("params")
        self.param_names = [str(name) for name in declared] if declared else list(PROMPT_PARAMS)
        self.blacklist_custom_nodes = list(
            options.get("blacklist_custom_nodes", ["ComfyUI-Manager"])
        )
        self.batch = 1
        self.workflow: dict[str, Any] = {}
        self.workflow_sha256 = ""
        self._defaults: dict[str, str] = {}
        self.params: dict[str, str] = {}
        self.applied: dict[str, Any] | None = None
        self.pipeline: Pipeline | None = None
        self._frame_lock = asyncio.Lock()

    def _prompt_targets(self) -> list[str]:
        if self.prompt_nodes:
            return self.prompt_nodes
        return [
            node_id
            for node_id, node in self.workflow.items()
            if isinstance(node.get("inputs", {}).get("prompt"), str)
        ]

    def _render_workflow(self) -> dict[str, Any]:
        workflow = copy.deepcopy(self.workflow)
        for node_id in self._prompt_targets():
            workflow[node_id]["inputs"].update(self.params)
        return workflow

    async def load(self) -> None:
        raw = self.workflow_path.read_bytes()
        self.workflow_sha256 = hashlib.sha256(raw).hexdigest()
        workflow = json.loads(raw)
        if self.engine_root:
            engine_dir = engine_dir_for_device(self.engine_root)
            for node in workflow.values():
                if "engine_dir" in node.get("inputs", {}):
                    node["inputs"]["engine_dir"] = engine_dir
        self.workflow = workflow
        targets = self._prompt_targets()
        if targets:
            inputs = workflow[targets[0]]["inputs"]
            self._defaults = {
                key: (str(inputs[key]) if key in TEXT_PARAMS else inputs[key])
                for key in self.param_names
                if key in inputs
            }
        self.params = dict(self._defaults)
        self.pipeline = Pipeline(
            width=self.width,
            height=self.height,
            cwd=self.workspace,
            disable_cuda_malloc=True,
            gpu_only=True,
            preview_method="none",
            blacklist_custom_nodes=self.blacklist_custom_nodes,
            bootstrap_default_prompt=False,
            max_workers=1,
        )
        self.applied = self._render_workflow()
        await self.pipeline.apply_prompts([self.applied], skip_warmup=True)
        frame = av.VideoFrame.from_ndarray(
            np.zeros((self.height, self.width, 3), dtype=np.uint8), format="rgb24"
        )
        frame.pts = 0
        frame.time_base = Fraction(1, 30)
        await asyncio.wait_for(self.process([frame]), timeout=FIRST_FRAME_TIMEOUT_S)
        self._require_workflow_active()
        # The warmup frame is done. Stop the runner so it does not sit in LoadTensor
        # between sessions. Weights and engines stay loaded; cold unload is separate.
        await self.idle()

    def _require_workflow_active(self) -> None:
        """Fail the load when ComfyStream fell back to its passthrough workflow."""
        expected = {node["class_type"] for node in self.workflow.values()}
        active = self.pipeline.client.current_prompts if self.pipeline else []
        running = {
            node["class_type"]
            for prompt in active
            for node in cast(Mapping[str, Any], prompt).values()
        }
        if not expected <= running:
            raise RuntimeError(
                f"workflow {self.workflow_path.name} did not run "
                f"(missing nodes: {sorted(expected - running)}); see ComfyUI errors above"
            )

    def defaults(self) -> dict[str, Any]:
        return dict(self._defaults)

    def reset(self) -> None:
        self.params = dict(self._defaults)

    async def scrub(self) -> None:
        """Put the workflow prompt back to the file default. Models stay loaded."""
        self.reset()
        await self.apply(self.defaults())
        try:
            from nodes.tensor_utils.flux_klein_stream import reset_session
        except ImportError:
            return
        reset_session()

    async def idle(self) -> None:
        """Stop prompt execution without unloading the pipeline."""
        if self.pipeline is not None and self.pipeline.are_prompts_running():
            await self.pipeline.stop_streaming()

    async def apply(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            if key not in self.param_names:
                continue
            self.params[key] = str(value) if key in TEXT_PARAMS else value
        rendered = self._render_workflow()
        if self.pipeline is None or rendered == self.applied:
            return
        # apply_prompts swaps the stored workflow without executing it. update_prompts
        # validates by running the graph, which blocks here waiting for an input frame.
        # The swap cancels in-flight execution, so it must land between frames.
        async with self._frame_lock:
            await self.pipeline.apply_prompts([rendered], skip_warmup=True)
        self.applied = rendered

    def describe(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow_path.name,
            "workflow_sha256": self.workflow_sha256[:12],
            "resolution": f"{self.width}x{self.height}",
            "batch": self.batch,
        }

    async def process(self, frames: list[av.VideoFrame]) -> list[av.VideoFrame]:
        if self.pipeline is None:
            raise RuntimeError("pipeline not loaded")
        async with self._frame_lock:
            for frame in frames:
                if frame.width != self.width or frame.height != self.height:
                    frame = frame.reformat(width=self.width, height=self.height, format="rgb24")
                await self.pipeline.put_video_frame(frame)
            if not self.pipeline.are_prompts_running():
                await self.pipeline.start_streaming()
            return [await self.pipeline.get_processed_video_frame() for _ in frames]
