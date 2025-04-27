import av
import torch
import numpy as np
import asyncio

from typing import Any, Dict, Union, List
from comfystream.client import ComfyStreamClient
from utils import temporary_log_level

WARMUP_RUNS = 5


class Pipeline:
    def __init__(self, comfyui_inference_log_level: int = None, **kwargs):
        """Initialize the pipeline with the given configuration.
        Args:
            comfyui_inference_log_level: The logging level for ComfyUI inference.
                Defaults to None, using the global ComfyUI log level.
            **kwargs: Additional arguments to pass to the ComfyStreamClient
        """
        self.client = ComfyStreamClient(**kwargs)
        self.processed_audio_buffer = np.array([], dtype=np.int16)
        self._comfyui_inference_log_level = comfyui_inference_log_level

    async def warm_video(self):
        dummy_frame = av.VideoFrame.from_ndarray(np.random.randn(1, 512, 512, 3))

        for _ in range(WARMUP_RUNS):
            await self.client.put_video_input(dummy_frame)
            await self.client.get_video_output()

    async def warm_audio(self):
        dummy_frame = av.AudioFrame()
        dummy_frame.side_data.input = np.random.randint(-32768, 32767, int(48000 * 0.5), dtype=np.int16)   # TODO: adds a lot of delay if it doesn't match the buffer size, is warmup needed?
        dummy_frame.sample_rate = 48000

        for _ in range(WARMUP_RUNS):
            self.client.put_audio_input(dummy_frame)
            await self.client.get_audio_output()

    async def set_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        if isinstance(prompts, list):
            await self.client.set_prompts(prompts)
        else:
            await self.client.set_prompts([prompts])

    async def update_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        if isinstance(prompts, list):
            await self.client.update_prompts(prompts)
        else:
            await self.client.update_prompts([prompts])

    async def put_video_frame(self, frame: av.VideoFrame):
        await self.client.put_video_input(frame)

    async def put_audio_frame(self, frame: av.AudioFrame):
        await self.client.put_audio_input(frame)
    
    async def get_processed_video_frame(self):
        # TODO: make it generic to support purely generative video cases
        async with temporary_log_level("comfy", self._comfyui_inference_log_level):
            return await self.client.get_video_output()

    async def get_processed_audio_frame(self):
        # TODO: make it generic to support purely generative audio cases and also add frame skipping
        frame = await self.audio_incoming_frames.get()
        if frame.samples > len(self.processed_audio_buffer):
            async with temporary_log_level("comfy", self._comfyui_inference_log_level):
                out_tensor = await self.client.get_audio_output()
            self.processed_audio_buffer = np.concatenate([self.processed_audio_buffer, out_tensor])
        out_data = self.processed_audio_buffer[:frame.samples]
        self.processed_audio_buffer = self.processed_audio_buffer[frame.samples:]

        processed_frame = self.audio_postprocess(out_data)
        processed_frame.pts = frame.pts
        processed_frame.time_base = frame.time_base
        processed_frame.sample_rate = frame.sample_rate
        
        return processed_frame
    
    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata."""
        nodes_info = await self.client.get_available_nodes()
        return nodes_info
    
    async def cleanup(self):
        await self.client.cleanup()
