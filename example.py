import av
import torch
import asyncio
import json
import numpy as np

from comfystream.utils import convert_prompt
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfystream import tensor_cache


async def main():
    # Configuration options:
    # https://github.com/hiddenswitch/ComfyUI/blob/89d07f3adf32a6703181343bc732bd85104bb653/comfy/cli_args_types.py#L37
    cwd = "/workspace/ComfyUI"
    config = Configuration(cwd=cwd, disable_cuda_malloc=True, gpu_only=True, comfyui_inference_log_level=None)
    

    with open("./workflows/comfystream/sd15-tensorrt-api.json", "r") as f:
        prompt = json.load(f)
    prompt = convert_prompt(prompt)

    async with EmbeddedComfyClient(config) as client:
        # Comfy will cache nodes that only need to be run once (i.e. a node that loads model weights)
        # We can run the prompt once before actual inputs come in to "warmup"
        input = torch.randn(1, 512, 512, 3)
        tensor_cache.image_inputs.put(input)
        await client.queue_prompt(prompt)
        output = await tensor_cache.image_outputs.get()
        print(output.shape)

if __name__ == "__main__":
    asyncio.run(main())