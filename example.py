import torch
import asyncio
import json
import av
from comfystream.client import ComfyStreamClient
from comfystream.utils import convert_prompt
from comfy.distributed.executors import ProcessPoolExecutor

from comfy.api.components.schema.prompt import PromptDictInput
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient

import torch
import base64
import io
from PIL import Image
import os

def tensor_to_base64_image(tensor):
    """
    Convert a 3D PyTorch tensor [H,W,C] in [0,1] range to a base64-encoded PNG.
    """
    # Clamp and scale from [0,1] -> [0,255]
    array = (tensor.clamp(0, 1).numpy() * 255).astype("uint8")
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


async def main():
    # Configuration options:
    # https://github.com/hiddenswitch/ComfyUI/blob/89d07f3adf32a6703181343bc732bd85104bb653/comfy/cli_args_types.py#L37
    executor = ProcessPoolExecutor(max_workers=1)
    cwd = "/Users/varshithb/Work/ComfyUI"
    config = Configuration(cwd=cwd)
    client = EmbeddedComfyClient(config, max_workers=2, executor=executor)

    print("PID", os.getpid())

    for _ in range(10):
        with open("./workflows/comfystream/tensor-utils-example-api.json", "r") as f:
            prompt = json.load(f)

        prompt["2"]["inputs"]["image"] = tensor_to_base64_image(torch.randn(512, 512, 3))

        prompt = convert_prompt(prompt)
        output = await client.queue_prompt(prompt)
        print(output)

    # await client.set_prompts([prompt])

    # dummy_frame = av.VideoFrame()
    # dummy_frame.side_data.input = torch.randn(1, 512, 512, 3)

    # client.put_video_input(dummy_frame)
    # output = await client.get_video_output()
    # print(output.shape)
    


if __name__ == "__main__":
    asyncio.run(main())
