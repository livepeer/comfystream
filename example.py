import os
import json
import torch
import asyncio

from comfystream.client import ComfyStreamClient


async def main():
    client  = ComfyStreamClient(max_workers=3, executor_type="process", cwd="/Users/varshithb/Work/ComfyUI")

    print("example.py PID", os.getpid())
    workflow_file = "./workflows/comfystream/tensor-utils-example-api.json"
    with open(workflow_file, "r") as f:
        prompt = json.load(f)

    await client.set_prompts([prompt, prompt, prompt])

    for _ in range(10):
        client.put_video_input(torch.randn(1, 512, 512, 3))

    for _ in range(10):
        output = await client.get_video_output()
        print(output.shape)

if __name__ == "__main__":
    asyncio.run(main())