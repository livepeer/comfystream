import av
import json
import asyncio
import numpy as np

from comfystream.client import ComfyStreamClient



async def main():    
    client = ComfyStreamClient(cwd="/Users/varshithb/Work/ComfyUI",max_workers=3, executor_type="process")

    workflow_file = "./workflows/comfystream/tensor-utils-example-api.json"
    with open(workflow_file, "r") as f:
        prompt = json.load(f)

    await client.set_prompts([prompt, prompt, prompt])
    for idx in range(10):
        original_np_array = np.random.rand(512, 512, 3).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(original_np_array)
        frame.pts = idx
        await client.put_video_input(frame)

    for _ in range(10):
        output_frame = await client.get_video_output()
        print(output_frame.pts)


if __name__ == "__main__":
    asyncio.run(main())
