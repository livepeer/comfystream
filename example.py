import os
import json
import time
import asyncio
import numpy as np
import base64

from comfystream.utils import convert_prompt
from comfy.distributed.executors import ProcessPoolExecutor
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient


async def main():
    executor = ProcessPoolExecutor(max_workers=3)
    cwd = "/Users/varshithb/Work/ComfyUI"
    config = Configuration(cwd=cwd)
    client = EmbeddedComfyClient(config, executor=executor)

    print("PID", os.getpid())
    workflow_file = "./workflows/comfystream/tensor-utils-example-api.json"
    with open(workflow_file, "r") as f:
        prompt = json.load(f)

    original_np_array = np.random.rand(512, 512, 3).astype(np.float16)
    raw_bytes = base64.b64encode(original_np_array.tobytes()).decode("utf-8")
    prompt["2"]["inputs"]["bytes"] = raw_bytes
    prompt = convert_prompt(prompt)
    

    output = await client.queue_prompt(prompt)
    print(type(output["1"]["results"][0]))
    output_bytes = output["1"]["results"][0]
    
    output_bytes = base64.b64decode(output_bytes)
    deserialized_np_array = np.frombuffer(output_bytes, dtype=original_np_array.dtype).reshape(original_np_array.shape)
    assert np.array_equal(original_np_array, deserialized_np_array), "Deserialized array does not match the original!"
    print("Successfully serialized, sent, received, and deserialized the tensor. Arrays match!")


if __name__ == "__main__":
    asyncio.run(main())
