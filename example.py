import os
import json
import asyncio
import multiprocessing as mp

from comfystream.utils import convert_prompt
from comfy.distributed.executors import ProcessPoolExecutor
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfystream.tensor_cache import init_tensor_cache


async def main():
    ctx  = mp.get_context("spawn")
    manager = ctx.Manager()
    inputs_proxy = manager.Queue()
    outputs_proxy = manager.Queue()
    executor = ProcessPoolExecutor(max_workers=3, initializer=init_tensor_cache, initargs=(inputs_proxy, outputs_proxy,))
    cwd = "/Users/varshithb/Work/ComfyUI"   
    config = Configuration(cwd=cwd)
    client = EmbeddedComfyClient(config, executor=executor)

    print("PID", os.getpid())
    workflow_file = "./workflows/comfystream/tensor-utils-example-api.json"
    with open(workflow_file, "r") as f:
        prompt = convert_prompt(json.load(f))

    tasks = []
    for idx in range(3):
        inputs_proxy.put(f"Hello{idx}")
        tasks.append(client.queue_prompt(prompt))

    await asyncio.gather(*tasks)
    for idx in range(3):
        output = outputs_proxy.get()
        print(output)

if __name__ == "__main__":
    asyncio.run(main())