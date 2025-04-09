import os
import json
import time
import base64
import asyncio
import numpy as np

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

    serialize_times = []
    queue_times = []
    deserialize_times = []
    num_iterations = 10

    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        workflow_file = "./workflows/comfystream/tensor-utils-example-api.json"
        try:
            with open(workflow_file, "r") as f:
                prompt = json.load(f)
        except FileNotFoundError:
            print(f"Error: Workflow file not found at {workflow_file}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {workflow_file}")
            return

        original_np_array = np.random.rand(512, 512, 3).astype(np.float32)

        serialize_start = time.time()
        raw_bytes = original_np_array.tobytes()
        bytes_b64 = base64.b64encode(raw_bytes).decode('utf-8')
        dtype_str = str(original_np_array.dtype)
        shape_json = json.dumps(original_np_array.shape)
        serialize_end = time.time()
        serialize_duration = serialize_end - serialize_start
        serialize_times.append(serialize_duration)
        print(f"  Serialization took: {serialize_duration:.4f} seconds")

        load_node_id = "2"
        try:
            prompt[load_node_id]["inputs"]["bytes_b64"] = bytes_b64
            prompt[load_node_id]["inputs"]["dtype"] = dtype_str
            prompt[load_node_id]["inputs"]["shape"] = shape_json
        except KeyError:
             print(f"  Error: Could not find inputs for node '{load_node_id}' in workflow. Check node ID and workflow structure.")
             continue

        prompt = convert_prompt(prompt)

        queue_start = time.time()
        output = await client.queue_prompt(prompt)
        queue_end = time.time()
        queue_duration = queue_end - queue_start
        queue_times.append(queue_duration)
        print(f"  Queue prompt took: {queue_duration:.4f} seconds")

        deserialize_start = time.time()
        reconstructed_np_array = None
        save_node_id = "1"
        try:
            save_output_ui = output[save_node_id]
            received_b64_str = save_output_ui['bytes_b64'][0]
            received_dtype_str = save_output_ui['dtype'][0]
            received_shape_json = save_output_ui['shape'][0]

            received_raw_bytes = base64.b64decode(received_b64_str)
            received_shape_tuple = tuple(json.loads(received_shape_json))
            received_dtype = np.dtype(received_dtype_str)

            reconstructed_np_array = np.frombuffer(received_raw_bytes, dtype=received_dtype).reshape(received_shape_tuple)

        except KeyError as e:
             print(f"  Error accessing output from node '{save_node_id}': {e}. Check node ID and output structure.")
             print("  Full output received:", json.dumps(output, indent=2))
             continue
        except (TypeError, ValueError, json.JSONDecodeError, base64.binascii.Error) as e:
            print(f"  Error during deserialization: {e}")
            continue

        deserialize_end = time.time()
        deserialize_duration = deserialize_end - deserialize_start
        deserialize_times.append(deserialize_duration)
        print(f"  Deserialization took: {deserialize_duration:.4f} seconds")

        if reconstructed_np_array is not None:
            try:
                np.testing.assert_allclose(original_np_array, reconstructed_np_array, rtol=1e-6, atol=1e-6)
                print("  ✅ Assertion Passed: Arrays are close.")
            except AssertionError as e:
                print(f"  ❌ Assertion Failed: Arrays differ! {e}")

    if not serialize_times or not queue_times or not deserialize_times:
        print("\nNo successful iterations to calculate median times.")
        return

    median_serialize = np.median(serialize_times)
    median_queue = np.median(queue_times)
    median_deserialize = np.median(deserialize_times)
    total_cycle_times = [s + q + d for s, q, d in zip(serialize_times, queue_times, deserialize_times)]
    median_total_cycle = np.median(total_cycle_times)

    print(f"\n--- Medians over {len(serialize_times)} successful iterations ---")
    print(f"  Serialization:    {median_serialize:.4f} seconds")
    print(f"  Queue Prompt:     {median_queue:.4f} seconds")
    print(f"  Deserialization:  {median_deserialize:.4f} seconds")
    print(f"  Total Cycle:      {median_total_cycle:.4f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
