import asyncio
import logging
from typing import List
import multiprocessing as mp

from comfystream.utils import convert_prompt
from comfystream.tensor_cache import init_tensor_cache

from comfy.cli_args_types import Configuration
from comfy.distributed.executors import ProcessPoolExecutor
from comfy.api.components.schema.prompt import PromptDictInput
from comfy.client.embedded_comfy_client import EmbeddedComfyClient

logger = logging.getLogger(__name__)


class ComfyStreamClient:
    def __init__(self, max_workers: int = 1, executor_type: str = "process", **kwargs):
        print("ComfyStreamClient __init__ start")
        config = Configuration(**kwargs)
        
        if executor_type == "process":
            ctx = mp.get_context("spawn")
            manager = ctx.Manager()
            self.image_inputs = manager.Queue()
            self.image_outputs = manager.Queue()
            executor = ProcessPoolExecutor(max_workers=max_workers, initializer=init_tensor_cache, initargs=(self.image_inputs, self.image_outputs,))
        else:
            executor = None
        self.comfy_client = EmbeddedComfyClient(config, executor=executor)

        self.running_prompts = {}
        self.current_prompts = []
        self.cleanup_lock = asyncio.Lock()

        print("ComfyStreamClient __init__ end")

    async def set_prompts(self, prompts: List[PromptDictInput]):
        print("set_prompts start")
        self.current_prompts = [convert_prompt(prompt) for prompt in prompts]
        for idx in range(len(self.current_prompts)):
            print(f"Scheduling run_prompt for idx {idx}")
            task = asyncio.create_task(self.run_prompt(idx))
            self.running_prompts[idx] = task
        print("set_prompts end")

    async def update_prompts(self, prompts: List[PromptDictInput]):
        # TODO: currently under the assumption that only already running prompts are updated
        if len(prompts) != len(self.current_prompts):
            raise ValueError(
                "Number of updated prompts must match the number of currently running prompts."
            )
        self.current_prompts = [convert_prompt(prompt) for prompt in prompts]

    async def run_prompt(self, prompt_index: int):
        print("running prompt")
        while True:
            try:
                await self.comfy_client.queue_prompt(self.current_prompts[prompt_index])
            except Exception as e:
                await self.cleanup()
                logger.error(f"Error running prompt: {str(e)}")
                raise

    async def cleanup(self):
        async with self.cleanup_lock:
            tasks_to_cancel = list(self.running_prompts.values())
            for task in tasks_to_cancel:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self.running_prompts.clear()

            if self.comfy_client.is_running:
                try:
                    await self.comfy_client.__aexit__()
                except Exception as e:
                    logger.error(f"Error during ComfyClient cleanup: {e}")


            await self.cleanup_queues()
            logger.info("Client cleanup complete")

        
    async def cleanup_queues(self):
        # TODO: add for audio as well
        while not self.image_inputs.empty():
            self.image_inputs.get()

        while not self.image_outputs.empty():
            await self.image_outputs.get()

    def put_video_input(self, frame):
        # TODO: add better frame management
        self.image_inputs.put(frame)
    
    def put_audio_input(self, frame):
        self.audio_inputs.put(frame)

    async def get_video_output(self):
        # TODO: explore making this async
        return self.image_outputs.get()
    
    async def get_audio_output(self):
        # TODO: explore making this async
        return self.audio_outputs.get()

    async def get_available_nodes(self):
        """Get metadata and available nodes info in a single pass"""
        # TODO: make it for for multiple prompts
        if not self.running_prompts:
            return {}

        try:
            from comfy.nodes.package import import_all_nodes_in_workspace
            nodes = import_all_nodes_in_workspace()

            all_prompts_nodes_info = {}
            
            for prompt_index, prompt in enumerate(self.current_prompts):
                # Get set of class types we need metadata for, excluding LoadTensor and SaveTensor
                needed_class_types = {
                    node.get('class_type') 
                    for node in prompt.values()
                }
                remaining_nodes = {
                    node_id 
                    for node_id, node in prompt.items() 
                }
                nodes_info = {}

                # Only process nodes until we've found all the ones we need
                for class_type, node_class in nodes.NODE_CLASS_MAPPINGS.items():
                    if not remaining_nodes:  # Exit early if we've found all needed nodes
                        break

                    if class_type not in needed_class_types:
                        continue

                    # Get metadata for this node type (same as original get_node_metadata)
                    input_data = node_class.INPUT_TYPES() if hasattr(node_class, 'INPUT_TYPES') else {}
                    input_info = {}

                    # Process required inputs
                    if 'required' in input_data:
                        for name, value in input_data['required'].items():
                            if isinstance(value, tuple):
                                if len(value) == 1 and isinstance(value[0], list):
                                    # Handle combo box case where value is ([option1, option2, ...],)
                                    input_info[name] = {
                                        'type': 'combo',
                                        'value': value[0],  # The list of options becomes the value
                                    }
                                elif len(value) == 2:
                                    input_type, config = value
                                    input_info[name] = {
                                        'type': input_type,
                                        'required': True,
                                        'min': config.get('min', None),
                                        'max': config.get('max', None),
                                        'widget': config.get('widget', None)
                                    }
                                elif len(value) == 1:
                                    # Handle simple type case like ('IMAGE',)
                                    input_info[name] = {
                                        'type': value[0]
                                    }
                            else:
                                logger.error(f"Unexpected structure for required input {name}: {value}")

                    # Process optional inputs with same logic
                    if 'optional' in input_data:
                        for name, value in input_data['optional'].items():
                            if isinstance(value, tuple):
                                if len(value) == 1 and isinstance(value[0], list):
                                    # Handle combo box case where value is ([option1, option2, ...],)
                                    input_info[name] = {
                                        'type': 'combo',
                                        'value': value[0],  # The list of options becomes the value
                                    }
                                elif len(value) == 2:
                                    input_type, config = value
                                    input_info[name] = {
                                        'type': input_type,
                                        'required': False,
                                        'min': config.get('min', None),
                                        'max': config.get('max', None),
                                        'widget': config.get('widget', None)
                                    }
                                elif len(value) == 1:
                                    # Handle simple type case like ('IMAGE',)
                                    input_info[name] = {
                                        'type': value[0]
                                    }
                            else:
                                logger.error(f"Unexpected structure for optional input {name}: {value}")

                    # Now process any nodes in our prompt that use this class_type
                    for node_id in list(remaining_nodes):
                        node = prompt[node_id]
                        if node.get('class_type') != class_type:
                            continue

                        node_info = {
                            'class_type': class_type,
                            'inputs': {}
                        }

                        if 'inputs' in node:
                            for input_name, input_value in node['inputs'].items():
                                input_metadata = input_info.get(input_name, {})
                                node_info['inputs'][input_name] = {
                                    'value': input_value,
                                    'type': input_metadata.get('type', 'unknown'),
                                    'min': input_metadata.get('min', None),
                                    'max': input_metadata.get('max', None),
                                    'widget': input_metadata.get('widget', None)
                                }
                                # For combo type inputs, include the list of options
                                if input_metadata.get('type') == 'combo':
                                    node_info['inputs'][input_name]['value'] = input_metadata.get('value', [])

                        nodes_info[node_id] = node_info
                        remaining_nodes.remove(node_id)

                    all_prompts_nodes_info[prompt_index] = nodes_info

            return all_prompts_nodes_info

        except Exception as e:
            logger.error(f"Error getting node info: {str(e)}")
            return {}
