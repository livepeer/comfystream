import asyncio
from asyncio import QueueEmpty as AsyncQueueEmpty
from queue import Empty as SyncQueueEmpty
from typing import List
import logging

from comfystream import tensor_cache
from comfystream.utils import convert_prompt

from comfy.api.components.schema.prompt import PromptDictInput
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import Comfy

logger = logging.getLogger(__name__)


class ComfyStreamClient:
    def __init__(self, max_workers: int = 1, **kwargs):
        config = Configuration(**kwargs)
        self.max_workers = max_workers
        self.config = config
        self.running_prompts = {}
        self.current_prompts = []
        self._cleanup_lock = asyncio.Lock()
        self._prompt_update_lock = asyncio.Lock()
        self._client_cm = None
        self._client = None

    async def set_prompts(self, prompts: List[PromptDictInput]):
        await self.cancel_running_prompts()
        # Lazily start a single embedded client per session (no context manager usage)
        if self._client is None:
            try:
                self._client_cm = Comfy(**vars(self.config))
            except Exception:
                self._client_cm = Comfy()
            self._client = await self._client_cm.__aenter__()

        self.current_prompts = [convert_prompt(prompt) for prompt in prompts]
        for idx in range(len(self.current_prompts)):
            task = asyncio.create_task(self.run_prompt(idx))
            self.running_prompts[idx] = task

    async def update_prompts(self, prompts: List[PromptDictInput]):
        async with self._prompt_update_lock:
            # TODO: currently under the assumption that only already running prompts are updated
            if len(prompts) != len(self.current_prompts):
                raise ValueError(
                    "Number of updated prompts must match the number of currently running prompts."
                )
            # Update in-place; active run loops will pick the new prompts on next iteration
            for idx, prompt in enumerate(prompts):
                converted_prompt = convert_prompt(prompt)
                self.current_prompts[idx] = converted_prompt

    async def _queue_with_client(self, comfy_client, prompt):
        # Prefer progress API if available
        if hasattr(comfy_client, "queue_with_progress"):
            task = comfy_client.queue_with_progress(prompt)
            async for _ in task.progress():
                pass
            return
        if hasattr(comfy_client, "queue_prompt"):
            return await comfy_client.queue_prompt(prompt)
        if hasattr(comfy_client, "queue"):
            return await comfy_client.queue(prompt)
        raise AttributeError("Comfy client does not support known queue methods")

    async def run_prompt(self, prompt_index: int):
        # Reuse the single embedded client started in set_prompts
        try:
            while True:
                async with self._prompt_update_lock:
                    prompt = self.current_prompts[prompt_index]
                await self._queue_with_client(self._client, prompt)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.cleanup()
            logger.error(f"Error running prompt: {str(e)}")
            raise

    async def cleanup(self):
        await self.cancel_running_prompts()
        async with self._cleanup_lock:
            await self.cleanup_queues()
            logger.info("Client cleanup complete")

    async def close(self):
        # Full teardown of the embedded client (for process shutdown)
        await self.cancel_running_prompts()
        async with self._cleanup_lock:
            try:
                await self.cleanup_queues()
            except Exception:
                pass
            if self._client_cm is not None:
                try:
                    await self._client_cm.__aexit__(None, None, None)
                except Exception:
                    pass
            self._client = None
            self._client_cm = None

    async def cancel_running_prompts(self):
        async with self._cleanup_lock:
            tasks_to_cancel = list(self.running_prompts.values())
            # Request cancellation first for all tasks
            for task in tasks_to_cancel:
                task.cancel()
            # Then wait with a timeout so we can never hang here
            for task in tasks_to_cancel:
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except asyncio.CancelledError:
                    pass
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for prompt task to cancel; continuing cleanup")
                except Exception:
                    # Do not block cleanup on task errors
                    pass
            self.running_prompts.clear()

        
    async def cleanup_queues(self):
        # Drain synchronous input queues without blocking
        while True:
            try:
                tensor_cache.image_inputs.get_nowait()
            except SyncQueueEmpty:
                break

        while True:
            try:
                tensor_cache.audio_inputs.get_nowait()
            except SyncQueueEmpty:
                break

        # Drain async output queues without awaiting indefinitely
        while True:
            try:
                tensor_cache.image_outputs.get_nowait()
            except AsyncQueueEmpty:
                break

        while True:
            try:
                tensor_cache.audio_outputs.get_nowait()
            except AsyncQueueEmpty:
                break

    def put_video_input(self, frame):
        if tensor_cache.image_inputs.full():
            tensor_cache.image_inputs.get(block=True)
        tensor_cache.image_inputs.put(frame)
    
    def put_audio_input(self, frame):
        tensor_cache.audio_inputs.put(frame)

    async def get_video_output(self):
        return await tensor_cache.image_outputs.get()
    
    async def get_audio_output(self):
        return await tensor_cache.audio_outputs.get()

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
