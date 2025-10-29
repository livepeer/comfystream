import asyncio
import logging
from typing import List
import contextlib

from comfystream import tensor_cache
from comfystream.utils import convert_prompt
from comfystream.exceptions import ComfyStreamInputTimeoutError

from comfy.api.components.schema.prompt import PromptDictInput
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient

logger = logging.getLogger(__name__)


class ComfyStreamClient:
    def __init__(self, max_workers: int = 1, **kwargs):
        config = Configuration(**kwargs)
        self.comfy_client = EmbeddedComfyClient(config, max_workers=max_workers)
        self.running_prompts = {}
        self.current_prompts = []
        self._cleanup_lock = asyncio.Lock()
        self._prompt_update_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

        # PromptRunner state
        self._shutdown_event = asyncio.Event()
        self._run_enabled_event = asyncio.Event()
        self._runner_task = None

    async def set_prompts(self, prompts: List[PromptDictInput]):
        """Set new prompts, replacing any existing ones.
        
        Args:
            prompts: List of prompt dictionaries to set
            
        Raises:
            ValueError: If prompts list is empty
            Exception: If prompt conversion or validation fails
        """
        if not prompts:
            raise ValueError("Cannot set empty prompts list")
            
        # Pause runner while swapping prompts to avoid interleaving
        was_running = self._run_enabled_event.is_set()
        self._run_enabled_event.clear()
        self.current_prompts = [convert_prompt(prompt) for prompt in prompts]
        logger.info(f"Configured {len(self.current_prompts)} prompt(s)")
        # Ensure runner exists (IDLE until resumed)
        await self.ensure_prompt_tasks_running()
        if was_running:
            self._run_enabled_event.set()

    async def update_prompts(self, prompts: List[PromptDictInput]):
        async with self._prompt_update_lock:
            # TODO: currently under the assumption that only already running prompts are updated
            if len(prompts) != len(self.current_prompts):
                raise ValueError(
                    "Number of updated prompts must match the number of currently running prompts."
                )
            # Validation step before updating the prompt, only meant for a single prompt for now
            for idx, prompt in enumerate(prompts):
                converted_prompt = convert_prompt(prompt)
                try:
                    # Lightweight validation by queueing is retained for compatibility
                    await self.comfy_client.queue_prompt(converted_prompt)
                    self.current_prompts[idx] = converted_prompt
                except Exception as e:
                    raise Exception(f"Prompt update failed: {str(e)}") from e

    async def ensure_prompt_tasks_running(self):
        # Ensure the single runner task exists (does not force running)
        if self._runner_task and not self._runner_task.done():
            return
        if not self.current_prompts:
            return
        self._shutdown_event.clear()
        self._runner_task = asyncio.create_task(self._runner_loop())

    async def _runner_loop(self):
        try:
            while not self._shutdown_event.is_set():
                # IDLE until running is enabled
                await self._run_enabled_event.wait()
                # Snapshot prompts without holding the lock during network I/O
                async with self._prompt_update_lock:
                    prompts_snapshot = list(self.current_prompts)
                for prompt_index, prompt in enumerate(prompts_snapshot):
                    if self._shutdown_event.is_set() or not self._run_enabled_event.is_set():
                        break
                    try:
                        await self.comfy_client.queue_prompt(prompt)
                    except asyncio.CancelledError:
                        raise
                    except ComfyStreamInputTimeoutError:
                        logger.info(f"Input for prompt {prompt_index} timed out, continuing")
                        continue
                    except Exception as e:
                        logger.error(f"Error running prompt: {str(e)}")
                        await asyncio.sleep(0.05)
                        continue
        except asyncio.CancelledError:
            pass

    async def cleanup(self):
        # Signal runner to shutdown
        self._stop_event.set()
        self._shutdown_event.set()
        if self._runner_task:
            self._runner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner_task
            self._runner_task = None

        # Pause running
        self._run_enabled_event.clear()

        async with self._cleanup_lock:
            if getattr(self.comfy_client, "is_running", False):
                try:
                    await self.comfy_client.__aexit__()
                except Exception as e:
                    logger.error(f"Error during ComfyClient cleanup: {e}")

            await self.cleanup_queues()
            logger.info("Client cleanup complete")

    async def cancel_running_prompts(self):
        """Compatibility: pause the runner without destroying it."""
        self._run_enabled_event.clear()

        
    async def cleanup_queues(self):
        while not tensor_cache.image_inputs.empty():
            tensor_cache.image_inputs.get()

        while not tensor_cache.audio_inputs.empty():
            tensor_cache.audio_inputs.get()

        while not tensor_cache.image_outputs.empty():
            await tensor_cache.image_outputs.get()

        while not tensor_cache.audio_outputs.empty():
            await tensor_cache.audio_outputs.get()

        while not tensor_cache.text_outputs.empty():
            await tensor_cache.text_outputs.get()

    # Explicit lifecycle helpers for external controllers (FrameProcessor)
    def resume(self):
        self._run_enabled_event.set()

    def pause(self):
        self._run_enabled_event.clear()

    async def stop_prompts_immediately(self):
        """Cancel the runner task to immediately stop any in-flight prompt execution."""
        self._run_enabled_event.clear()
        if self._runner_task:
            self._runner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner_task
            self._runner_task = None

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
    
    async def get_text_output(self):
        try:
            return tensor_cache.text_outputs.get_nowait()
        except asyncio.QueueEmpty:
            # Expected case - queue is empty, no text available
            return None
        except Exception as e:
            # Unexpected errors logged for debugging
            logger.warning(f"Unexpected error in get_text_output: {e}")
            return None

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