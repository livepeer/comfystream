import asyncio
from typing import List
from comfystream.exceptions import ComfyStreamInputTimeoutError
import logging

from comfystream import tensor_cache
from comfystream.utils import convert_prompt

from comfy.api.components.schema.prompt import PromptDictInput
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient

logger = logging.getLogger(__name__)


class ComfyStreamClient:
    def __init__(self, max_workers: int = 1, **kwargs):
        config = Configuration(**kwargs)
        self.comfy_client = EmbeddedComfyClient(config, max_workers=max_workers)
        self.running_prompts = {} # To be used for cancelling tasks
        self.current_prompts = []
        self._cleanup_lock = asyncio.Lock()
        self._prompt_update_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        # Add runtime error detection
        self._runtime_error_detected = asyncio.Event()
        # Add delayed execution support for modality switching
        self._execution_paused = False
        self._awaiting_first_frame = False

    async def set_prompts(self, prompts: List[PromptDictInput], timeout_override: float = None):
        """Set new prompts, replacing any existing ones.
        
        Prompts are prepared and execution will start automatically when first input frame arrives.
        This prevents timeout errors when switching between modalities and works seamlessly with warmup.
        
        Args:
            prompts: List of prompt dictionaries to set
            timeout_override: If provided, overrides timeout in LoadTensor/LoadAudioTensor nodes (for warmup)
            
        Raises:
            ValueError: If prompts list is empty
            Exception: If prompt conversion or validation fails
        """
        if not prompts:
            raise ValueError("Cannot set empty prompts list")
            
        # Cancel existing prompts first to avoid conflicts
        await self.cancel_running_prompts()
        # Reset stop event and runtime error detection for new prompts
        self._stop_event.clear()
        self._runtime_error_detected.clear()
        self._execution_paused = True
        self._awaiting_first_frame = True
        
        self.current_prompts = [convert_prompt(prompt, timeout_override=timeout_override) for prompt in prompts]
        
        if timeout_override is not None:
            logger.info(f"Applied {timeout_override}s timeout override to workflows")
        
        timeout_msg = f" with {timeout_override}s timeout" if timeout_override else ""
        logger.info(f"Prepared {len(self.current_prompts)} prompt(s){timeout_msg} - execution will start when first frame arrives")

    def _start_prompt_tasks(self):
        """Start the prompt execution tasks."""
        for idx in range(len(self.current_prompts)):
            task = asyncio.create_task(self.run_prompt(idx))
            self.running_prompts[idx] = task

    async def start_execution(self):
        """Start execution of prepared prompts (used after set_prompts with start_immediately=False)."""
        if not self._execution_paused:
            logger.warning("Execution is not paused, start_execution() has no effect")
            return
            
        if not self.current_prompts:
            raise ValueError("No prompts prepared for execution")
            
        self._execution_paused = False
        self._awaiting_first_frame = False
        logger.info(f"Starting execution of {len(self.current_prompts)} prepared prompt(s)")
        self._start_prompt_tasks()

    def _start_execution_if_awaiting(self):
        """Internal method to start execution when first frame arrives."""
        if self._awaiting_first_frame and self._execution_paused:
            self._execution_paused = False
            self._awaiting_first_frame = False
            logger.info(f"First frame received - starting execution of {len(self.current_prompts)} prompt(s)")
            self._start_prompt_tasks()

    async def update_prompts(self, prompts: List[PromptDictInput], timeout_override: float = None):
        """Update existing prompts and optionally override timeout values.
        
        Args:
            prompts: List of updated prompt dictionaries
            timeout_override: Optional timeout override for LoadTensor/LoadAudioTensor nodes
        """
        async with self._prompt_update_lock:
            # TODO: currently under the assumption that only already running prompts are updated
            if len(prompts) != len(self.current_prompts):
                raise ValueError(
                    "Number of updated prompts must match the number of currently running prompts."
                )
                
            # Validation step before updating the prompt, only meant for a single prompt for now
            for idx, prompt in enumerate(prompts):
                converted_prompt = convert_prompt(prompt, timeout_override=timeout_override)
                try:
                    await self.comfy_client.queue_prompt(converted_prompt)
                    self.current_prompts[idx] = converted_prompt
                except Exception as e:
                    raise Exception(f"Prompt update failed: {str(e)}") from e
                    
            if timeout_override is not None:
                logger.info(f"Applied {timeout_override}s timeout override to updated workflows")
            
            timeout_msg = f" with {timeout_override}s timeout" if timeout_override else ""
            logger.info(f"Updated {len(prompts)} prompt(s){timeout_msg}")


    async def run_prompt(self, prompt_index: int):
        while not self._stop_event.is_set() and not self._runtime_error_detected.is_set():
            async with self._prompt_update_lock:
                try:
                    await self.comfy_client.queue_prompt(self.current_prompts[prompt_index])
                except asyncio.CancelledError:
                    raise
                except ComfyStreamInputTimeoutError as e:
                    # Expected timeout condition - minimal logging
                    logger.info(f"Waiting for {e.input_type} input (timeout: {e.timeout_seconds}s)")
                    self._runtime_error_detected.set()
                    break
                except Exception as e:
                    # Unexpected workflow errors - log with warning level
                    logger.warning(f"Unexpected workflow issue in prompt {prompt_index}: {str(e)}")
                    self._runtime_error_detected.set()
                    break
        
        # Log why the prompt loop ended
        if self._runtime_error_detected.is_set():
            logger.info(f"Prompt {prompt_index} stopped due to runtime error detection")
        elif self._stop_event.is_set():
            logger.info(f"Prompt {prompt_index} stopped due to stop event")

    def stop_prompts_execution(self):
        """Manually stop all running prompts due to external conditions.
        
        This method triggers the same stop behavior as workflow execution errors,
        causing all prompt loops to stop immediately. Prompts will remain stopped
        until set_prompts() is called again.
        
        Useful for integration with pipeline lifecycle events like warmup completion,
        stream disconnection, or other external stop conditions.
        """
        logger.info("Manually stopping all prompt execution")
        self._runtime_error_detected.set()

    async def cleanup(self):
        # Set stop event to signal prompt loops to exit
        self._stop_event.set()
        
        await self.cancel_running_prompts()
        async with self._cleanup_lock:
            if self.comfy_client.is_running:
                try:
                    await self.comfy_client.__aexit__()
                except Exception as e:
                    logger.error(f"Error during ComfyClient cleanup: {e}")

            await self.cleanup_queues()
            logger.info("Client cleanup complete")

    async def cancel_running_prompts(self):
        async with self._cleanup_lock:
            tasks_to_cancel = list(self.running_prompts.values())
            for task in tasks_to_cancel:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self.running_prompts.clear()

        
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

    def put_video_input(self, frame):
        if tensor_cache.image_inputs.full():
            tensor_cache.image_inputs.get(block=True)
        tensor_cache.image_inputs.put(frame)
        # Start execution if we're awaiting the first frame
        self._start_execution_if_awaiting()
    
    def put_audio_input(self, frame):
        tensor_cache.audio_inputs.put(frame)
        # Start execution if we're awaiting the first frame
        self._start_execution_if_awaiting()

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
