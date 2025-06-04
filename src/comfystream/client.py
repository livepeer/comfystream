import asyncio
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

from comfystream import tensor_cache
from comfystream.utils import convert_prompt

from comfy.api.components.schema.prompt import PromptDictInput
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient

import torch

logger = logging.getLogger(__name__)

@dataclass
class ErrorReport:
    """Class for reporting errors in the ComfyStreamClient."""
    timestamp: datetime
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ComfyStreamClient:
    def __init__(self, max_workers: int = 1, **kwargs):
        config = Configuration(**kwargs)
        self.comfy_client = EmbeddedComfyClient(config, max_workers=max_workers)
        self.running_prompts = {} # To be used for cancelling tasks
        self.current_prompts = []
        self.cleanup_lock = asyncio.Lock()
        self.error_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self._is_shutting_down = False

    def is_done(self) -> bool:
        """Check if the client is shutting down or done."""
        return self.shutdown_event.is_set() or self._is_shutting_down

    async def report_error(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Report an error to the error queue."""
        error = ErrorReport(
            timestamp=datetime.now(),
            error_type=error_type,
            message=message,
            details=details
        )
        await self.error_queue.put(error)
        logger.error(f"{error_type}: {message}")

    async def get_error(self) -> Optional[ErrorReport]:
        """Get the next error from the error queue."""
        try:
            return await self.error_queue.get()
        except asyncio.CancelledError:
            return None

    async def set_prompts(self, prompts: List[PromptDictInput]):
        if self.is_done():
            await self.report_error("ClientError", "Cannot set prompts while client is shutting down")
            return

        try:
            await self.cancel_running_prompts()
            self.current_prompts = [convert_prompt(prompt) for prompt in prompts]
            for idx in range(len(self.current_prompts)):
                task = asyncio.create_task(self.run_prompt(idx))
                self.running_prompts[idx] = task
        except Exception as e:
            await self.report_error("PromptError", f"Error setting prompts: {str(e)}")
            raise

    async def update_prompts(self, prompts: List[PromptDictInput]):
        if self.is_done():
            await self.report_error("ClientError", "Cannot update prompts while client is shutting down")
            return

        try:
            if len(prompts) != len(self.current_prompts):
                error_msg = "Number of updated prompts must match the number of currently running prompts."
                await self.report_error("ValidationError", error_msg)
                raise ValueError(error_msg)
            self.current_prompts = [convert_prompt(prompt) for prompt in prompts]
        except Exception as e:
            await self.report_error("PromptError", f"Error updating prompts: {str(e)}")
            raise

    async def run_prompt(self, prompt_index: int):
        while not self.is_done():
            try:
                await self.comfy_client.queue_prompt(self.current_prompts[prompt_index])
            except asyncio.CancelledError:
                logger.info(f"Prompt {prompt_index} cancelled")
                break
            except Exception as e:
                await self.report_error(
                    "PromptError",
                    f"Error running prompt {prompt_index}: {str(e)}",
                    {"prompt_index": prompt_index}
                )
                if not self.is_done():
                    await self.cleanup()
                break

    async def cleanup(self):
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        self.shutdown_event.set()

        try:
            # First cancel any running prompts
            await self.cancel_running_prompts()
            
            async with self.cleanup_lock:
                if self.comfy_client.is_running:
                    try:
                        await self.comfy_client.__aexit__()
                    except Exception as e:
                        await self.report_error(
                            "CleanupError",
                            f"Error during ComfyClient cleanup: {str(e)}"
                        )

                # Clear queues without logging individual items
                await self.cleanup_queues()
                logger.info("Client cleanup complete")
        except Exception as e:
            await self.report_error("CleanupError", f"Error during cleanup: {str(e)}")
            raise

    async def cancel_running_prompts(self):
        async with self.cleanup_lock:
            tasks_to_cancel = list(self.running_prompts.values())
            for task in tasks_to_cancel:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self.running_prompts.clear()

    async def cleanup_queues(self):
        try:
            # Clear queues and move CUDA tensors to CPU before discarding
            while not tensor_cache.image_inputs.empty():
                try:
                    tensor = tensor_cache.image_inputs.get_nowait()
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        tensor.cpu()
                except Exception as e:
                    logger.error(f"Error clearing image inputs queue: {e}")

            while not tensor_cache.audio_inputs.empty():
                try:
                    tensor = tensor_cache.audio_inputs.get_nowait()
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        tensor.cpu()
                except Exception as e:
                    logger.error(f"Error clearing audio inputs queue: {e}")

            while not tensor_cache.image_outputs.empty():
                try:
                    tensor = await tensor_cache.image_outputs.get_nowait()
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        tensor.cpu()
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Error clearing image outputs queue: {e}")

            while not tensor_cache.audio_outputs.empty():
                try:
                    tensor = await tensor_cache.audio_outputs.get_nowait()
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        tensor.cpu()
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Error clearing audio outputs queue: {e}")

            # Clear CUDA cache after all tensors are moved to CPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")

        except Exception as e:
            await self.report_error("QueueError", f"Error cleaning up queues: {str(e)}")
            raise

    def put_video_input(self, frame):
        if self.is_done():
            logger.warning("Cannot put video input while client is shutting down")
            return

        try:
            if tensor_cache.image_inputs.full():
                tensor_cache.image_inputs.get(block=True)
            tensor_cache.image_inputs.put(frame)
        except Exception as e:
            logger.error(f"Error putting video input: {str(e)}")
            raise
    
    def put_audio_input(self, frame):
        if self.is_done():
            logger.warning("Cannot put audio input while client is shutting down")
            return

        try:
            tensor_cache.audio_inputs.put(frame)
        except Exception as e:
            logger.error(f"Error putting audio input: {str(e)}")
            raise

    async def get_video_output(self):
        if self.is_done():
            # Instead of raising an error, return None to indicate stream end
            return None

        try:
            return await tensor_cache.image_outputs.get()
        except Exception as e:
            await self.report_error("OutputError", f"Error getting video output: {str(e)}")
            raise
    
    async def get_audio_output(self):
        if self.is_done():
            # Instead of raising an error, return None to indicate stream end
            return None

        try:
            return await tensor_cache.audio_outputs.get()
        except Exception as e:
            await self.report_error("OutputError", f"Error getting audio output: {str(e)}")
            raise

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
