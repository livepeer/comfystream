import asyncio
from typing import List, Dict, Any
import logging

from comfystream import tensor_cache
from comfystream.utils import convert_prompt
from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfy.nodes.package import import_all_nodes_in_workspace

logger = logging.getLogger(__name__)

class ComfyStreamClient:
    def __init__(self, max_workers: int = 1, **kwargs):
        # Persist configuration for rebuilds
        self._config_kwargs: Dict[str, Any] = dict(kwargs)
        self._max_workers: int = max_workers
        # Build embedded client (single-executor mode only)
        self._build_embedded()
        
        # Simplified state management
        self._active_prompt = None  # Single active prompt
        self._prompt_task = None    # Single running task
        self._cleanup_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()  # Event to signal shutdown
        self._input_event = asyncio.Event()     # Event to signal new input availability
        
        # Track running prompts for better cleanup coordination
        self._running_prompt_tasks = set()  # Track active prompt execution tasks
        
        # Always set a basic default prompt
        self._set_default_prompt()

    def _build_embedded(self):
        config = Configuration(**self._config_kwargs)
        # Always use single-executor mode to avoid VRAM growth across restarts
        self.comfy_client = EmbeddedComfyClient(config, max_workers=self._max_workers)

    def _set_default_prompt(self):
        """Set a simple default prompt that always works."""
        self._active_prompt = {
            "1": {
                "inputs": {"images": ["2", 0]},
                "class_type": "SaveTensor",
                "_meta": {"title": "SaveTensor"}
            },
            "2": {
                "inputs": {},
                "class_type": "LoadTensor",
                "_meta": {"title": "LoadTensor"}
            }
        }
        logger.info("Default prompt set (LoadTensor -> SaveTensor)")

    async def set_workflow(self, workflow: dict):
        """External method: Set a new workflow and start processing."""
        try:
            # Convert and validate workflow
            # If the workflow is already a validated Prompt (immutabledict), avoid reconverting
            if isinstance(workflow, dict):
                converted_workflow = convert_prompt(workflow)
            else:
                # Best effort: ensure it's a Prompt; this is a no-op if already validated
                converted_workflow = Prompt.validate(workflow)
            
            # Check if we're already running the same workflow to avoid unnecessary restarts
            if (self._active_prompt is not None and 
                self._active_prompt == converted_workflow and 
                self._prompt_task and not self._prompt_task.done()):
                logger.info("Workflow unchanged, skipping restart to preserve models")
                return
            
            # Only stop execution if we need to change the workflow
            if self._prompt_task and not self._prompt_task.done():
                await self._stop_current_execution()
            
            # Set as active prompt
            self._active_prompt = converted_workflow
            
            # Start processing
            await self._start_execution()
            
            logger.info("New workflow set and started")
            
        except Exception as e:
            logger.error(f"Failed to set workflow: {e}")
            # Fall back to default
            self._set_default_prompt()
            await self._start_execution()
            logger.info("Fell back to default prompt")

    async def _start_execution(self):
        """Internal method: Start executing the active prompt."""
        if self._active_prompt is None:
            self._set_default_prompt()
            
        # Clear shutdown event and input event to start fresh
        self._shutdown_event.clear()
        self._input_event.clear()
        
        # Ensure EmbeddedComfyClient context is active so cleanup runs properly later
        if not self.comfy_client.is_running:
            await self.comfy_client.__aenter__()
        
        # Start prompt execution task
        self._prompt_task = asyncio.create_task(self._run_prompt_loop())
        logger.info("Prompt execution started")

    async def _stop_current_execution(self):
        """Internal method: Stop current prompt execution."""
        await self.cancel_prompts(flush_queues=False, force=False, timeout=2.0)


    async def stop(self):
        """Public method: Stop prompt scheduling and drain queues to prepare for cleanup or restart."""
        # Set shutdown event to stop the loop
        self._shutdown_event.set()
        
        # Cancel the prompt task if running
        if self._prompt_task and not self._prompt_task.done():
            self._prompt_task.cancel()
            try:
                await self._prompt_task
            except asyncio.CancelledError:
                pass
        self._prompt_task = None
        
        # Wait for running prompts to complete or cancel them after timeout
        if self._running_prompt_tasks:
            logger.info(f"Waiting for {len(self._running_prompt_tasks)} running prompts to complete...")
            try:
                # Wait up to 3 seconds for prompts to complete naturally
                await asyncio.wait_for(
                    asyncio.gather(*self._running_prompt_tasks, return_exceptions=True),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for prompts, cancelling remaining tasks")
                # Cancel remaining tasks
                for task in self._running_prompt_tasks:
                    if not task.done():
                        task.cancel()
                # Wait for cancellations
                await asyncio.gather(*self._running_prompt_tasks, return_exceptions=True)
            finally:
                self._running_prompt_tasks.clear()
        
        # Drain tensor cache queues immediately to prevent stale data in next stream
        await self.cleanup_queues()

    async def _run_prompt_loop(self):
        """Internal method: Main prompt execution loop."""
        while not self._shutdown_event.is_set():
            try:
                # Check shutdown event
                if self._shutdown_event.is_set():
                    break
                    
                # If executor was shut down, exit loop cleanly
                if not self.comfy_client.is_running:
                    logger.warning("EmbeddedComfyClient not running; exiting prompt scheduler loop")
                    break

                # Wait for new input signal to avoid uncontrollable queuing
                try:
                    await asyncio.wait_for(self._input_event.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    # No new input yet; keep idling
                    continue

                # Queue one prompt execution
                try:
                    prompt_task = asyncio.create_task(
                        self.comfy_client.queue_prompt(self._active_prompt)
                    )
                    self._running_prompt_tasks.add(prompt_task)
                    
                    await asyncio.wait_for(prompt_task, timeout=5.0)
                    
                    # Clean up completed task
                    self._running_prompt_tasks.discard(prompt_task)
                    
                except asyncio.TimeoutError:
                    logger.debug("Prompt execution timeout in streaming mode")
                    # Clean up timed out task
                    self._running_prompt_tasks.discard(prompt_task)
                    # Do not clear the input event; try again next loop
                    continue

                # Clear the input signal if queues are empty; otherwise keep it set to drain backlog
                try:
                    if tensor_cache.image_inputs.empty() and tensor_cache.audio_inputs.empty():
                        self._input_event.clear()
                except Exception:
                    # Defensive: if queues not available, just clear
                    self._input_event.clear()
            except Exception as e:
                logger.error(f"Error in prompt loop: {str(e)}")
                # Don't cleanup here, let the calling method handle it
                raise

    async def _await_no_tasks(self, timeout: float = 2.0):
        """Wait until EmbeddedComfyClient has no queued tasks or until timeout."""
        try:
            deadline = asyncio.get_event_loop().time() + timeout
            while getattr(self.comfy_client, 'task_count', 0) > 0:
                if asyncio.get_event_loop().time() >= deadline:
                    raise TimeoutError("Comfy tasks did not drain in time")
                await asyncio.sleep(0.05)
        except Exception:
            # Don't raise in stop path
            pass

    async def cancel_prompts(self, flush_queues: bool = True, force: bool = False, timeout: float = 2.0):
        """Cooperatively cancel current and pending prompt scheduling and optionally flush queues.

        Args:
            flush_queues: Clear input/output queues after cancellation
            force: Forcefully exit EmbeddedComfyClient context if still running
            timeout: Seconds to wait for executor tasks to drain
        """
        # Signal cooperative shutdown
        self._shutdown_event.set()

        # Best-effort interrupt running execution
        try:
            if hasattr(self.comfy_client, 'interrupt'):
                self.comfy_client.interrupt()
        except Exception as e:
            logger.debug(f"Interrupt ignored: {e}")

        # Cancel scheduler task
        if self._prompt_task and not self._prompt_task.done():
            self._prompt_task.cancel()
            try:
                await self._prompt_task
            except asyncio.CancelledError:
                pass
        self._prompt_task = None

        # Wait for executor to drain
        await self._await_no_tasks(timeout=timeout)

        # Optionally flush queues and cleanup CUDA
        if flush_queues:
            try:
                await self.cleanup_queues()
            except Exception as e:
                logger.debug(f"Queue flush failed: {e}")

        # Optionally force context exit
        if force:
            try:
                if self.comfy_client.is_running:
                    await self.comfy_client.__aexit__()
            except Exception as e:
                logger.debug(f"Forced context exit error: {e}")

        logger.info("Prompt cancellation complete")

    async def cleanup(self):
        """Clean shutdown of the client."""
        async with self._cleanup_lock:
            # Stop current execution
            await self.cancel_prompts(flush_queues=True, force=False, timeout=2.0)
            
            # Shutdown embedded client
            await self._shutdown_embedded_client()
            
            # Cleanup queues
            await self.cleanup_queues()
            
            logger.info("Client cleanup complete")
    
    async def _shutdown_embedded_client(self):
        """Shutdown the EmbeddedComfyClient via its async context to free VRAM."""
        if not self.comfy_client:
            return
        try:
            if self.comfy_client.is_running:
                logger.info("Stopping EmbeddedComfyClient (context exit)...")
                await self.comfy_client.__aexit__()
                logger.info("EmbeddedComfyClient stopped and cleaned up")
            else:
                # Enter then exit to force cleanup if needed
                logger.info("Entering and exiting EmbeddedComfyClient to force cleanup")
                await self.comfy_client.__aenter__()
                await self.comfy_client.__aexit__()
                logger.info("EmbeddedComfyClient cleaned up via context cycle")
        except Exception as e:
            logger.error(f"Error during embedded client shutdown: {e}")
    


    # Legacy compatibility methods for BufferedComfyStreamProcessor
    async def set_prompts(self, prompts: List[Dict[str, Any]]):
        """Legacy method: Set prompts (now just uses first prompt)."""
        if prompts and len(prompts) > 0:
            await self.set_workflow(prompts[0])
        else:
            # Fall back to default
            self._set_default_prompt()
            await self._start_execution()

    async def update_prompts(self, prompts: List[Dict[str, Any]]):
        """Legacy method: Update prompts (same as set_prompts)."""
        await self.set_prompts(prompts)

    @property
    def running_prompts(self):
        """Legacy property: Returns dict showing if prompt is running."""
        return {0: self._prompt_task} if self._prompt_task and not self._prompt_task.done() else {}
    
    @property 
    def current_prompts(self):
        """Legacy property: Returns current active prompt as list."""
        return [self._active_prompt] if self._active_prompt else []

        
    async def cleanup_queues(self):
        while not tensor_cache.image_inputs.empty():
            tensor_cache.image_inputs.get()

        while not tensor_cache.audio_inputs.empty():
            tensor_cache.audio_inputs.get()

        while not tensor_cache.image_outputs.empty():
            await tensor_cache.image_outputs.get()

        while not tensor_cache.audio_outputs.empty():
            await tensor_cache.audio_outputs.get()

    def put_video_input(self, frame):
        # Smart frame dropping: only keep the latest frame to avoid lag
        # Clear the queue and keep only the newest frame for real-time processing
        dropped_count = 0
        while not tensor_cache.image_inputs.empty():
            try:
                tensor_cache.image_inputs.get(block=False)
                dropped_count += 1
            except:
                break
        
        tensor_cache.image_inputs.put(frame)
        # Signal input availability
        self._input_event.set()
    
    def put_audio_input(self, frame):
        tensor_cache.audio_inputs.put(frame)
        # Signal input availability
        self._input_event.set()

    async def get_video_output(self, timeout=None):
        """Get video output. 
        
        Args:
            timeout: Maximum time to wait in seconds. None for blocking, 0 for non-blocking.
                    Returns None if timeout occurs.
        """
        if timeout is None:
            return await tensor_cache.image_outputs.get()
        elif timeout == 0:
            # Non-blocking check
            if tensor_cache.image_outputs.empty():
                return None
            try:
                return await asyncio.wait_for(tensor_cache.image_outputs.get(), timeout=0.001)
            except asyncio.TimeoutError:
                return None
        else:
            # Timeout specified
            try:
                return await asyncio.wait_for(tensor_cache.image_outputs.get(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
    
    async def get_audio_output(self, timeout=None):
        """Get audio output.
        
        Args:
            timeout: Maximum time to wait in seconds. None for blocking, 0 for non-blocking.
                    Returns None if timeout occurs.
        """
        if timeout is None:
            return await tensor_cache.audio_outputs.get()
        elif timeout == 0:
            # Non-blocking check
            if tensor_cache.audio_outputs.empty():
                return None
            try:
                return await asyncio.wait_for(tensor_cache.audio_outputs.get(), timeout=0.001)
            except asyncio.TimeoutError:
                return None
        else:
            # Timeout specified
            try:
                return await asyncio.wait_for(tensor_cache.audio_outputs.get(), timeout=timeout)
            except asyncio.TimeoutError:
                return None

    async def get_available_nodes(self):
        """Get metadata and available nodes info in a single pass"""
        # TODO: make it for for multiple prompts
        if not self._running_prompt_tasks:
            return {}

        try:
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
