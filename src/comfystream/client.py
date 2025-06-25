import asyncio
import logging
from typing import List, Union
import multiprocessing as mp
import os
import sys
import numpy as np
import torch
import av

from comfystream.utils import convert_prompt
from comfystream.tensor_cache import init_tensor_cache

from comfy.cli_args_types import Configuration
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.api.components.schema.prompt import PromptDictInput
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfystream.frame_proxy import FrameProxy

logger = logging.getLogger(__name__)

def _test_worker_init():
    """Test function to verify worker process initialization."""
    return os.getpid()

class ComfyStreamClient:
    def __init__(self, 
                 max_workers: int = 1, 
                 **kwargs):
        logger.info(f"[ComfyStreamClient] Main Process ID: {os.getpid()}")
        logger.info(f"[ComfyStreamClient] __init__ start, max_workers: {max_workers}")
        
        # Store default dimensions
        self.width = kwargs.get('width', 512)
        self.height = kwargs.get('height', 512)

        # Ensure workspace path is absolute
        if 'cwd' in kwargs:
            if not os.path.isabs(kwargs['cwd']):
                # Convert relative path to absolute path from current working directory
                kwargs['cwd'] = os.path.abspath(kwargs['cwd'])
            logger.info(f"[ComfyStreamClient] Using absolute workspace path: {kwargs['cwd']}")
        
        # Register TensorRT paths in main process BEFORE creating ComfyUI client
        self.register_tensorrt_paths_main_process(kwargs.get('cwd'))
        
        # Cache nodes information in main process to avoid ProcessPoolExecutor conflicts
        self._initialize_nodes_cache()
        
        logger.info("[ComfyStreamClient] Config kwargs: %s", kwargs)
        
        try:
            self.config = Configuration(**kwargs)            
            logger.info("[ComfyStreamClient] Configuration created")
            logger.info(f"[ComfyStreamClient] Current working directory: {os.getcwd()}")
                        
            logger.info("[ComfyStreamClient] Initializing process executor")
            ctx = mp.get_context("spawn")
            logger.info(f"[ComfyStreamClient] Using multiprocessing context: {ctx.get_start_method()}")
            
            manager = ctx.Manager()
            logger.info("[ComfyStreamClient] Created multiprocessing context and manager")

            self.image_inputs = manager.Queue(maxsize=50)
            self.image_outputs = manager.Queue(maxsize=50)
            self.audio_inputs = manager.Queue(maxsize=50)
            self.audio_outputs = manager.Queue(maxsize=50)
            logger.info("[ComfyStreamClient] Created manager queues")
            
            logger.info("[ComfyStreamClient] About to create ProcessPoolExecutor...")
            
            executor = ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_tensor_cache,
                initargs=(self.image_inputs, self.image_outputs, self.audio_inputs, self.audio_outputs, kwargs.get('cwd'))
            )
            logger.info("[ComfyStreamClient] ProcessPoolExecutor created successfully")
            
            # Create EmbeddedComfyClient with the executor
            logger.info("[ComfyStreamClient] Creating EmbeddedComfyClient with executor")
            self.comfy_client = EmbeddedComfyClient(self.config, executor=executor)
            logger.info("[ComfyStreamClient] EmbeddedComfyClient created successfully")
            
            # Submit a test task to ensure worker processes are initialized
            logger.info("[ComfyStreamClient] Testing worker process initialization...")
            test_future = executor.submit(_test_worker_init)
            try:
                worker_pid = test_future.result(timeout=30)  # 30 second timeout
                logger.info(f"[ComfyStreamClient] Worker process initialized successfully (PID: {worker_pid})")
            except Exception as e:
                logger.info(f"[ComfyStreamClient] Error initializing worker process: {str(e)}")
                raise
            
        except Exception as e:
            logger.info(f"[ComfyStreamClient] Error during initialization: {str(e)}")
            logger.info(f"[ComfyStreamClient] Error type: {type(e)}")
            import traceback
            logger.info(f"[ComfyStreamClient] Error traceback: {traceback.format_exc()}")
            raise
                    
        self.running_prompts = {}
        self.current_prompts = []
        self.cleanup_lock = asyncio.Lock()
        self.max_workers = max_workers
        self.worker_tasks = []
        self.next_worker = 0
        self.distribution_lock = asyncio.Lock()
        self.shutting_down = False  # Add shutdown flag
        self.distribution_task = None  # Track distribution task
        logger.info("[ComfyStreamClient] Initialized successfully")

    async def set_prompts(self, prompts: List[PromptDictInput]):
        logger.info("set_prompts start")
        self.current_prompts = [convert_prompt(prompt) for prompt in prompts]
        
        # Start the distribution manager only if not already running
        if self.distribution_task is None or self.distribution_task.done():
            self.shutting_down = False  # Reset shutdown flag
            self.distribution_task = asyncio.create_task(self.distribute_frames())
            self.running_prompts[-1] = self.distribution_task  # Use -1 as a special key for the manager
        logger.info("set_prompts end")

    async def distribute_frames(self):
        """Manager that distributes frames across workers in round-robin fashion"""
        logger.info(f"[ComfyStreamClient] Starting frame distribution manager")
        
        try:
            # Initialize worker tasks
            self.worker_tasks = []
            for worker_id in range(self.max_workers):
                worker_task = asyncio.create_task(self.worker_loop(worker_id))
                self.worker_tasks.append(worker_task)
                self.running_prompts[worker_id] = worker_task
            
            # Keep the manager running to monitor workers
            while not self.shutting_down:
                await asyncio.sleep(1.0)  # Check periodically
                
                # Only restart crashed workers if we're not shutting down
                if not self.shutting_down:
                    for worker_id, task in enumerate(self.worker_tasks):
                        if task.done():
                            # Check if the task was cancelled (graceful shutdown) or crashed
                            if task.cancelled():
                                logger.info(f"Worker {worker_id} was cancelled (graceful shutdown)")
                            else:
                                # Check if there was an exception
                                try:
                                    task.result()
                                    logger.info(f"Worker {worker_id} completed normally")
                                except Exception as e:
                                    logger.warning(f"Worker {worker_id} crashed with error: {e}, restarting")
                                    new_task = asyncio.create_task(self.worker_loop(worker_id))
                                    self.worker_tasks[worker_id] = new_task
                                    self.running_prompts[worker_id] = new_task
                                    
        except asyncio.CancelledError:
            logger.info("[ComfyStreamClient] Distribution manager cancelled")
        except Exception as e:
            logger.error(f"[ComfyStreamClient] Error in distribution manager: {e}")
        finally:
            logger.info("[ComfyStreamClient] Distribution manager stopped")

    async def worker_loop(self, worker_id: int):
        """Simple worker loop - just process frames continuously"""
        logger.info(f"[Worker {worker_id}] Started")
        
        frame_count = 0
        try:
            while not self.shutting_down:
                try:
                    # Simple round-robin prompt selection
                    prompt_index = worker_id % len(self.current_prompts)
                    current_prompt = self.current_prompts[prompt_index]
                    
                    # Just process the prompt
                    await self.comfy_client.queue_prompt(current_prompt)
                    frame_count += 1
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self.shutting_down:
                        break
                    logger.error(f"[Worker {worker_id}] Error: {e}")
                    await asyncio.sleep(0.1)
        finally:
            logger.info(f"[Worker {worker_id}] Processed {frame_count} frames")

    async def cleanup(self):
        async with self.cleanup_lock:
            logger.info("[ComfyStreamClient] Starting cleanup...")
            
            # Set shutdown flag to stop workers gracefully
            self.shutting_down = True
            
            # Cancel distribution task first
            if self.distribution_task and not self.distribution_task.done():
                self.distribution_task.cancel()
                try:
                    await self.distribution_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel all worker tasks
            for task in self.worker_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete cancellation
            if self.worker_tasks:
                try:
                    await asyncio.gather(*self.worker_tasks, return_exceptions=True)
                    logger.info("[ComfyStreamClient] All worker tasks stopped")
                except Exception as e:
                    logger.error(f"Error waiting for worker tasks: {e}")
            
            # Clear the tasks list
            self.worker_tasks.clear()
            self.running_prompts.clear()
            
            # Cleanup the ComfyUI client and its executor
            if hasattr(self, 'comfy_client') and self.comfy_client.is_running:
                try:
                    # Get the executor before closing the client
                    executor = getattr(self.comfy_client, 'executor', None)
                    
                    # Close the client first
                    await self.comfy_client.__aexit__(None, None, None)
                    logger.info("[ComfyStreamClient] ComfyUI client closed")
                    
                    # Then shutdown the executor and terminate processes
                    if executor:
                        logger.info("[ComfyStreamClient] Shutting down executor...")
                        # Shutdown the executor
                        executor.shutdown(wait=False)
                        
                        # Force terminate any remaining processes
                        if hasattr(executor, '_processes'):
                            for process in executor._processes:
                                if process.is_alive():
                                    logger.info(f"[ComfyStreamClient] Terminating worker process {process.pid}")
                                    process.terminate()
                                    # Give it a moment to terminate gracefully
                                    try:
                                        process.join(timeout=2.0)
                                    except:
                                        pass
                                    # Force kill if still alive
                                    if process.is_alive():
                                        logger.warning(f"[ComfyStreamClient] Force killing worker process {process.pid}")
                                        process.kill()
                        
                        logger.info("[ComfyStreamClient] Executor shutdown completed")
                        
                except Exception as e:
                    logger.error(f"Error during ComfyClient cleanup: {e}")

            await self.cleanup_queues()
            
            # Reset state for potential reuse
            self.shutting_down = False
            self.distribution_task = None
            
            logger.info("[ComfyStreamClient] Client cleanup complete")

    async def cleanup_queues(self):
        # TODO: add for audio as well
        while not self.image_inputs.empty():
            self.image_inputs.get()

        while not self.image_outputs.empty():
            self.image_outputs.get()

    def put_video_input(self, frame):
        try:
            # Check if frame is FrameProxy
            if isinstance(frame, FrameProxy):
                proxy = frame
            else:
                proxy = FrameProxy.avframe_to_frameproxy(frame)
            
            # Handle queue being full
            if self.image_inputs.full():
                # logger.warning(f"[ComfyStreamClient] Input queue full, dropping oldest frame")
                try:
                    self.image_inputs.get_nowait()
                except Exception:
                    pass
            
            self.image_inputs.put_nowait(proxy)
            # logger.info(f"[ComfyStreamClient] Video input queued. Queue size: {self.image_inputs.qsize()}")
        except Exception as e:
            logger.error(f"[ComfyStreamClient] Error putting video frame: {str(e)}")

    def put_audio_input(self, frame):
        self.audio_inputs.put(frame)

    async def get_video_output(self):
        try:
            logger.debug(f"[ComfyStreamClient] get_video_output called - PID: {os.getpid()}")
            tensor = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.image_outputs.get),
                timeout=5.0
            )
            # logger.info(f"[ComfyStreamClient] get_video_output returning tensor: {tensor.shape} - PID: {os.getpid()}")
            return tensor
        except asyncio.TimeoutError:
            logger.warning(f"[ComfyStreamClient] get_video_output timeout - PID: {os.getpid()}")
            return torch.zeros((1, 3, self.height, self.width), dtype=torch.float32)
        except Exception as e:
            logger.error(f"[ComfyStreamClient] Error getting video output: {str(e)} - PID: {os.getpid()}")
            return torch.zeros((1, 3, self.height, self.width), dtype=torch.float32)
    
    async def get_audio_output(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.audio_outputs.get)
    
    async def get_available_nodes(self):
        """Get metadata and available nodes info using cached nodes to avoid ProcessPoolExecutor conflicts"""
        if not self.current_prompts:
            return {}

        # Use cached nodes instead of calling import_all_nodes_in_workspace from worker process
        if self._nodes is None:
            logger.warning("[ComfyStreamClient] Nodes cache not available, returning empty result")
            return {}

        try:
            all_prompts_nodes_info = {}
            
            for prompt_index, prompt in enumerate(self.current_prompts):
                # Get set of class types we need metadata for
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
                for class_type, node_class in self._nodes.NODE_CLASS_MAPPINGS.items():
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

    def register_tensorrt_paths_main_process(self, workspace_path):
        """Register TensorRT paths in the main process for validation"""
        try:
            from comfy.cmd import folder_paths
            
            if workspace_path:
                base_dir = workspace_path
                tensorrt_models_dir = os.path.join(base_dir, "models", "tensorrt")
                tensorrt_outputs_dir = os.path.join(base_dir, "outputs", "tensorrt")
            else:
                tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt")
                tensorrt_outputs_dir = os.path.join(folder_paths.models_dir, "outputs", "tensorrt")
            
            # logger.info(f"[ComfyStreamClient] Registering TensorRT paths in main process")
            # logger.info(f"[ComfyStreamClient] TensorRT models dir: {tensorrt_models_dir}")
            # logger.info(f"[ComfyStreamClient] TensorRT outputs dir: {tensorrt_outputs_dir}")
            
            # Register TensorRT paths
            if "tensorrt" in folder_paths.folder_names_and_paths:
                existing_paths = folder_paths.folder_names_and_paths["tensorrt"][0]
                for path in [tensorrt_models_dir, tensorrt_outputs_dir]:
                    if path not in existing_paths:
                        existing_paths.append(path)
                folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
            else:
                folder_paths.folder_names_and_paths["tensorrt"] = (
                    [tensorrt_models_dir, tensorrt_outputs_dir], 
                    {".engine"}
                )
            
            # Verify registration
            # available_files = folder_paths.get_filename_list("tensorrt")
            # logger.info(f"[ComfyStreamClient] Main process TensorRT files: {available_files}")
            
        except Exception as e:
            logger.error(f"[ComfyStreamClient] Error registering TensorRT paths in main process: {e}")
            import traceback
            logger.error(f"[ComfyStreamClient] Traceback: {traceback.format_exc()}")

    async def update_prompts(self, prompts: List[PromptDictInput]):
        """Update the existing processing prompts without restarting workers."""
        
        # Simply update the current prompts - worker loops will pick up changes on next iteration
        self.current_prompts = [convert_prompt(prompt) for prompt in prompts]
        
        logger.info("[ComfyStreamClient] Prompts updated")

    def _initialize_nodes_cache(self):
        """Initialize nodes cache in main process to avoid ProcessPoolExecutor conflicts"""
        try:
            logger.info("[ComfyStreamClient] Initializing nodes cache in main process...")
            from comfy.nodes.package import import_all_nodes_in_workspace
            self._nodes = import_all_nodes_in_workspace()
            logger.info(f"[ComfyStreamClient] Cached {len(self._nodes.NODE_CLASS_MAPPINGS)} node types")
        except Exception as e:
            logger.error(f"[ComfyStreamClient] Error initializing nodes cache: {e}")
            self._nodes = None