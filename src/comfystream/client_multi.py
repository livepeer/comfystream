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
from comfystream.tensor_cache_multi import init_tensor_cache

from comfy.cli_args_types import Configuration
from comfy.distributed.executors import ProcessPoolExecutor  # Use ComfyUI's executor
from comfy.api.components.schema.prompt import PromptDictInput
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfystream.frame_proxy import FrameProxy

logger = logging.getLogger(__name__)

def _test_worker_init():
    """Test function to verify worker process initialization."""
    return os.getpid()

class ComfyStreamClient:
    def __init__(self, max_workers: int = 1, executor_type: str = "process", **kwargs):
        logger.info(f"[ComfyStreamClient] Main Process ID: {os.getpid()}")
        logger.info("[ComfyStreamClient] __init__ start, max_workers:", max_workers, "executor_type:", executor_type)
        
        # Store default dimensions
        self.width = kwargs.get('width', 512)
        self.height = kwargs.get('height', 512)
        
        # Ensure workspace path is absolute
        if 'cwd' in kwargs and not os.path.isabs(kwargs['cwd']):
            kwargs['cwd'] = os.path.abspath(kwargs['cwd'])
            logger.info(f"[ComfyStreamClient] Converted workspace path to absolute: {kwargs['cwd']}")
        
        logger.info("[ComfyStreamClient] Config kwargs:", kwargs)
        
        try:
            self.config = Configuration(**kwargs)
            print("[ComfyStreamClient] Configuration created")
            
            if executor_type == "process":
                logger.info("[ComfyStreamClient] Initializing process executor")
                ctx = mp.get_context("spawn")
                logger.info(f"[ComfyStreamClient] Using multiprocessing context: {ctx.get_start_method()}")
                
                manager = ctx.Manager()
                logger.info("[ComfyStreamClient] Created multiprocessing context and manager")
                
                # Create queues with a reasonable size limit to prevent memory issues
                self.image_inputs = manager.Queue(maxsize=30)
                self.image_outputs = manager.Queue(maxsize=30)
                self.audio_inputs = manager.Queue(maxsize=10)
                self.audio_outputs = manager.Queue(maxsize=10)
                logger.info("[ComfyStreamClient] Created manager queues")
                
                logger.info("[ComfyStreamClient] About to create ProcessPoolExecutor...")
                try:
                    # Create executor first
                    executor = ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=init_tensor_cache,
                        initargs=(self.image_inputs, self.image_outputs, self.audio_inputs, self.audio_outputs)
                    )
                    logger.info("[ComfyStreamClient] ProcessPoolExecutor created successfully")
                    
                    # Create EmbeddedComfyClient with the executor
                    logger.info("[ComfyStreamClient] Creating EmbeddedComfyClient with executor")
                    self.comfy_client = EmbeddedComfyClient(self.config, executor=executor)
                    logger.info("[ComfyStreamClient] EmbeddedComfyClient created successfully")
                    
                    # Submit a test task to ensure worker processes are initialized
                    logger.info("[ComfyStreamClient] Testing worker process initialization...")
                    test_future = executor.submit(_test_worker_init)  # Use the named function instead of lambda
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
                
            else:
                logger.info("[ComfyStreamClient] Using default executor")
                logger.info("[ComfyStreamClient] Creating EmbeddedComfyClient in main process")
                self.comfy_client = EmbeddedComfyClient(self.config)
                logger.info("[ComfyStreamClient] EmbeddedComfyClient created in main process")

            self.running_prompts = {}
            self.current_prompts = []
            self.cleanup_lock = asyncio.Lock()
            logger.info("[ComfyStreamClient] __init__ complete")

        except Exception as e:
            logger.info(f"[ComfyStreamClient] Error during initialization: {str(e)}")
            logger.info(f"[ComfyStreamClient] Error type: {type(e)}")
            import traceback
            logger.info(f"[ComfyStreamClient] Error traceback: {traceback.format_exc()}")
            raise

    async def set_prompts(self, prompts: List[PromptDictInput]):
        logger.info("set_prompts start")
        self.current_prompts = [convert_prompt(prompt) for prompt in prompts]
        for idx in range(len(self.current_prompts)):
            logger.info(f"Scheduling run_prompt for idx {idx}")
            task = asyncio.create_task(self.run_prompt(idx))
            self.running_prompts[idx] = task
        logger.info("set_prompts end")

    async def update_prompts(self, prompts: List[PromptDictInput]):
        # TODO: currently under the assumption that only already running prompts are updated
        if len(prompts) != len(self.current_prompts):
            raise ValueError(
                "Number of updated prompts must match the number of currently running prompts."
            )
        self.current_prompts = [convert_prompt(prompt) for prompt in prompts]

    async def run_prompt(self, prompt_index: int):
        logger.info(f"[ComfyStreamClient] Starting run_prompt for index {prompt_index}")
        while True:
            try:
                logger.debug(f"[ComfyStreamClient] Queueing prompt {prompt_index}")
                await self.comfy_client.queue_prompt(self.current_prompts[prompt_index])
                logger.debug(f"[ComfyStreamClient] Prompt {prompt_index} queued successfully")
            except Exception as e:
                logger.error(f"[ComfyStreamClient] Error in run_prompt {prompt_index}: {str(e)}")
                await self.cleanup()
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
            self.image_outputs.get()

    def put_video_input(self, frame, width=None, height=None, pts=None, time_base=None):
        # logger.debug(f"[ComfyStreamClient] Putting video input: {type(frame)}")
        try:
            if isinstance(frame, FrameProxy):
                # Already a FrameProxy, ensure tensor is in BCHW format and on CPU
                tensor = frame.side_data.input
                if len(tensor.shape) == 3:  # CHW format
                    tensor = tensor.unsqueeze(0)  # Add batch dimension -> BCHW
                elif len(tensor.shape) == 4:  # Already BCHW
                    tensor = tensor  # Keep as is
                else:
                    raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
                tensor = tensor.cpu()
                proxy = FrameProxy(
                    tensor=tensor,
                    width=frame.width,
                    height=frame.height,
                    pts=frame.pts,
                    time_base=frame.time_base
                )
            elif hasattr(frame, "to_ndarray") and hasattr(frame, "width") and hasattr(frame, "height"):
                # It's an av.VideoFrame, convert to BCHW format
                frame_np = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0)  # Convert to BCHW
                proxy = FrameProxy(
                    tensor=tensor,
                    width=frame.width,
                    height=frame.height,
                    pts=getattr(frame, 'pts', None),
                    time_base=getattr(frame, 'time_base', None)
                )
            else:
                # Assume it's a tensor, require width/height
                if width is None or height is None:
                    raise ValueError("Width and height must be provided for raw tensors")
                tensor = frame
                if len(tensor.shape) == 3:  # CHW format
                    tensor = tensor.unsqueeze(0)  # Add batch dimension -> BCHW
                elif len(tensor.shape) == 4:  # Already BCHW
                    tensor = tensor  # Keep as is
                else:
                    raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
                tensor = tensor.cpu()
                proxy = FrameProxy(
                    tensor=tensor,
                    width=width,
                    height=height,
                    pts=pts,
                    time_base=time_base
                )

            if self.image_inputs.full():
                try:
                    self.image_inputs.get_nowait()
                except Exception:
                    pass
            self.image_inputs.put_nowait(proxy)
            logger.debug(f"[ComfyStreamClient] Video input queued.")
        except Exception as e:
            logger.info(f"[ComfyStreamClient] Error putting video frame: {str(e)}")

    def put_audio_input(self, frame):
        self.audio_inputs.put(frame)

    async def get_video_output(self):
        try:
            tensor = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.image_outputs.get),
                timeout=5.0
            )
            # Add format conversion here
            if len(tensor.shape) == 4 and tensor.shape[1] != 3:  # If BHWC format
                tensor = tensor.permute(0, 3, 1, 2)  # Convert BHWC to BCHW
            return tensor
        except asyncio.TimeoutError:
            return torch.zeros((1, 3, self.height, self.width), dtype=torch.float32)  # Return BCHW format
        except Exception as e:
            logger.info(f"[ComfyStreamClient] Error getting video output: {str(e)}")
            return torch.zeros((1, 3, self.height, self.width), dtype=torch.float32)  # Return BCHW format
    
    async def get_audio_output(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.audio_outputs.get)

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

def execute_prompt_in_worker(config_dict, prompt):
    """Execute a prompt in the worker process"""
    logger.info(f"[execute_prompt_in_worker] Starting in process {os.getpid()}")
    try:
        import os
        import sys
        import torch
        from comfy.cli_args_types import Configuration
        from comfy.client.embedded_comfy_client import EmbeddedComfyClient
        
        # On Windows, we need to ensure the working directory is correct
        if sys.platform == 'win32':
            # Get the workspace directory from config
            workspace = config_dict.get('cwd', '..\\..')
            logger.info(f"[execute_prompt_in_worker] Setting working directory to: {workspace}")
            os.chdir(workspace)
            
            # Ensure Python path includes the workspace
            if workspace not in sys.path:
                sys.path.insert(0, workspace)
                logger.info(f"[execute_prompt_in_worker] Added {workspace} to Python path")
        
        logger.info(f"[execute_prompt_in_worker] Current working directory: {os.getcwd()}")
        logger.info(f"[execute_prompt_in_worker] Python path: {sys.path}")
        
        # Create a new client in the worker process
        logger.info("[execute_prompt_in_worker] Creating configuration")
        config = Configuration(**config_dict)
        
        logger.info("[execute_prompt_in_worker] Creating EmbeddedComfyClient")
        # Try to initialize CUDA before creating the client
        if torch.cuda.is_available():
            logger.info(f"[execute_prompt_in_worker] CUDA device count: {torch.cuda.device_count()}")
            # Set the device explicitly
            torch.cuda.set_device(0)
            logger.info(f"[execute_prompt_in_worker] Set CUDA device to: {torch.cuda.current_device()}")
        
        client = EmbeddedComfyClient(config)
        
        # Execute the prompt
        logger.info("[execute_prompt_in_worker] Setting up event loop")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("[execute_prompt_in_worker] Queueing prompt")
            loop.run_until_complete(client.queue_prompt(prompt))
            logger.info("[execute_prompt_in_worker] Prompt queued successfully")
        finally:
            logger.info("[execute_prompt_in_worker] Closing event loop")
            loop.close()
    except Exception as e:
        logger.info(f"[execute_prompt_in_worker] Error: {str(e)}")
        logger.info(f"[execute_prompt_in_worker] Error type: {type(e)}")
        import traceback
        logger.info(f"[execute_prompt_in_worker] Error traceback: {traceback.format_exc()}")
        raise