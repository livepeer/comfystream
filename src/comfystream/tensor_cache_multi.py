# TODO: add better frame management, improve eviction policy fifo might not be the best, skip alternate frames instead
# TODO: also make the tensor_cache solution backward compatible for when not using process pool -- after the multi process solution is stable
from comfystream import tensor_cache
import logging
import queue
import torch
import asyncio
import os
from comfy.cmd import folder_paths
logger = logging.getLogger(__name__)

image_inputs = None
image_outputs = None

audio_inputs = None
audio_outputs = None

# Global frame ID tracking for worker processes
current_frame_id = None
frame_id_mapping = {}  # Maps tensor id to frame_id

class FrameData:
    """Wrapper class to carry frame metadata through the processing pipeline"""
    def __init__(self, tensor, frame_id=None):
        self.tensor = tensor
        self.frame_id = frame_id

# Create wrapper classes that match the interface of the original queues
class MultiProcessInputQueue:
    def __init__(self, mp_queue):
        self.queue = mp_queue
    
    def get(self, block=True, timeout=None):
        result = self.queue.get(block=block, timeout=timeout)
        
        # Extract frame metadata and store it globally for this worker
        global current_frame_id
        if hasattr(result, 'side_data') and hasattr(result.side_data, 'frame_id'):
            current_frame_id = result.side_data.frame_id
            # logger.info(f"[MultiProcessInputQueue] Frame {current_frame_id} retrieved by worker PID: {os.getpid()}")
        
        return result
    
    def get_nowait(self):
        result = self.queue.get_nowait()
        
        # Extract frame metadata and store it globally for this worker
        global current_frame_id
        if hasattr(result, 'side_data') and hasattr(result.side_data, 'frame_id'):
            current_frame_id = result.side_data.frame_id
            # logger.info(f"[MultiProcessInputQueue] Frame {current_frame_id} retrieved (nowait) by worker PID: {os.getpid()}")
        
        return result
    
    def put(self, item, block=True, timeout=None):
        return self.queue.put(item, block=block, timeout=timeout)
    
    def put_nowait(self, item):
        return self.queue.put_nowait(item)
    
    def empty(self):
        return self.queue.empty()
    
    def full(self):
        return self.queue.full()

class MultiProcessOutputQueue:
    def __init__(self, mp_queue):
        self.queue = mp_queue
    
    async def get(self):
        # Convert synchronous get to async
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.queue.get)
        
        # Check if this is a tuple with frame_id
        if isinstance(result, tuple) and len(result) == 2:
            frame_id, tensor = result
            return (frame_id, tensor)
        else:
            # Backward compatibility - return just the tensor
            return result
    
    async def put(self, item):
        # Convert synchronous put to async
        loop = asyncio.get_event_loop()
        # Ensure tensor is on CPU before sending
        if torch.is_tensor(item):
            item = item.cpu()
        # logger.info(f"[MultiProcessOutputQueue] Frame sent from worker PID: {os.getpid()}")
        return await loop.run_in_executor(None, self.queue.put, item)
    
    def put_nowait(self, item):
        try:
            # Check if we have a current frame ID to associate with this output
            global current_frame_id
            
            # Ensure tensor is on CPU before sending
            if torch.is_tensor(item):
                item = item.cpu()
            
            # If we have a frame ID, send it as a tuple
            if current_frame_id is not None:
                output_data = (current_frame_id, item)
                # logger.info(f"[MultiProcessOutputQueue] Frame {current_frame_id} sent (nowait) from worker PID: {os.getpid()}")
            else:
                output_data = item
                # logger.info(f"[MultiProcessOutputQueue] Frame sent (nowait) without ID from worker PID: {os.getpid()}")
                
            self.queue.put_nowait(output_data)
        except queue.Full:
            try:
                self.queue.get_nowait()  # Drop oldest
                # Try again with the same logic
                if current_frame_id is not None:
                    output_data = (current_frame_id, item)
                else:
                    output_data = item
                self.queue.put_nowait(output_data)
            except Exception:
                pass  # If still full, drop this frame

def init_tensor_cache(image_inputs, image_outputs, audio_inputs, audio_outputs, workspace_path=None):
    """Initialize the tensor cache for a worker process.
    
    Args:
        image_inputs: Multiprocessing Queue for input images
        image_outputs: Multiprocessing Queue for output images
        audio_inputs: Multiprocessing Queue for input audio
        audio_outputs: Multiprocessing Queue for output audio
        workspace_path: The ComfyUI workspace path (should be C:\sd\ComfyUI-main)
    """
    logger.info(f"[init_tensor_cache] Setting up tensor_cache queues in worker - PID: {os.getpid()}")
    logger.info(f"[init_tensor_cache] Workspace path: {workspace_path}")
    logger.info(f"[init_tensor_cache] Current working directory: {os.getcwd()}")

    # Initialize folder_paths in worker process
    # Another attempt to fix the tensorrt paths issue via ProcessPoolExecutor
    '''
    try:
        # Import both possible folder_paths modules
        from comfy.cmd import folder_paths as comfy_folder_paths
        
        # Also try to import the direct folder_paths (which TensorRT loader uses)
        import sys
        try:
            import folder_paths as direct_folder_paths
            logger.info("[init_tensor_cache] Successfully imported direct folder_paths")
        except ImportError:
            # If direct import fails, create an alias
            sys.modules['folder_paths'] = comfy_folder_paths
            direct_folder_paths = comfy_folder_paths
            logger.info("[init_tensor_cache] Created folder_paths alias to comfy.cmd.folder_paths")
        
        logger.info(f"[init_tensor_cache] comfy_folder_paths.models_dir: {comfy_folder_paths.models_dir}")
        logger.info(f"[init_tensor_cache] direct_folder_paths.models_dir: {direct_folder_paths.models_dir}")
        
        # Use the workspace_path as the base directory for TensorRT paths
        if workspace_path:
            base_dir = workspace_path
        else:
            # Fallback to the parent directory of models_dir
            base_dir = os.path.dirname(comfy_folder_paths.models_dir)
            
        # Set up both models/tensorrt and outputs/tensorrt directories
        tensorrt_models_dir = os.path.join(base_dir, "models", "tensorrt")
        tensorrt_outputs_dir = os.path.join(base_dir, "outputs", "tensorrt")
        
        logger.info(f"[init_tensor_cache] TensorRT models folder: {tensorrt_models_dir}")
        logger.info(f"[init_tensor_cache] TensorRT outputs folder: {tensorrt_outputs_dir}")
        logger.info(f"[init_tensor_cache] Models dir exists: {os.path.exists(tensorrt_models_dir)}")
        logger.info(f"[init_tensor_cache] Outputs dir exists: {os.path.exists(tensorrt_outputs_dir)}")
        
        # Register TensorRT paths in BOTH folder_paths modules
        tensorrt_config = ([tensorrt_models_dir, tensorrt_outputs_dir], {".engine"})
        
        # Update comfy.cmd.folder_paths
        comfy_folder_paths.folder_names_and_paths["tensorrt"] = tensorrt_config
        logger.info("[init_tensor_cache] Registered TensorRT paths in comfy.cmd.folder_paths")
        
        # Update direct folder_paths (which TensorRT loader uses)
        direct_folder_paths.folder_names_and_paths["tensorrt"] = tensorrt_config
        logger.info("[init_tensor_cache] Registered TensorRT paths in direct folder_paths")
        
        # Also update any existing modules in sys.modules
        for module_name, module in sys.modules.items():
            if (module_name.endswith('folder_paths') or module_name == 'folder_paths') and hasattr(module, 'folder_names_and_paths'):
                module.folder_names_and_paths["tensorrt"] = tensorrt_config
                logger.info(f"[init_tensor_cache] Updated TensorRT paths in {module_name}")
        
        # Verify the registration worked
        logger.info(f"[init_tensor_cache] comfy_folder_paths TensorRT files: {comfy_folder_paths.get_filename_list('tensorrt')}")
        logger.info(f"[init_tensor_cache] direct_folder_paths TensorRT files: {direct_folder_paths.get_filename_list('tensorrt')}")
        
    except Exception as e:
        logger.error(f"[init_tensor_cache] Error initializing folder_paths: {e}")
        import traceback
        logger.error(f"[init_tensor_cache] Traceback: {traceback.format_exc()}")
    '''

    # Replace the queues with our wrapped versions that match the original interface
    tensor_cache.image_inputs = MultiProcessInputQueue(image_inputs)
    tensor_cache.image_outputs = MultiProcessOutputQueue(image_outputs)
    tensor_cache.audio_inputs = MultiProcessInputQueue(audio_inputs)
    tensor_cache.audio_outputs = MultiProcessOutputQueue(audio_outputs)
    
    logger.info(f"[init_tensor_cache] tensor_cache.image_outputs id: {id(tensor_cache.image_outputs)} - PID: {os.getpid()}")
    logger.info(f"[init_tensor_cache] Initialization complete - PID: {os.getpid()}")
    
    return os.getpid()  # Return PID for verification

# THis was an attempt to fix the tensorrt paths issue via ProcessPoolExecutor
'''
def register_tensorrt_paths(workspace_path=None):
    """Register TensorRT paths in folder_paths at import time"""
    try:
        # Use workspace_path if provided, otherwise fall back to folder_paths.models_dir
        if workspace_path:
            base_dir = workspace_path
            tensorrt_models_dir = os.path.join(base_dir, "models", "tensorrt")
        else:
            # Create tensorrt subdirectory in the models directory
            tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt")
        
        print(f"[TensorRT] workspace_path: {workspace_path}")
        print(f"[TensorRT] folder_paths.models_dir: {folder_paths.models_dir}")
        print(f"[TensorRT] Registering paths:")
        print(f"[TensorRT] - Models: {tensorrt_models_dir}")
        
        if "tensorrt" in folder_paths.folder_names_and_paths:
            # Update existing registration
            existing_paths = folder_paths.folder_names_and_paths["tensorrt"][0]
            if tensorrt_models_dir not in existing_paths:
                existing_paths.append(tensorrt_models_dir)
            folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
        else:
            # Create new registration (same as Depth-Anything approach)
            folder_paths.folder_names_and_paths["tensorrt"] = (
                [tensorrt_models_dir], 
                {".engine"}
            )
        
        # Verify registration
        available_files = folder_paths.get_filename_list("tensorrt")
        print(f"[TensorRT] Available engine files: {available_files}")
        
    except Exception as e:
        print(f"[TensorRT] Error registering paths: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to original behavior
        if "tensorrt" in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths["tensorrt"][0].append(
                os.path.join(folder_paths.models_dir, "tensorrt"))
            folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
        else:
            folder_paths.folder_names_and_paths["tensorrt"] = (
                [os.path.join(folder_paths.models_dir, "tensorrt")], 
                {".engine"}
            )
'''