from comfystream import tensor_cache
import logging
import queue
import torch
import asyncio
import os
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
            # Ensure tensor is on CPU
            if torch.is_tensor(item):
                item = item.cpu()
            self.queue.put_nowait(item)
        except queue.Full:
            # Simple: drop one old frame and try again
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(item)
            except:
                pass # If still fails, just drop this frame

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

    # Replace the queues with our wrapped versions that match the original interface
    tensor_cache.image_inputs = MultiProcessInputQueue(image_inputs)
    tensor_cache.image_outputs = MultiProcessOutputQueue(image_outputs)
    tensor_cache.audio_inputs = MultiProcessInputQueue(audio_inputs)
    tensor_cache.audio_outputs = MultiProcessOutputQueue(audio_outputs)
    
    logger.info(f"[init_tensor_cache] tensor_cache.image_outputs id: {id(tensor_cache.image_outputs)} - PID: {os.getpid()}")
    logger.info(f"[init_tensor_cache] Initialization complete - PID: {os.getpid()}")
    
    return os.getpid()  # Return PID for verification