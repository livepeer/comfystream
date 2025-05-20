# TODO: add better frame management, improve eviction policy fifo might not be the best, skip alternate frames instead
# TODO: also make the tensor_cache solution backward compatible for when not using process pool -- after the multi process solution is stable
from comfystream import tensor_cache
import queue
import torch
import asyncio
from queue import Queue
from asyncio import Queue as AsyncQueue

image_inputs = None
image_outputs = None

audio_inputs = None
audio_outputs = None

# Create wrapper classes that match the interface of the original queues
class MultiProcessInputQueue:
    def __init__(self, mp_queue):
        self.queue = mp_queue
    
    def get(self, block=True, timeout=None):
        return self.queue.get(block=block, timeout=timeout)
    
    def get_nowait(self):
        return self.queue.get_nowait()
    
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
        return await loop.run_in_executor(None, self.queue.get)
    
    async def put(self, item):
        # Convert synchronous put to async
        loop = asyncio.get_event_loop()
        # Ensure tensor is on CPU before sending
        if torch.is_tensor(item):
            item = item.cpu()
        return await loop.run_in_executor(None, self.queue.put, item)
    
    def put_nowait(self, item):
        try:
            # Ensure tensor is on CPU before sending
            if torch.is_tensor(item):
                item = item.cpu()
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()  # Drop oldest
                self.queue.put_nowait(item)
            except Exception:
                pass  # If still full, drop this frame

def init_tensor_cache(image_inputs, image_outputs, audio_inputs, audio_outputs):
    """Initialize the tensor cache for a worker process.
    
    Args:
        image_inputs: Multiprocessing Queue for input images
        image_outputs: Multiprocessing Queue for output images
        audio_inputs: Multiprocessing Queue for input audio
        audio_outputs: Multiprocessing Queue for output audio
    """
    print("[init_tensor_cache] Setting up tensor_cache queues in worker")
    
    # Replace the queues with our wrapped versions that match the original interface
    tensor_cache.image_inputs = MultiProcessInputQueue(image_inputs)
    tensor_cache.image_outputs = MultiProcessOutputQueue(image_outputs)
    tensor_cache.audio_inputs = MultiProcessInputQueue(audio_inputs)
    tensor_cache.audio_outputs = MultiProcessOutputQueue(audio_outputs)
    
    print("[init_tensor_cache] tensor_cache.image_outputs id:", id(tensor_cache.image_outputs))
    print("[init_tensor_cache] Initialization complete")