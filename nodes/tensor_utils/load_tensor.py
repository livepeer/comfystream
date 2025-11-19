import queue

from comfystream import tensor_cache
from comfystream.exceptions import ComfyStreamInputTimeoutError


class LoadTensor:
    CATEGORY = "ComfyStream/Loaders"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    DESCRIPTION = "Load image tensor from ComfyStream input with timeout."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "timeout_seconds": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 30.0,
                        "step": 0.1,
                        "tooltip": "Timeout in seconds",
                    },
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, batch_size: int = 1, timeout_seconds: float = 1.0):
        """
        Load tensor(s) from the tensor cache.
        If batch_size > 1, loads multiple tensors and stacks them into a batch.
        """
        if batch_size == 1:
            # Single tensor loading with timeout
            try:
                frame = tensor_cache.image_inputs.get(block=True, timeout=timeout_seconds)
                frame.side_data.skipped = False
                return (frame.side_data.input,)
            except queue.Empty:
                raise ComfyStreamInputTimeoutError("video", timeout_seconds)
        else:
            # Batch tensor loading - only process if we have enough real frames
            batch_images = []
            
            # Collect images up to batch_size, but only use real frames
            for i in range(batch_size):
                if not tensor_cache.image_inputs.empty():
                    try:
                        frame = tensor_cache.image_inputs.get(block=True, timeout=timeout_seconds)
                        frame.side_data.skipped = False
                        batch_images.append(frame.side_data.input)
                    except queue.Empty:
                        # If timeout occurs, stop collecting and use what we have
                        break
                else:
                    # If queue is empty, stop collecting
                    break
            
            # Only proceed if we have at least one real frame
            if not batch_images:
                # No frames available - raise timeout error instead of creating dummy
                raise ComfyStreamInputTimeoutError("video", timeout_seconds)
            
            # If we have fewer frames than requested, pad with the last available frame
            # This is better than dummy tensors as it maintains visual continuity
            while len(batch_images) < batch_size:
                batch_images.append(batch_images[-1])
            
            # Stack images into a batch
            if len(batch_images) > 1:
                batch_tensor = torch.cat(batch_images, dim=0)
            else:
                batch_tensor = batch_images[0]
                
            return (batch_tensor,)
