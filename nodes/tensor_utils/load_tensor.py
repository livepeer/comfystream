import torch
from comfystream import tensor_cache


class LoadTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            }
        }

    @classmethod
    def IS_CHANGED():
        return float("nan")

    def execute(self, batch_size: int = 1):
        """
        Load tensor(s) from the tensor cache.
        If batch_size > 1, loads multiple tensors and stacks them into a batch.
        """
        if batch_size == 1:
            # Single tensor loading (original behavior)
            frame = tensor_cache.image_inputs.get(block=True)
            frame.side_data.skipped = False
            return (frame.side_data.input,)
        else:
            # Batch tensor loading
            batch_images = []
            
            # Collect images up to batch_size
            for i in range(batch_size):
                if not tensor_cache.image_inputs.empty():
                    frame = tensor_cache.image_inputs.get(block=False)
                    frame.side_data.skipped = False
                    batch_images.append(frame.side_data.input)
                else:
                    # If we don't have enough images, pad with the last available image
                    if batch_images:
                        batch_images.append(batch_images[-1])
                    else:
                        # If no images available, create a dummy tensor
                        dummy_tensor = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
                        batch_images.append(dummy_tensor)
            
            # Stack images into a batch
            if len(batch_images) > 1:
                batch_tensor = torch.cat(batch_images, dim=0)
            else:
                batch_tensor = batch_images[0]
                
            return (batch_tensor,)
