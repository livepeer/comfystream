import torch
import numpy as np
from typing import List, Union

from comfystream import tensor_cache


class LoadBatchTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, batch_size: int):
        """
        Load a batch of images from the tensor cache.
        Collects up to batch_size images from the queue.
        """
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


class SaveBatchTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, images: torch.Tensor):
        """
        Save a batch of images to the tensor cache.
        Splits the batch and puts each image individually.
        """
        # Split batch into individual images
        if images.dim() == 4 and images.shape[0] > 1:
            # Batch of images
            for i in range(images.shape[0]):
                single_image = images[i:i+1]  # Keep batch dimension
                tensor_cache.image_outputs.put_nowait(single_image)
        else:
            # Single image
            tensor_cache.image_outputs.put_nowait(images)
        
        return images
