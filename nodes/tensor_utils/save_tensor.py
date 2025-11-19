import torch

from comfystream import tensor_cache


class SaveTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "split_batch": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, images: torch.Tensor, split_batch: bool = False):
        """
        Save tensor(s) to the tensor cache.
        If split_batch is True and images is a batch, splits it into individual images.
        """
        if split_batch and images.dim() == 4 and images.shape[0] > 1:
            # Split batch into individual images
            for i in range(images.shape[0]):
                single_image = images[i:i+1]  # Keep batch dimension
                tensor_cache.image_outputs.put_nowait(single_image)
        else:
            # Save as single tensor (original behavior)
            tensor_cache.image_outputs.put_nowait(images)
        
        return images
