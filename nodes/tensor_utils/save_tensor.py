import numpy as np
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
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    @staticmethod
    def _split_images(images):
        """Yield individual images for batched tensors/lists without changing interface."""
        # Torch tensor inputs with optional batch dimension in dim 0
        if isinstance(images, torch.Tensor):
            if images.dim() >= 4 and images.shape[0] > 1:
                for img in images:
                    yield img
                return
            yield images
            return

        # Numpy arrays (should rarely occur, but handled for completeness)
        if isinstance(images, np.ndarray):
            if images.ndim >= 4 and images.shape[0] > 1:
                for img in images:
                    yield img
                return
            yield images
            return

        # Lists/tuples of images already separated
        if isinstance(images, (list, tuple)):
            for img in images:
                yield img
            return

        # Fallback to passing through any other type as-is
        yield images

    def execute(self, images: torch.Tensor):
        for img in self._split_images(images):
            # Schedule the put operation on the main event loop thread safely
            if tensor_cache.main_loop:
                tensor_cache.main_loop.call_soon_threadsafe(
                    tensor_cache.image_outputs.put_nowait, img
                )
            else:
                # Fallback implementation (mostly for tests without init or direct execution)
                try:
                    tensor_cache.image_outputs.put_nowait(img)
                except RuntimeError:
                    # If we are in a thread with no loop, this might fail or be unsafe, 
                    # but if main_loop is not set we have few options. 
                    pass
        return images
