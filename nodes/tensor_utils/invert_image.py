import torch


class InvertImage:
    """Invert IMAGE tensors in [0, 1] (1.0 - image)."""

    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    def execute(self, images: torch.Tensor):
        return (1.0 - images,)
