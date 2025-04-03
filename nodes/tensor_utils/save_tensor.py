import torch

class SaveTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("IMAGES",)
    RETURN_NAMES = ("images",)
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
        return (images,)
