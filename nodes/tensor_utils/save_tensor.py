import torch
from comfy.nodes.package_typing import CustomNode
from comfystream.tensor_cache import image_outputs


class SaveTensor(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "tensor_utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls):
        return float("nan")

    def execute(self, images: torch.Tensor):
        image_outputs.put_nowait(images)
        return images


NODE_CLASS_MAPPINGS = {
    "SaveTensor": SaveTensor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveTensor": "Save Tensor"
}
