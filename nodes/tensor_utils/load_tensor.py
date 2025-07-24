from comfy.comfy_types.node_typing import ComfyNodeABC as CustomNode
from comfystream.tensor_cache import image_inputs


class LoadTensor(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "tensor_utils"

    @classmethod
    def IS_CHANGED(cls):
        return float("nan")

    def execute(self):
        frame = image_inputs.get(block=True)
        frame.side_data.skipped = False
        return (frame.side_data.input,)


NODE_CLASS_MAPPINGS = {
    "LoadTensor": LoadTensor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTensor": "Load Tensor"
}
