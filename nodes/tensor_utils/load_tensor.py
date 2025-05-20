import os
from comfystream import tensor_cache

class LoadTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {}

    @classmethod
    def IS_CHANGED():
        return float("nan")

    def execute(self):
        print("LoadTensor PID:", os.getpid())
        image = tensor_cache.image_inputs.get()
        return (image,)