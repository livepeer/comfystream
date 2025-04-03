import os

class LoadTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    @classmethod
    def IS_CHANGED():
        return float("nan")

    def execute(self, image):
        print("PID", os.getpid())
        return (image,)
