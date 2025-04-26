class SaveTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bytes",)
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bytes": ("STRING",),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, bytes: str):
        return {"ui": {"results": [bytes]}}
