class LoadTensor:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bytes",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bytes": ("STRING", {"multiline": True}),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, bytes: str):
        return (bytes,)
