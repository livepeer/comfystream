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
                "strs": ("STRING",),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, strs: str):
        tensor_cache.image_outputs.put(strs)
        return strs