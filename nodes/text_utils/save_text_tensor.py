from comfystream import tensor_cache

class SaveTextTensor:
    CATEGORY = "text_utils"
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, text: str):
        # Store text in the async text cache for retrieval by the pipeline
        tensor_cache.text_outputs.put_nowait(text)
        return ()