from comfystream import tensor_cache
import logging

class SaveTextTensor:
    CATEGORY = "text_utils"
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, text: str):
        tensor_cache.text_outputs.put_nowait(text)
        logging.info(f"Saved text tensor: {text}")
        return (text,) 