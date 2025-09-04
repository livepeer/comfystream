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
                "text": ("STRING",),
            },
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, text: str):
        try:
            # Put text into tensor_cache text_outputs queue
            tensor_cache.text_outputs.put_nowait(text)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to queue text output: {e}")
        return ()


