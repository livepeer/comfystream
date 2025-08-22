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
            "optional": {
                "debug_info": ("STRING", {"default": ""}),
            },
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, text: str, debug_info: str = ""):
        try:
            # Put text into a dedicated queue; if not present, fallback to audio_outputs for compatibility
            if hasattr(tensor_cache, "text_outputs"):
                tensor_cache.text_outputs.put_nowait(text)
            else:
                # Backward-compatible path to avoid breaking runtime if text queue is not configured
                tensor_cache.audio_outputs.put_nowait(text)  # type: ignore[arg-type]
            if debug_info:
                logging.getLogger(__name__).debug(f"SaveTextTensor queued text len={len(text)} info={debug_info}")
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to queue text output: {e}")
        return (text,)


