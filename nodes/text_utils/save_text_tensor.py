import logging
from comfystream import tensor_cache

logger = logging.getLogger(__name__)

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
            },
            "optional": {
                "debug_info": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log detailed debugging information about text queuing"
                })
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, text: str, debug_info: bool = False):
        """Execute SaveTextTensor with improved debugging."""
        try:
            # Check for warmup sentinel value or actual content
            is_sentinel = "__WARMUP_SENTINEL__" in text if text else False
            
            # Only publish non-empty text or sentinel values to prevent spam
            if (text and text.strip()) or is_sentinel:
                if debug_info:
                    if is_sentinel:
                        logger.debug(f"SaveTextTensor queuing warmup sentinel")
                    else:
                        logger.debug(f"SaveTextTensor queuing text (length: {len(text)} chars): '{text[:50]}...'")
                
                # Store text in the async text cache for retrieval by the pipeline
                tensor_cache.text_outputs.put_nowait(text)
            else:
                # Skip empty or whitespace-only text
                if debug_info:
                    logger.debug(f"SaveTextTensor skipping empty text")
            
            return ()
            
        except Exception as e:
            logger.error(f"Error in SaveTextTensor: {e}")
            return ()