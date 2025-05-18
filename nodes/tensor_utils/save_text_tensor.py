import json

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
                "data": ("DICT",),  # Accept any dictionary as input.
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, data):
        result_json = json.dumps(data)
        tensor_cache.text_outputs.put_nowait(result_json)
        return (result_json,)
