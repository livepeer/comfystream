class SaveFalResult:
    """Output node for a FalSubmit/FalCollect pair.

    ComfyUI only executes a prompt that contains an OUTPUT_NODE, and
    queue_prompt only returns UI data from those nodes. FalCollect is not an
    output node, so this sink publishes request_id and result_json.
    """

    CATEGORY = "tensor_utils"
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "request_id": ("STRING",),
                "result_json": ("STRING",),
            }
        }

    def execute(self, request_id, result_json):
        return {
            "ui": {
                "request_id": [request_id],
                "result_json": [result_json],
            }
        }
