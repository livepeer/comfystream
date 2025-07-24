from comfy.comfy_types.node_typing import ComfyNodeABC as CustomNode
from comfystream.tensor_cache import audio_outputs

class SaveAudioTensor(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("WAVEFORM",)
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "audio_utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls):
        return float("nan")

    def execute(self, audio):
        audio_outputs.put_nowait(audio)
        return ()


NODE_CLASS_MAPPINGS = {
    "SaveAudioTensor": SaveAudioTensor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAudioTensor": "Save Audio Tensor"
}

