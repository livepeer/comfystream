from .load_audio_tensor import LoadAudioTensor
from .load_audio_tensor_stream import LoadAudioTensorStream
from .save_audio_tensor import SaveAudioTensor
from .pitch_shift import PitchShifter

NODE_CLASS_MAPPINGS = {
    "LoadAudioTensor": LoadAudioTensor,
    "SaveAudioTensor": SaveAudioTensor,
    "PitchShifter": PitchShifter,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
