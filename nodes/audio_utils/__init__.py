from .load_audio_tensor import LoadAudioTensor
from .load_audio_tensor_stream import LoadAudioTensorStream
from .save_audio_tensor import SaveAudioTensor
from .pitch_shift import PitchShifter
from .audio_transcription_node import AudioTranscriptionNode

NODE_CLASS_MAPPINGS = {
    "LoadAudioTensor": LoadAudioTensor, 
    "LoadAudioTensorStream": LoadAudioTensorStream,
    "SaveAudioTensor": SaveAudioTensor, 
    "PitchShifter": PitchShifter, 
    "AudioTranscriptionNode": AudioTranscriptionNode
}

__all__ = ["NODE_CLASS_MAPPINGS"]
