from .load_audio_tensor import LoadAudioTensor
from .load_audio_tensor_stream import LoadAudioTensorStream
from .save_audio_tensor import SaveAudioTensor
from .pitch_shift import PitchShifter

# Import from ComfyUI-Stream-Pack
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ComfyUI', 'custom_nodes', 'ComfyUI-Stream-Pack', 'src'))


NODE_CLASS_MAPPINGS = {
    "LoadAudioTensor": LoadAudioTensor, 
    "LoadAudioTensorStream": LoadAudioTensorStream,
    "SaveAudioTensor": SaveAudioTensor, 
    "PitchShifter": PitchShifter,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
