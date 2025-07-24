from .load_audio_tensor import LoadAudioTensor, NODE_DISPLAY_NAME_MAPPINGS as load_audio_display_mappings
from .save_audio_tensor import SaveAudioTensor, NODE_DISPLAY_NAME_MAPPINGS as save_audio_display_mappings
from .pitch_shift import PitchShifter, NODE_DISPLAY_NAME_MAPPINGS as pitch_shift_display_mappings

NODE_CLASS_MAPPINGS = {"LoadAudioTensor": LoadAudioTensor, "SaveAudioTensor": SaveAudioTensor, "PitchShifter": PitchShifter}

# Combine display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(load_audio_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(save_audio_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(pitch_shift_display_mappings)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
