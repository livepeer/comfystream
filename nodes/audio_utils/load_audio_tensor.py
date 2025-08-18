import numpy as np
import logging

from comfystream import tensor_cache

logger = logging.getLogger(__name__)

class LoadAudioTensor:
    CATEGORY = "audio_utils"
    RETURN_TYPES = ("WAVEFORM", "INT")
    RETURN_NAMES = ("waveform", "sample_rate")
    FUNCTION = "execute"
    
    def __init__(self):
        self.audio_buffer = np.empty(0, dtype=np.int16)
        self.buffer_samples = None
        self.sample_rate = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "buffer_size": ("FLOAT", {"default": 500.0}),
            }
        }
    
    @classmethod
    def IS_CHANGED():
        return float("nan")
    
    def execute(self, buffer_size):
        if self.sample_rate is None or self.buffer_samples is None:
            frame = tensor_cache.audio_inputs.get(block=True)
            # Use fixed 16kHz sample rate for ComfyUI audio processing
            self.sample_rate = 16000  # Fixed target sample rate
            self.buffer_samples = int(self.sample_rate * buffer_size / 1000)
            
            # Process the initial frame - expect pre-processed audio
            if hasattr(frame, 'side_data') and hasattr(frame.side_data, 'input'):
                # Already processed by trickle processor
                self.leftover = frame.side_data.input
            else:
                # Fallback for direct audio data
                self.leftover = getattr(frame, 'samples', frame)
                if not isinstance(self.leftover, np.ndarray):
                    self.leftover = np.array(self.leftover, dtype=np.int16)
        
        if self.leftover.shape[0] < self.buffer_samples:
            chunks = [self.leftover] if self.leftover.size > 0 else []
            total_samples = self.leftover.shape[0]
            
            while total_samples < self.buffer_samples:
                frame = tensor_cache.audio_inputs.get(block=True)
                
                # Process each frame - expect pre-processed audio
                if hasattr(frame, 'side_data') and hasattr(frame.side_data, 'input'):
                    # Already processed by trickle processor
                    processed_audio = frame.side_data.input
                    frame_rate = getattr(frame, 'sample_rate', self.sample_rate)
                else:
                    # Fallback for direct audio data
                    processed_audio = getattr(frame, 'samples', frame)
                    if not isinstance(processed_audio, np.ndarray):
                        processed_audio = np.array(processed_audio, dtype=np.int16)
                    frame_rate = self.sample_rate
                
                # Warn about sample rate mismatches
                if frame_rate != self.sample_rate:
                    logger.warning(f"Expected {self.sample_rate}Hz audio but got {frame_rate}Hz - audio may need conversion")
                
                chunks.append(processed_audio)
                total_samples += processed_audio.shape[0]
            
            merged_audio = np.concatenate(chunks, dtype=np.int16)
            buffered_audio = merged_audio[:self.buffer_samples]
            self.leftover = merged_audio[self.buffer_samples:]
        else:
            buffered_audio = self.leftover[:self.buffer_samples]
            self.leftover = self.leftover[self.buffer_samples:]
                
        return buffered_audio, self.sample_rate
