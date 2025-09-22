import numpy as np
import torch
import queue
import logging

from comfystream import tensor_cache

logger = logging.getLogger(__name__)

class LoadAudioTensor:
    CATEGORY = "audio_utils"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    
    def __init__(self):
        self.audio_buffer = np.empty(0, dtype=np.int16)
        self.buffer_samples = None
        self.sample_rate = None
        self.last_sample_rate = 44100  # Default fallback sample rate
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "buffer_size": ("FLOAT", {"default": 500.0}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
    
    def _get_audio_frame_with_timeout(self, timeout_seconds=1.0):
        """Get audio frame with timeout, return None if no frame available."""
        try:
            return tensor_cache.audio_inputs.get(block=True, timeout=timeout_seconds)
        except queue.Empty:
            return None
    
    def _create_silent_audio(self, buffer_samples, sample_rate):
        """Create silent audio buffer as fallback."""
        logger.info(f"No audio input available, generating silent audio buffer ({buffer_samples} samples at {sample_rate}Hz)")
        return np.zeros(buffer_samples, dtype=np.int16)
    
    def execute(self, buffer_size):
        # Initialize if needed
        if self.sample_rate is None or self.buffer_samples is None:
            frame = self._get_audio_frame_with_timeout(1.0)
            
            if frame is None:
                # No audio input available - use last known sample rate or default
                self.sample_rate = self.last_sample_rate
                logger.warning(f"No audio frames available in tensor cache, using sample rate: {self.sample_rate} Hz")
                self.buffer_samples = int(self.sample_rate * buffer_size / 1000)
                self.leftover = np.empty(0, dtype=np.int16)
                
                # Return silent audio immediately
                buffered_audio = self._create_silent_audio(self.buffer_samples, self.sample_rate)
            else:
                # Normal ComfyStream mode - remember this sample rate for future fallbacks
                self.sample_rate = frame.sample_rate
                self.last_sample_rate = self.sample_rate
                self.buffer_samples = int(self.sample_rate * buffer_size / 1000)
                self.leftover = frame.side_data.input
        
        # Handle case where we need to generate silent audio immediately
        if not hasattr(self, 'leftover'):
            buffered_audio = self._create_silent_audio(self.buffer_samples, self.sample_rate)
        # If we have leftover data, use it first
        elif self.leftover.shape[0] >= self.buffer_samples:
            buffered_audio = self.leftover[:self.buffer_samples]
            self.leftover = self.leftover[self.buffer_samples:]
        elif self.leftover.shape[0] < self.buffer_samples:
            # Need more audio data
            chunks = [self.leftover] if self.leftover.size > 0 else []
            total_samples = self.leftover.shape[0]
            
            while total_samples < self.buffer_samples:
                frame = self._get_audio_frame_with_timeout(1.0)
                
                if frame is None:
                    # No more audio available, pad with silence
                    remaining_samples = self.buffer_samples - total_samples
                    silence = np.zeros(remaining_samples, dtype=np.int16)
                    chunks.append(silence)
                    logger.debug(f"Padded {remaining_samples} samples with silence")
                    break
                else:
                    # Normal frame processing
                    if frame.sample_rate != self.sample_rate:
                        raise ValueError("Sample rate mismatch")
                    chunks.append(frame.side_data.input)
                    total_samples += frame.side_data.input.shape[0]
            
            if chunks:
                merged_audio = np.concatenate(chunks, dtype=np.int16)
                buffered_audio = merged_audio[:self.buffer_samples]
                self.leftover = merged_audio[self.buffer_samples:] if merged_audio.shape[0] > self.buffer_samples else np.empty(0, dtype=np.int16)
            else:
                # No chunks at all, create silent buffer
                buffered_audio = self._create_silent_audio(self.buffer_samples, self.sample_rate)
                
        # Convert numpy array to torch tensor and normalize int16 to float32
        waveform_tensor = torch.from_numpy(buffered_audio.astype(np.float32) / 32768.0)
        
        # Ensure proper tensor shape: (batch, channels, samples)
        if waveform_tensor.dim() == 1:
            # Mono: (samples,) -> (1, 1, samples)
            waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)
        elif waveform_tensor.dim() == 2:
            # Assume (channels, samples) and add batch dimension
            waveform_tensor = waveform_tensor.unsqueeze(0)
        
        # Return AUDIO dictionary format
        audio_dict = {
            "waveform": waveform_tensor,
            "sample_rate": self.sample_rate
        }
        
        return (audio_dict,)
