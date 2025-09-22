import numpy as np
import torch
import queue
import logging
from typing import Optional, Any

from comfystream import tensor_cache

logger = logging.getLogger(__name__)

class LoadAudioTensor:
    CATEGORY = "ComfyStream/Loaders"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    DESCRIPTION = "Load audio tensor from ComfyStream input with configurable buffer size and timeout. Raises exception if no audio input available within timeout period."
    
    def __init__(self):
        self.audio_buffer = np.empty(0, dtype=np.int16)
        self.buffer_samples = None
        self.sample_rate = None
        self.last_sample_rate = 44100  # Default fallback sample rate
        self.leftover = np.empty(0, dtype=np.int16)  # Initialize to prevent race conditions
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "buffer_size": ("FLOAT", {
                    "default": 500.0,
                    "tooltip": "Audio buffer size in milliseconds"
                }),
            },
            "optional": {
                "timeout_seconds": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 10.0, 
                    "step": 0.1,
                    "tooltip": "Maximum time to wait for audio frames before raising an error"
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
    
    def _get_audio_frame_with_timeout(self, timeout_seconds: float) -> Optional[Any]:
        """Get audio frame with timeout, return None if no frame available."""
        try:
            return tensor_cache.audio_inputs.get(block=True, timeout=timeout_seconds)
        except queue.Empty:
            return None
    
    
    def _initialize_if_needed(self, buffer_size: float, timeout_seconds: float) -> None:
        """Initialize audio parameters if needed.
        
        Args:
            buffer_size: Buffer size in milliseconds
            timeout_seconds: Timeout for waiting for frames
            
        Raises:
            RuntimeError: When no input available within timeout
        """
        if self.sample_rate is None or self.buffer_samples is None:
            frame = self._get_audio_frame_with_timeout(timeout_seconds)
            
            if frame is None:
                error_msg = f"No audio frames available in tensor cache after {timeout_seconds}s timeout. ComfyStream may not be receiving audio input or the workflow may not have audio input nodes."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                # Normal ComfyStream mode - remember this sample rate for future fallbacks
                self.sample_rate = frame.sample_rate
                self.last_sample_rate = self.sample_rate
                self.buffer_samples = int(self.sample_rate * buffer_size / 1000)
                self.leftover = frame.side_data.input
                logger.info(f"Audio input initialized: {self.sample_rate} Hz, {self.buffer_samples} samples per buffer")

    def _process_audio_buffer(self, timeout_seconds: float) -> np.ndarray:
        """Process the audio buffer and return buffered audio data.
        
        Args:
            timeout_seconds: Timeout for waiting for frames
            
        Returns:
            Buffered audio data as numpy array
        """
        # If we have enough leftover data, use it first
        if self.leftover.shape[0] >= self.buffer_samples:
            buffered_audio = self.leftover[:self.buffer_samples]
            self.leftover = self.leftover[self.buffer_samples:]
            return buffered_audio
        
        # Need more audio data
        return self._collect_audio_chunks(timeout_seconds)

    def _collect_audio_chunks(self, timeout_seconds: float) -> np.ndarray:
        """Collect audio chunks to fill the buffer."""
        chunks = [self.leftover] if self.leftover.size > 0 else []
        total_samples = self.leftover.shape[0]
        
        while total_samples < self.buffer_samples:
            frame = self._get_audio_frame_with_timeout(timeout_seconds)
            
            if frame is None:
                error_msg = f"Audio stream interrupted after {timeout_seconds}s timeout, insufficient data available (need {self.buffer_samples} samples, have {total_samples})"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                # Normal frame processing
                if frame.sample_rate != self.sample_rate:
                    raise ValueError(f"Sample rate mismatch: expected {self.sample_rate}Hz, got {frame.sample_rate}Hz")
                chunks.append(frame.side_data.input)
                total_samples += frame.side_data.input.shape[0]
        
        if chunks:
            merged_audio = np.concatenate(chunks, dtype=np.int16)
            buffered_audio = merged_audio[:self.buffer_samples]
            self.leftover = merged_audio[self.buffer_samples:] if merged_audio.shape[0] > self.buffer_samples else np.empty(0, dtype=np.int16)
            return buffered_audio
        else:
            # This should not happen given the logic above, but just in case
            error_msg = f"No audio data collected after timeout"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _format_audio_output(self, buffered_audio: np.ndarray) -> tuple:
        """Format buffered audio data into ComfyUI AUDIO format."""
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

    def execute(self, buffer_size: float, timeout_seconds: float = 1.0) -> tuple:
        self._initialize_if_needed(buffer_size, timeout_seconds)
        buffered_audio = self._process_audio_buffer(timeout_seconds)
        return self._format_audio_output(buffered_audio)
