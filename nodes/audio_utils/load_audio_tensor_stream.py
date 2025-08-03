"""
Streaming Audio Tensor Loader for ComfyUI.

This node provides a pass-through interface for real-time audio processing,
designed for use with nodes that handle their own buffering (like AudioTranscriptionNode).
"""

import numpy as np
import logging
from typing import Optional, Union

from comfystream import tensor_cache

logger = logging.getLogger(__name__)


class LoadAudioTensorStream:
    """
    Streaming audio loader that passes individual audio frames without buffering.
    
    This node is optimized for real-time workflows where downstream nodes
    (like AudioTranscriptionNode) handle their own buffering strategies.
    """
    
    CATEGORY = "audio_utils"
    RETURN_TYPES = ("WAVEFORM", "INT")
    
    def __init__(self):
        self.sample_rate = None
        self.frame_count = 0
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_sample_rate": ("INT", {
                    "default": 16000,
                    "min": 8000,
                    "max": 48000,
                    "step": 1000,
                    "tooltip": "Target sample rate for audio processing (Whisper works best with 16kHz)"
                }),
                "timeout_ms": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Timeout in milliseconds for getting audio frames"
                })
            }
        }
    
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls):
        return float("nan")
    
    def execute(self, target_sample_rate: int = 16000, timeout_ms: int = 100):
        """
        Get the next audio frame from the input stream.
        
        Args:
            target_sample_rate: Target sample rate for processing
            timeout_ms: Timeout for frame retrieval
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Get frame from audio input cache (non-blocking for real-time)
            try:
                frame = tensor_cache.audio_inputs.get(block=True, timeout=timeout_ms/1000.0)
            except:
                # Return empty frame if no audio available
                logger.debug("No audio frame available, returning empty frame")
                return np.array([], dtype=np.int16), target_sample_rate
            
            if frame is None:
                return np.array([], dtype=np.int16), target_sample_rate
            
            # Extract audio data
            if hasattr(frame, 'side_data') and hasattr(frame.side_data, 'input'):
                audio_data = frame.side_data.input
                frame_sample_rate = getattr(frame, 'sample_rate', target_sample_rate)
            else:
                # Handle direct audio data
                audio_data = frame
                frame_sample_rate = target_sample_rate
            
            # Initialize sample rate on first frame
            if self.sample_rate is None:
                self.sample_rate = frame_sample_rate
                logger.info(f"Initialized audio stream: {frame_sample_rate}Hz -> {target_sample_rate}Hz target")
            
            # Convert to target format if needed
            audio_data = self._normalize_audio_frame(audio_data, frame_sample_rate, target_sample_rate)
            
            self.frame_count += 1
            
            if self.frame_count % 100 == 0:
                logger.debug(f"Processed {self.frame_count} audio frames")
            
            return audio_data, target_sample_rate
            
        except Exception as e:
            logger.error(f"Error in LoadAudioTensorStream: {e}")
            return np.array([], dtype=np.int16), target_sample_rate
    
    def _normalize_audio_frame(self, audio_data: np.ndarray, 
                             source_rate: int, target_rate: int) -> np.ndarray:
        """
        Normalize audio frame to target format.
        
        Args:
            audio_data: Input audio data
            source_rate: Source sample rate
            target_rate: Target sample rate
            
        Returns:
            Normalized audio data
        """
        if audio_data is None or audio_data.size == 0:
            return np.array([], dtype=np.int16)
        
        # Ensure numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Handle multi-channel audio (take first channel)
        if audio_data.ndim > 1:
            if audio_data.shape[1] > audio_data.shape[0]:
                # Shape is (samples, channels) - take first channel
                audio_data = audio_data[:, 0]
            else:
                # Shape is (channels, samples) - take first channel
                audio_data = audio_data[0, :]
        
        # Convert to int16 format
        if audio_data.dtype != np.int16:
            if audio_data.dtype in [np.float32, np.float64]:
                # Convert from float [-1, 1] to int16 [-32768, 32767]
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        # Simple resampling if rates don't match
        if source_rate != target_rate and source_rate > 0:
            # Basic resampling using linear interpolation
            # For production use, consider using scipy.signal.resample or librosa
            ratio = target_rate / source_rate
            new_length = int(len(audio_data) * ratio)
            
            if new_length > 0:
                indices = np.linspace(0, len(audio_data) - 1, new_length)
                audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data.astype(float))
                audio_data = audio_data.astype(np.int16)
            else:
                audio_data = np.array([], dtype=np.int16)
        
        return audio_data


# Register the node
NODE_CLASS_MAPPINGS = {
    "LoadAudioTensorStream": LoadAudioTensorStream
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudioTensorStream": "Load Audio Tensor (Stream)"
}