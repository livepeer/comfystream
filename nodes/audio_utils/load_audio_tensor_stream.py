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
                    "default": 50,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Timeout in milliseconds for getting audio frames (lower = more responsive)"
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
                return (np.array([], dtype=np.int16), target_sample_rate)
            
            if frame is None:
                return (np.array([], dtype=np.int16), target_sample_rate)
            
            # Extract audio data - handle both side_data format and direct numpy arrays
            if hasattr(frame, 'side_data') and hasattr(frame.side_data, 'input'):
                # Legacy format with side_data
                audio_data = frame.side_data.input
                frame_sample_rate = getattr(frame, 'sample_rate', target_sample_rate)
            elif isinstance(frame, np.ndarray):
                # Direct numpy array from trickle processor
                audio_data = frame
                frame_sample_rate = target_sample_rate  # Use target rate since direct arrays don't carry rate info
            else:
                # Handle other direct audio data
                audio_data = frame
                frame_sample_rate = target_sample_rate
            
            # Initialize sample rate on first frame
            if self.sample_rate is None:
                self.sample_rate = frame_sample_rate
                logger.info(f"LoadAudioTensorStream: Initialized audio stream: {frame_sample_rate}Hz -> {target_sample_rate}Hz target")
            
            # Convert to target format if needed
            audio_data = self._normalize_audio_frame(audio_data, frame_sample_rate, target_sample_rate)
            
            self.frame_count += 1
            
            # Validate frame quality before sending
            if len(audio_data) == 0:
                logger.debug(f"LoadAudioTensorStream: Empty frame {self.frame_count}, skipping")
                return (np.array([], dtype=np.int16), target_sample_rate)
            
            # Return tuple format expected by AudioTranscriptionNode
            return (audio_data, target_sample_rate)
            
        except Exception as e:
            logger.error(f"Error in LoadAudioTensorStream: {e}")
            return (np.array([], dtype=np.int16), target_sample_rate)
    
    def _normalize_audio_frame(self, audio_data: np.ndarray, 
                             source_rate: int, target_rate: int) -> np.ndarray:
        """
        Normalize audio frame using centralized conversion logic.
        
        Args:
            audio_data: Input audio data
            source_rate: Source sample rate
            target_rate: Target sample rate
            
        Returns:
            Normalized audio data
        """
        if audio_data is None or (hasattr(audio_data, 'size') and audio_data.size == 0):
            return np.array([], dtype=np.int16)
        
        # Ensure numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Convert to int16 if needed
        if audio_data.dtype != np.int16:
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        # Handle sample rate conversion if needed
        if source_rate != target_rate and source_rate > 0:
            ratio = target_rate / source_rate
            new_length = int(len(audio_data) * ratio)
            if new_length > 0:
                indices = np.linspace(0, len(audio_data) - 1, new_length)
                audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data.astype(np.float64)).astype(np.int16)
        
        # Convert to mono (simplified)
        if audio_data.ndim > 1:
            if audio_data.shape[1] > audio_data.shape[0]:
                audio_data = audio_data[:, 0]
            else:
                audio_data = audio_data[0, :]
        
        return audio_data


# Register the node
NODE_CLASS_MAPPINGS = {
    "LoadAudioTensorStream": LoadAudioTensorStream
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudioTensorStream": "Load Audio Tensor (Stream)"
}