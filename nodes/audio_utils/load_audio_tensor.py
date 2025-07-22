import numpy as np

from comfystream import tensor_cache

class LoadAudioTensor:
    CATEGORY = "audio_utils"
    RETURN_TYPES = ("AUDIO", "INT")
    FUNCTION = "execute"
    
    def __init__(self):
        self.audio_buffer = np.empty(0, dtype=np.int16)
        self.buffer_samples = None
        self.sample_rate = None
        self.leftover = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "buffer_size": ("FLOAT", {"default": 500.0}),
                "mode": (["chunked", "sliding window"], {"default": "chunked"}),
            }
        }
    
    @classmethod
    def IS_CHANGED():
        return float("nan")
    
    def execute(self, buffer_size, mode):
        # Initialize on first run
        if self.sample_rate is None or self.buffer_samples is None:
            frame = tensor_cache.audio_inputs.get(block=True)
            self.sample_rate = frame.sample_rate
            self.buffer_samples = int(self.sample_rate * buffer_size / 1000)
            self.leftover = frame.side_data.input
        
        # Collect enough samples for the buffer
        if self.leftover.shape[0] < self.buffer_samples:
            chunks = [self.leftover] if self.leftover.size > 0 else []
            total_samples = self.leftover.shape[0]
            
            while total_samples < self.buffer_samples:
                frame = tensor_cache.audio_inputs.get(block=True)
                if frame.sample_rate != self.sample_rate:
                    raise ValueError("Sample rate mismatch")
                chunks.append(frame.side_data.input)
                total_samples += frame.side_data.input.shape[0]
            
            merged_audio = np.concatenate(chunks, dtype=np.int16)
            buffered_audio = merged_audio[:self.buffer_samples]
            remaining_audio = merged_audio[self.buffer_samples:]
        else:
            buffered_audio = self.leftover[:self.buffer_samples]
            remaining_audio = self.leftover[self.buffer_samples:]
        
        # Handle different modes
        if mode == "chunked":
            # Chunked mode: clear all processed data, start fresh with remaining
            self.leftover = remaining_audio
        elif mode == "sliding_window":
            # Sliding window mode: keep a sliding window of data
            # Move the window forward by the buffer size, but keep some overlap
            overlap_samples = self.buffer_samples // 2  # 50% overlap
            advance_samples = self.buffer_samples - overlap_samples
            
            if self.leftover.shape[0] >= advance_samples:
                # Slide the window forward by advance_samples
                self.leftover = self.leftover[advance_samples:]
            else:
                # Not enough data to slide, use remaining
                self.leftover = remaining_audio
                
        return buffered_audio, self.sample_rate
