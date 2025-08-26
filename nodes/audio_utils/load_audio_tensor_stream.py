import numpy as np

from comfystream import tensor_cache


class LoadAudioTensorStream:
    CATEGORY = "audio_utils"
    RETURN_TYPES = ("WAVEFORM", "INT")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_sample_rate": ("INT", {"default": 16000, "min": 8000, "max": 96000}),
            },
            "optional": {
                # Accept both names for compatibility with different workflows
                "frame_ms": ("FLOAT", {"default": 500.0, "min": 10.0, "max": 2000.0}),
                "timeout_ms": ("FLOAT", {"default": 500.0, "min": 10.0, "max": 2000.0}),
            },
        }

    @classmethod
    def IS_CHANGED():
        return float("nan")

    def __init__(self):
        self.leftover = np.empty(0, dtype=np.int16)
        self.sample_rate = None
    def _convert_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio data to int16 format with proper scaling."""
        if audio.dtype in [np.float32, np.float64]:
            # Float audio in range [-1, 1] needs to be scaled to int16 range [-32768, 32767]
            audio = np.clip(audio, -1.0, 1.0)
            return (audio * 32767).astype(np.int16)
        else:
            # Already integer format, just convert to int16
            return audio.astype(np.int16)
         
    def _resample_if_needed(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return self._convert_to_int16(audio)
        # Simple linear resample to avoid heavy deps in node; acceptable for warmup/streaming
        ratio = dst_rate / float(src_rate)
        new_len = int(round(audio.shape[0] * ratio))
        if new_len <= 1 or audio.shape[0] <= 1:
            return np.zeros((max(new_len, 0),), dtype=np.int16)
        xp = np.linspace(0, 1, audio.shape[0], endpoint=True)
        fp = audio.astype(np.float32)
        x_new = np.linspace(0, 1, new_len, endpoint=True)
        out = np.interp(x_new, xp, fp).astype(np.int16)
        return out

    def execute(self, target_sample_rate: int, frame_ms: float = None, timeout_ms: float = None):
        # Determine frame size from provided args
        eff_ms = frame_ms if frame_ms is not None else (timeout_ms if timeout_ms is not None else 500.0)
        buffer_samples = int(target_sample_rate * eff_ms / 1000.0)

        # Initialize sample rate on first frame
        if self.sample_rate is None:
            frame = tensor_cache.audio_inputs.get(block=True)
            src_rate = getattr(frame, "sample_rate", target_sample_rate)
            audio = getattr(frame.side_data, "input", np.empty(0, dtype=np.int16))
            self.leftover = self._resample_if_needed(audio, src_rate, target_sample_rate)
            self.sample_rate = target_sample_rate

        # Accumulate until we have enough samples
        chunks = [self.leftover] if self.leftover.size > 0 else []
        total = self.leftover.shape[0]
        while total < buffer_samples:
            frame = tensor_cache.audio_inputs.get(block=True)
            src_rate = getattr(frame, "sample_rate", target_sample_rate)
            audio = getattr(frame.side_data, "input", np.empty(0, dtype=np.int16))
            resampled = self._resample_if_needed(audio, src_rate, target_sample_rate)
            chunks.append(resampled)
            total += resampled.shape[0]

        merged = np.concatenate(chunks, dtype=np.int16) if len(chunks) > 1 else chunks[0]
        out = merged[:buffer_samples]
        self.leftover = merged[buffer_samples:]

        return out, self.sample_rate


