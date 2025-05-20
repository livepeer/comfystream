from faster_whisper import WhisperModel
import numpy as np
from collections import deque


class WhisperTranscribe:
    CATEGORY = "audio_utils"
    RETURN_TYPES = ("DICT",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    def __init__(self):
        self._model = None
        self._device = None
        self._model_size = None
        self._audio_buffer = deque()
        self._buffer_size = 2
        self._sample_rate = 16000
        self._last_result = {"text": "", "language": "", "status": "buffering"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("WAVEFORM",),
                "model_size": (
                    ["tiny", "small", "base", "medium", "large"],
                    {"default": "base"},
                ),
                "language": (["en", "es", "fr", "de", "it"], {"default": "en"}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
                "buffer_size": ("INT", {"default": 2, "min": 1, "max": 10}),
            }
        }

    @classmethod
    def IS_CHANGED(cls):
        return float("nan")

    def _initialize_model(self, model_size, device):
        # Only reinitialize the model if parameters have changed.
        if (
            self._model is None
            or self._model_size != model_size
            or self._device != device
        ):
            try:
                self._model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type="float16" if device == "cuda" else "int8",
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Whisper model: {e}")
            self._model_size = model_size
            self._device = device

    # def _buffer_audio(self, audio):
    #     self._audio_buffer.append(audio)

    #     # Check if the buffer contains enough chunks.
    #     if len(self._audio_buffer) >= self._buffer_size:
    #         buffered_audio = np.concatenate(self._audio_buffer, axis=0)
    #         self._audio_buffer = deque([self._audio_buffer[-1]])
    #         return buffered_audio
    #     return None

    def _buffer_audio(self, audio):
        self._audio_buffer.append(audio)

        # Check if the buffer contains enough chunks
        if len(self._audio_buffer) >= self._buffer_size:
            buffered_audio = np.concatenate(self._audio_buffer, axis=0)
            self._audio_buffer.clear()  # Clear the buffer after processing
            return buffered_audio
        return None

    def execute(
        self, audio, model_size="base", device="cpu", language="en", buffer_size=2
    ):
        audio_float = audio.astype(np.float32) / 32768.0
        self._buffer_size = buffer_size

        self._initialize_model(model_size, device)

        buffered_audio = self._buffer_audio(audio_float)
        if buffered_audio is None:
            try:
                segments, info = self._model.transcribe(
                    audio_float,
                    language=language,
                    # condition_on_previous_text=True,
                    # vad_filter=True,
                    # word_timestamps=True,
                )
                partial_transcription = "".join([seg.text for seg in segments])

                self._last_result["text"] += partial_transcription
                self._last_result["status"] = "partial"
            except Exception as e:
                self._last_result = {"error": str(e)}

            return (self._last_result,)

        try:
            segments, info = self._model.transcribe(
                buffered_audio,
                language=language,
                condition_on_previous_text=True,
                vad_filter=True,
                word_timestamps=True,
            )
            transcription = "".join([seg.text for seg in segments])

            self._last_result = {
                "text": transcription,
                "language": info.language,
                "status": "transcribed",
            }
        except Exception as e:
            self._last_result = {"error": str(e)}

        return (self._last_result,)
