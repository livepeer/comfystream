from faster_whisper import WhisperModel

class WhisperTranscribe:
    CATEGORY = "audio_utils"
    RETURN_TYPES = ("DICT",)
    FUNCTION = "execute"
    OUTPUT_NODE = False

    def __init__(self):
        self.model = None
        self._device = None
        self._model_size = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("WAVEFORM",),
                "model_size": (["tiny", "base", "small", "medium", "large"], {"default": "base"}),
                "language": (["en", "es", "fr", "de", "it"], {"default": "en"}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def _initialize_model(self, model_size, device):
        # Only reinitialize the model if parameters have changed.
        if self.model is None or self._model_size != model_size or self._device != device:
            try:
                self.model = WhisperModel(
                    model_size, 
                    device=device, 
                    compute_type="float16" if device == "cuda" else "int8"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Whisper model: {e}")
            self._model_size = model_size
            self._device = device

    def execute(self, audio, model_size="base", device="cpu", language="en"):
        self._initialize_model(model_size, device)
        
        try:
            segments, info = self.model.transcribe(audio, language=language)
            transcription = "".join([seg.text for seg in segments])
        except Exception as e:
            return ({"error": str(e)},)            
        return ({"text": transcription, "language": info.language},)
