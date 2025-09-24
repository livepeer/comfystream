"""ComfyStream specific exceptions."""

import logging

class ComfyStreamInputTimeoutError(Exception):
    """Raised when input tensors are not available within timeout."""

    def __init__(self, input_type: str, timeout_seconds: float, needed_samples: int = None, available_samples: int = None):
        self.input_type = input_type
        self.timeout_seconds = timeout_seconds
        if input_type == "audio":
            self.args = (f"No audio frames available after {timeout_seconds}s timeout (needed: {needed_samples} samples, available: {available_samples} samples)",timeout_seconds)
        else:
            self.args = (f"No {input_type} frames available after {timeout_seconds}s timeout",timeout_seconds)

        message = f"No {input_type} frames available after {timeout_seconds}s timeout"
        super().__init__(message)

class ComfyStreamAudioBufferError(ComfyStreamInputTimeoutError):
    """Audio buffer insufficient data error."""
    
    def __init__(self, timeout_seconds: float, needed_samples: int, available_samples: int):
        self.needed_samples = needed_samples
        self.available_samples = available_samples
        super().__init__("audio", timeout_seconds, needed_samples, available_samples)


class ComfyStreamTimeoutFilter(logging.Filter):
    """Filter to suppress verbose ComfyUI execution logs for ComfyStream timeout exceptions."""

    def filter(self, record):
        """Filter out ComfyUI execution error logs for ComfyStream timeout exceptions."""
        # Only filter ERROR level messages from ComfyUI execution system
        if record.levelno != logging.ERROR:
            return True

        # Check if this is from ComfyUI execution system
        if not (record.name.startswith("comfy") and ("execution" in record.name or record.name == "comfy")):
            return True

        # Get the full message including any exception info
        message = record.getMessage()

        # Check if this is a ComfyStream timeout-related error
        timeout_indicators = [
            "ComfyStreamInputTimeoutError",
            "ComfyStreamAudioBufferError",
            "No video frames available",
            "No audio frames available"
        ]

        # Suppress if any timeout indicator is found in the message
        for indicator in timeout_indicators:
            if indicator in message:
                return False

        # Also check the exception info if present
        if record.exc_info and record.exc_info[1]:
            exc_str = str(record.exc_info[1])
            for indicator in timeout_indicators:
                if indicator in exc_str:
                    return False

        return True
