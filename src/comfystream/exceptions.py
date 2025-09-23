"""
ComfyStream specific exceptions.

These exceptions provide cleaner handling of expected conditions in the ComfyStream pipeline.
"""


class ComfyStreamInputTimeoutError(Exception):
    """
    Raised when input tensors (video/audio frames) are not available within the specified timeout.
    
    This is an expected condition that occurs when:
    - Streams are switching modalities
    - Input frames stop flowing temporarily  
    - During warmup before frames arrive
    - Stream connections are interrupted
    
    This exception should be handled gracefully as it indicates a temporary condition
    rather than a critical error.
    """
    
    def __init__(self, input_type: str, timeout_seconds: float, additional_info: str = None):
        """
        Args:
            input_type: Type of input that timed out (e.g., "video", "audio")
            timeout_seconds: The timeout value that was exceeded
            additional_info: Optional additional context about the timeout
        """
        self.input_type = input_type
        self.timeout_seconds = timeout_seconds
        self.additional_info = additional_info
        
        message = f"No {input_type} frames available after {timeout_seconds}s timeout"
        if additional_info:
            message += f". {additional_info}"
            
        super().__init__(message)


class ComfyStreamAudioBufferError(ComfyStreamInputTimeoutError):
    """
    Specific timeout error for audio buffer insufficient data scenarios.
    
    Raised when audio processing needs more samples than are available within the timeout.
    """
    
    def __init__(self, timeout_seconds: float, needed_samples: int, available_samples: int):
        """
        Args:
            timeout_seconds: The timeout that was exceeded
            needed_samples: Number of samples required for processing
            available_samples: Number of samples actually available
        """
        self.needed_samples = needed_samples
        self.available_samples = available_samples
        
        additional_info = f"insufficient data (need {needed_samples} samples, have {available_samples})"
        super().__init__("audio", timeout_seconds, additional_info)
