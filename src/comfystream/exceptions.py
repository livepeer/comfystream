"""ComfyStream specific exceptions."""

import logging
from typing import Dict, Any, Optional


def log_comfystream_error(
    exception: Exception,
    logger: Optional[logging.Logger] = None,
    level: int = logging.ERROR
) -> None:
    """
    Centralized logging function for ComfyStream exceptions.
    
    Args:
        exception: The exception to log
        logger: Optional logger to use (defaults to module logger)
        level: Log level (defaults to ERROR)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # If it's a ComfyStream timeout error with structured details, use its logging method
    if isinstance(exception, ComfyStreamInputTimeoutError):
        exception.log_error(logger)
    else:
        # For other exceptions, provide basic logging
        logger.log(level, f"ComfyStream error: {type(exception).__name__}: {str(exception)}")


class ComfyStreamInputTimeoutError(Exception):
    """Raised when input tensors are not available within timeout."""
    
    def __init__(
        self,
        input_type: str,
        timeout_seconds: float,
        details: Optional[Dict[str, Any]] = None
    ):
        self.input_type = input_type
        self.timeout_seconds = timeout_seconds
        self.details = details or {}
        message = f"No {input_type} frames available after {timeout_seconds}s timeout"
        super().__init__(message)
    
    def get_log_details(self) -> Dict[str, Any]:
        """Get structured details for logging."""
        base_details = {
            "input_type": self.input_type,
            "timeout_seconds": self.timeout_seconds
        }
        base_details.update(self.details)
        return base_details
    
    def log_error(self, logger: Optional[logging.Logger] = None) -> None:
        """Log the error with detailed information."""
        if logger is None:
            logger = logging.getLogger(__name__)
        
        details = self.get_log_details()
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        logger.error(f"ComfyStream timeout error: {str(self)} | Details: {detail_str}")


class ComfyStreamAudioBufferError(ComfyStreamInputTimeoutError):
    """Audio buffer insufficient data error."""
    
    def __init__(
        self,
        timeout_seconds: float,
        needed_samples: int,
        available_samples: int
    ):
        self.needed_samples = needed_samples
        self.available_samples = available_samples
        
        # Pass audio-specific details to the base class
        audio_details = {
            "needed_samples": needed_samples,
            "available_samples": available_samples,
        }
        super().__init__("audio", timeout_seconds, details=audio_details)
    
    def get_log_details(self) -> Dict[str, Any]:
        """Get structured details for logging, with audio-specific formatting."""
        details = super().get_log_details()
        return details


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
