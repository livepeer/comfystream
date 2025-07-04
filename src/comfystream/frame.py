"""
VideoFrame utilities for ComfyStream trickle protocol
Inspired by the Livepeer AI runner frame implementation
"""

import numpy as np
import av
import cv2
from typing import Optional, Union, Tuple
import time


class VideoFrame:
    """A wrapper for video frames that provides utility methods for trickle protocol"""
    
    def __init__(self, width: int, height: int, format: str = "rgb24"):
        self.width = width
        self.height = height
        self.format = format
        self.timestamp = time.time()
        self.pts: Optional[int] = None
        self.time_base: Optional[av.base.Fraction] = None  # type: ignore
        self._data: Optional[np.ndarray] = None
        
    @classmethod
    def from_numpy(cls, array: np.ndarray, format: str = "rgb24"):
        """Create VideoFrame from numpy array"""
        if len(array.shape) == 3:
            height, width, channels = array.shape
        else:
            height, width = array.shape
            channels = 1
            
        frame = cls(width, height, format)
        frame._data = array.copy()
        return frame
        
    @classmethod
    def from_av_frame(cls, av_frame: av.VideoFrame):
        """Create VideoFrame from PyAV VideoFrame"""
        array = av_frame.to_ndarray(format="rgb24")
        frame = cls.from_numpy(array, "rgb24")
        frame.pts = av_frame.pts
        frame.time_base = av_frame.time_base
        return frame
        
    @classmethod
    def create_dummy(cls, width: int = 512, height: int = 512, color: Tuple[int, int, int] = (128, 128, 128)):
        """Create a dummy VideoFrame with solid color"""
        array = np.full((height, width, 3), color, dtype=np.uint8)
        return cls.from_numpy(array)
        
    @classmethod
    def create_test_pattern(cls, width: int = 512, height: int = 512):
        """Create a test pattern VideoFrame"""
        # Create a simple test pattern with gradient
        array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Horizontal gradient
        for x in range(width):
            array[:, x, 0] = int(255 * x / width)  # Red gradient
            
        # Vertical gradient
        for y in range(height):
            array[y, :, 1] = int(255 * y / height)  # Green gradient
            
        # Blue checkerboard
        for y in range(0, height, 32):
            for x in range(0, width, 32):
                if (x // 32 + y // 32) % 2 == 0:
                    array[y:y+32, x:x+32, 2] = 255
                    
        return cls.from_numpy(array)
        
    def to_av_frame(self) -> av.VideoFrame:
        """Convert to PyAV VideoFrame"""
        if self._data is None:
            raise ValueError("No data available")
            
        av_frame = av.VideoFrame.from_ndarray(self._data, format=self.format)
        if self.pts is not None:
            av_frame.pts = self.pts
        if self.time_base is not None:
            av_frame.time_base = self.time_base
        return av_frame
        
    def to_numpy(self) -> np.ndarray:
        """Get numpy array representation"""
        if self._data is None:
            raise ValueError("No data available")
        return self._data.copy()
        
    def resize(self, new_width: int, new_height: int) -> 'VideoFrame':
        """Resize the frame"""
        if self._data is None:
            raise ValueError("No data available")
            
        resized_data = cv2.resize(self._data, (new_width, new_height))
        new_frame = VideoFrame(new_width, new_height, self.format)
        new_frame._data = resized_data
        new_frame.pts = self.pts
        new_frame.time_base = self.time_base
        return new_frame
        
    def save(self, filename: str):
        """Save frame to file"""
        if self._data is None:
            raise ValueError("No data available")
            
        # Convert RGB to BGR for OpenCV
        bgr_data = cv2.cvtColor(self._data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr_data)
        
    def __repr__(self):
        return f"VideoFrame(width={self.width}, height={self.height}, format={self.format})"


class AudioFrame:
    """A wrapper for audio frames that provides utility methods for trickle protocol"""
    
    def __init__(self, sample_rate: int = 48000, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.timestamp = time.time()
        self.pts: Optional[int] = None
        self.time_base: Optional[av.base.Fraction] = None  # type: ignore
        self._data: Optional[np.ndarray] = None
        
    @classmethod
    def from_numpy(cls, array: np.ndarray, sample_rate: int = 48000):
        """Create AudioFrame from numpy array"""
        if len(array.shape) == 1:
            channels = 1
        else:
            channels = array.shape[1] if len(array.shape) > 1 else 1
            
        frame = cls(sample_rate, channels)
        frame._data = array.copy()
        return frame
        
    @classmethod
    def from_av_frame(cls, av_frame: av.AudioFrame):
        """Create AudioFrame from PyAV AudioFrame"""
        array = av_frame.to_ndarray()
        frame = cls.from_numpy(array, av_frame.sample_rate)
        frame.pts = av_frame.pts
        frame.time_base = av_frame.time_base
        return frame
        
    @classmethod
    def create_silence(cls, duration_ms: int = 100, sample_rate: int = 48000, channels: int = 2):
        """Create a silent AudioFrame"""
        samples = int(sample_rate * duration_ms / 1000)
        array = np.zeros((samples, channels), dtype=np.int16)
        return cls.from_numpy(array, sample_rate)
        
    @classmethod
    def create_tone(cls, frequency: int = 440, duration_ms: int = 100, sample_rate: int = 48000, channels: int = 2):
        """Create a tone AudioFrame"""
        samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, samples)
        tone = np.sin(2 * np.pi * frequency * t)
        
        if channels == 1:
            array = (tone * 32767).astype(np.int16)
        else:
            array = np.column_stack([tone] * channels)
            array = (array * 32767).astype(np.int16)
            
        return cls.from_numpy(array, sample_rate)
        
    def to_av_frame(self) -> av.AudioFrame:
        """Convert to PyAV AudioFrame"""
        if self._data is None:
            raise ValueError("No data available")
            
        av_frame = av.AudioFrame.from_ndarray(self._data)
        av_frame.sample_rate = self.sample_rate
        if self.pts is not None:
            av_frame.pts = self.pts
        if self.time_base is not None:
            av_frame.time_base = self.time_base
        return av_frame
        
    def to_numpy(self) -> np.ndarray:
        """Get numpy array representation"""
        if self._data is None:
            raise ValueError("No data available")
        return self._data.copy()
        
    @property
    def samples(self) -> int:
        """Get number of samples"""
        if self._data is None:
            return 0
        return len(self._data)
        
    def __repr__(self):
        return f"AudioFrame(sample_rate={self.sample_rate}, channels={self.channels}, samples={self.samples})" 