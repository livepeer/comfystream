"""
OpenCV CUDA setup utilities for ComfyStream
"""

from .opencv_utils import setup_opencv_cuda, is_cuda_available

__all__ = ['setup_opencv_cuda', 'is_cuda_available']
