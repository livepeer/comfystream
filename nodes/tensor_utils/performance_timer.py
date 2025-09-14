import time
import torch
from typing import Dict, List, Optional
from contextlib import contextmanager


class PerformanceTimer:
    """Utility class for measuring performance metrics in ComfyStream workflows."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.current_timings: Dict[str, float] = {}
        self.batch_sizes: List[int] = []
        self.total_images_processed = 0
        
    def start_timing(self, operation: str):
        """Start timing an operation."""
        self.current_timings[operation] = time.time()
    
    def end_timing(self, operation: str):
        """End timing an operation and record the duration."""
        if operation in self.current_timings:
            duration = time.time() - self.current_timings[operation]
            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(duration)
            del self.current_timings[operation]
            return duration
        return 0.0
    
    def record_batch_processing(self, batch_size: int, num_images: int):
        """Record a batch processing event."""
        self.batch_sizes.append(batch_size)
        self.total_images_processed += num_images
    
    def get_fps(self, operation: str = "total") -> float:
        """Calculate FPS for a specific operation."""
        if operation not in self.timings or not self.timings[operation]:
            return 0.0
        
        total_time = sum(self.timings[operation])
        if total_time == 0:
            return 0.0
        
        return self.total_images_processed / total_time
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation."""
        if operation not in self.timings or not self.timings[operation]:
            return 0.0
        
        return sum(self.timings[operation]) / len(self.timings[operation])
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get a comprehensive performance summary."""
        summary = {
            "total_images_processed": self.total_images_processed,
            "total_fps": self.get_fps("total"),
            "average_batch_size": sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0,
        }
        
        for operation in self.timings:
            summary[f"{operation}_fps"] = self.get_fps(operation)
            summary[f"{operation}_avg_time"] = self.get_average_time(operation)
        
        return summary
    
    def reset(self):
        """Reset all performance data."""
        self.timings.clear()
        self.current_timings.clear()
        self.batch_sizes.clear()
        self.total_images_processed = 0
    
    @contextmanager
    def time_operation(self, operation: str):
        """Context manager for timing operations."""
        self.start_timing(operation)
        try:
            yield
        finally:
            self.end_timing(operation)


# Global performance timer instance
performance_timer = PerformanceTimer()


class PerformanceTimerNode:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("performance_summary",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "operation": ("STRING", {"default": "workflow_execution"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, operation: str, batch_size: int, num_images: int):
        """Record performance metrics and return summary."""
        performance_timer.record_batch_processing(batch_size, num_images)
        performance_timer.end_timing(operation)
        
        summary = performance_timer.get_performance_summary()
        
        # Format summary as readable string
        summary_str = f"Performance Summary:\n"
        summary_str += f"Total Images Processed: {summary['total_images_processed']}\n"
        summary_str += f"Total FPS: {summary['total_fps']:.2f}\n"
        summary_str += f"Average Batch Size: {summary['average_batch_size']:.2f}\n"
        
        for key, value in summary.items():
            if key not in ["total_images_processed", "total_fps", "average_batch_size"]:
                summary_str += f"{key}: {value:.4f}\n"
        
        return (summary_str,)


class StartPerformanceTimerNode:
    CATEGORY = "tensor_utils"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("timer_started",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "operation": ("STRING", {"default": "workflow_execution"}),
            }
        }

    @classmethod
    def IS_CHANGED(s):
        return float("nan")

    def execute(self, operation: str):
        """Start timing an operation."""
        performance_timer.start_timing(operation)
        return (f"Started timing: {operation}",)
