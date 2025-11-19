"""
Performance measurement nodes for ComfyStream batch processing.
These nodes integrate with the existing tensor_utils structure.
"""

from comfystream.utils import performance_timer


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
