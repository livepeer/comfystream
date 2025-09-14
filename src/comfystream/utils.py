import copy
import importlib
import time
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from comfy.api.components.schema.prompt import Prompt, PromptDictInput
from .modalities import (
    get_node_counts_by_type,
    get_convertible_node_keys,
)

def create_load_tensor_node():
    return {
        "inputs": {},
        "class_type": "LoadTensor",
        "_meta": {"title": "LoadTensor"},
    }

def create_save_tensor_node(inputs: Dict[Any, Any]):
    return {
        "inputs": inputs,
        "class_type": "SaveTensor",
        "_meta": {"title": "SaveTensor"},
    }

def _validate_prompt_constraints(counts: Dict[str, int]) -> None:
    """Validate that the prompt meets the required constraints."""
    if counts["primary_inputs"] > 1:
        raise Exception("too many primary inputs in prompt")

    if counts["primary_inputs"] == 0 and counts["inputs"] > 2:
        raise Exception("too many inputs in prompt")

    if counts["outputs"] > 3:
        raise Exception("too many outputs in prompt")

    if counts["primary_inputs"] + counts["inputs"] == 0:
        raise Exception("missing input")

    if counts["outputs"] == 0:
        raise Exception("missing output")

def convert_prompt(prompt: PromptDictInput) -> Prompt:
    """Convert a prompt by replacing specific node types with tensor equivalents."""
    try:
        # Note: lazy import is necessary to prevent KeyError during validation
        importlib.import_module("comfy.api.components.schema.prompt_node")
    except Exception:
        pass
    
    # Validate the schema
    Prompt.validate(prompt)
    prompt = copy.deepcopy(prompt)

    # Count nodes and validate constraints
    counts = get_node_counts_by_type(prompt)
    _validate_prompt_constraints(counts)
    
    # Collect nodes that need conversion
    convertible_keys = get_convertible_node_keys(prompt)

    # Replace nodes based on their conversion type
    for key in convertible_keys["PrimaryInputLoadImage"]:
        prompt[key] = create_load_tensor_node()

    # Conditional replacement: only if no primary input and exactly one LoadImage
    if counts["primary_inputs"] == 0 and len(convertible_keys["LoadImage"]) == 1:
        prompt[convertible_keys["LoadImage"][0]] = create_load_tensor_node()

    # Replace output nodes
    for key in convertible_keys["PreviewImage"] + convertible_keys["SaveImage"]:
        node = prompt[key]
        prompt[key] = create_save_tensor_node(node["inputs"])

    # Validate the processed prompt
    prompt = Prompt.validate(prompt)
    return prompt


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
