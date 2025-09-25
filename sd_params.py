"""
StreamDiffusion parameter classes for ComfyStream compatibility.
This provides the minimal interface expected by the Livepeer build script.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ControlNetConfig:
    """Configuration for a ControlNet model."""
    model_id: str
    conditioning_scale: float = 1.0
    preprocessor: str = "passthrough"
    preprocessor_params: Dict[str, Any] = None
    enabled: bool = True
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    
    def __post_init__(self):
        if self.preprocessor_params is None:
            self.preprocessor_params = {}


@dataclass
class IPAdapterConfig:
    """Configuration for IPAdapter."""
    type: str = "regular"  # "regular" or "faceid"
    scale: float = 1.0
    enabled: bool = True


@dataclass
class StreamDiffusionParams:
    """Parameters for StreamDiffusion model configuration."""
    model_id: str
    t_index_list: List[int]
    acceleration: str = "tensorrt"
    width: int = 512
    height: int = 512
    controlnets: Optional[List[ControlNetConfig]] = None
    use_safety_checker: bool = True
    ip_adapter: Optional[IPAdapterConfig] = None
    
    def __post_init__(self):
        if self.controlnets is None:
            self.controlnets = []
