"""
StreamDiffusion pipeline loading functions for ComfyStream compatibility.
This provides the minimal interface expected by the Livepeer build script.
"""

import os
import logging
from typing import Optional
from sd_params import StreamDiffusionParams

logger = logging.getLogger(__name__)


def load_streamdiffusion_sync(
    params: StreamDiffusionParams,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
    engine_dir: str = "engines",
    build_engines: bool = False,
    **kwargs
) -> None:
    """
    Load StreamDiffusion model synchronously and optionally build TensorRT engines.
    
    This is a stub implementation that focuses on engine building.
    For ComfyStream, the actual model loading is handled by ComfyUI nodes.
    
    Args:
        params: StreamDiffusion parameters configuration
        min_batch_size: Minimum batch size for dynamic engines
        max_batch_size: Maximum batch size for dynamic engines  
        engine_dir: Directory to save/load TensorRT engines
        build_engines: Whether to build TensorRT engines
        **kwargs: Additional keyword arguments
    """
    if not build_engines:
        logger.info("Engine building disabled, skipping StreamDiffusion engine build")
        return
        
    logger.info(f"Building StreamDiffusion TensorRT engines for model: {params.model_id}")
    logger.info(f"Target dimensions: {params.width}x{params.height}")
    logger.info(f"Batch size range: {min_batch_size}-{max_batch_size}")
    logger.info(f"Timesteps: {len(params.t_index_list)} ({params.t_index_list})")
    
    # Create engine directory
    os.makedirs(engine_dir, exist_ok=True)
    
    if params.controlnets:
        logger.info(f"ControlNets configured: {len(params.controlnets)}")
        for i, cn in enumerate(params.controlnets):
            logger.info(f"  {i}: {cn.model_id}")
    
    if params.ip_adapter:
        logger.info(f"IPAdapter configured: {params.ip_adapter.type}")
        
    # This is where the actual TensorRT engine building would happen
    # For now, we'll rely on the ComfyUI-StreamDiffusion node to handle this
    logger.info("StreamDiffusion engine building delegated to ComfyUI-StreamDiffusion node")
