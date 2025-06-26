from typing import Dict, Any, Optional
from comfystream.pipeline import Pipeline
from aiohttp import web
import asyncio
import uuid
import logging
from dataclasses import dataclass
from datetime import datetime
from comfystream.server.trickle import high_throughput_segment_publisher, TrickleSegmentEncoder
from trickle_pipeline import _run_trickle_pipeline_airunner_style
from trickle_pipeline import StreamManifest

logger = logging.getLogger(__name__)

async def _process_video_capability(request: web.Request, request_data: Dict, header_data: Dict) -> Dict:
    """Process video through ComfyStream pipeline with trickle streaming"""
    
    # Extract prompts and configuration
    prompts = request_data.get('prompts', [])
    width = request_data.get('width', 512)
    height = request_data.get('height', 512)
    stream_url = request_data.get('stream_url', '')
    
    if not prompts:
        raise ValueError("No prompts provided for video processing")
        
    if not stream_url:
        raise ValueError("No stream URL provided for video output")
    
    # Create a pipeline for this request
    pipeline = Pipeline(
        width=width,
        height=height,
        cwd=request.app["workspace"],
        disable_cuda_malloc=True,
        gpu_only=True,
        preview_method='none',
        comfyui_inference_log_level=request.app.get("comfui_inference_log_level", None),
    )
    
    try:
        await pipeline.set_prompts(prompts)
        
        # Create manifest for tracking
        manifest_id = str(uuid.uuid4())
        stream_manifest = StreamManifest(
            manifest_id=manifest_id,
            input_stream_url="",  # Will be set based on the request
            output_stream_url=stream_url,
            created_at=datetime.now(),
            status='starting',
            pipeline=pipeline,
            frame_queue=asyncio.Queue(),
            metadata={
                'width': width,
                'height': height,
                'capability': 'comfystream-video'
            }
        )
        
        # Start the streaming pipeline
        if stream_manifest.frame_queue is not None:
            stream_manifest.publisher_task = asyncio.create_task(
                high_throughput_segment_publisher(
                    stream_manifest.output_stream_url, 
                    stream_manifest.frame_queue, 
                    max_fps=24.0,  # Match segmenter FPS
                    skip_frame_on_backlog=True
                )
            )
        
        request.app["active_streams"][manifest_id] = stream_manifest
        stream_manifest.status = 'active'
        
        return {
            'success': True,
            'manifest_id': manifest_id,
            'stream_url': stream_url,
            'message': 'Video processing pipeline started'
        }
        
    except Exception as e:
        await pipeline.cleanup()
        raise e

async def _process_image_capability(request: web.Request, request_data: Dict, header_data: Dict) -> Dict:
    """Process image through ComfyStream pipeline"""
    
    # Extract prompts and configuration
    prompts = request_data.get('prompts', [])
    width = request_data.get('width', 512)
    height = request_data.get('height', 512)
    
    if not prompts:
        raise ValueError("No prompts provided for image processing")
    
    # Create a pipeline for this request
    pipeline = Pipeline(
        width=width,
        height=height,
        cwd=request.app["workspace"],
        disable_cuda_malloc=True,
        gpu_only=True,
        preview_method='none',
        comfyui_inference_log_level=request.app.get("comfui_inference_log_level", None),
    )
    
    try:
        await pipeline.set_prompts(prompts)
        
        # For single image processing, we don't need streaming
        # This would be implemented based on specific image processing needs
        
        return {
            'success': True,
            'message': 'Image processing completed',
            'width': width,
            'height': height
        }
        
    finally:
        await pipeline.cleanup()

async def _process_trickle_stream_direct(stream_manifest: StreamManifest):
    """Trickle stream processing using ai-runner pattern with FFmpeg timing management"""
    try:
        pipeline = stream_manifest.pipeline
        frame_queue = stream_manifest.frame_queue
        
        if frame_queue is None or pipeline is None:
            logger.error("Frame queue or pipeline is None, cannot process frames")
            return
            
        metadata = stream_manifest.metadata or {}
        width = metadata.get('width', 512)
        height = metadata.get('height', 512)
        
        logger.info(f"Starting ai-runner pattern trickle processing for stream {stream_manifest.manifest_id}")
        logger.info(f"Input: {stream_manifest.input_stream_url} → Output: {stream_manifest.output_stream_url}")
        
        # Use ai-runner's proven pattern: FFmpeg for decode → process frames → FFmpeg for encode
        await _run_trickle_pipeline_airunner_style(
            stream_manifest.input_stream_url,
            stream_manifest.output_stream_url, 
            pipeline,
            frame_queue,
            width,
            height,
            stream_manifest
        )
        
        logger.info(f"AI-runner pattern trickle processing finished for stream {stream_manifest.manifest_id}")
                        
    except Exception as e:
        logger.error(f"Trickle processor error for stream {stream_manifest.manifest_id}: {e}")
    finally:
        # Signal end of stream
        try:
            if stream_manifest.frame_queue is not None:
                await stream_manifest.frame_queue.put(None)
        except:
            pass
