"""
Secondary Channel Examples for ComfyStream Trickle Integration.

This module provides example implementations for using the secondary publish channel
feature with ComfyStream and TrickleClient. These examples demonstrate how to send
various types of metadata and analysis data alongside the main video stream.

Example use cases:
- ComfyUI pipeline status and metadata
- Frame analysis results
- Processing events and performance metrics
- Real-time diagnostics and monitoring
"""

import asyncio
import logging
import time
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SecondaryChannelExamples:
    """Example implementations for secondary channel usage in ComfyStream."""
    
    def __init__(self, client, request_id: str, processor=None):
        """Initialize with TrickleClient and processor references.
        
        Args:
            client: TrickleClient instance with secondary channel configured
            request_id: Stream request identifier  
            processor: ComfyStreamTrickleProcessor instance (optional)
        """
        self.client = client
        self.request_id = request_id
        self.processor = processor
    
    async def send_processing_metadata_to_secondary(self, pipeline=None):
        """Example: Send ComfyUI processing metadata to secondary channel.
        
        This demonstrates how to use the secondary publish channel to send
        detailed processing information alongside the main video stream.
        """
        if not hasattr(self.client, 'send_secondary_text'):
            logger.debug(f"Secondary text channel not available for {self.request_id}")
            return
            
        try:
            # Collect ComfyUI pipeline metadata
            pipeline_metadata = {
                "type": "comfyui_pipeline_status",
                "request_id": self.request_id,
                "timestamp": int(time.time() * 1000),
                "pipeline_info": {
                    "nodes_loaded": len(pipeline.client.current_prompts) if pipeline and hasattr(pipeline.client, 'current_prompts') else 0,
                    "pipeline_ready": self.processor.pipeline_ready if self.processor else True,
                    "model_info": {
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        "memory_allocated": torch.cuda.memory_allocated() // 1024**2 if torch.cuda.is_available() else 0,
                        "memory_reserved": torch.cuda.memory_reserved() // 1024**2 if torch.cuda.is_available() else 0
                    }
                },
                "processing_stats": {
                    "frames_processed": self.processor.frame_count if self.processor else 0,
                    "buffer_size": len(self.processor.frame_buffer.frames) if self.processor and hasattr(self.processor.frame_buffer, 'frames') else 0,
                    "last_frame_available": self.processor.last_processed_frame is not None if self.processor else False
                }
            }
            
            # Send to secondary channel
            await self.client.send_secondary_text(pipeline_metadata)
            logger.debug(f"Sent ComfyUI pipeline metadata to secondary channel for {self.request_id}")
            
        except Exception as e:
            logger.error(f"Error sending pipeline metadata to secondary channel for {self.request_id}: {e}")
    
    async def send_frame_analysis_to_secondary(self, frame_analysis: Dict[str, Any]):
        """Example: Send frame analysis results to secondary channel.
        
        This could be called when frames are processed to send detailed
        analysis results, object detection data, or other frame-specific metadata.
        
        Args:
            frame_analysis: Dictionary containing frame analysis results
        """
        if not hasattr(self.client, 'send_secondary_text'):
            logger.debug(f"Secondary text channel not available for {self.request_id}")
            return
            
        try:
            analysis_message = {
                "type": "frame_analysis",
                "request_id": self.request_id,
                "timestamp": int(time.time() * 1000),
                "frame_number": self.processor.frame_count if self.processor else 0,
                "analysis": frame_analysis
            }
            
            await self.client.send_secondary_text(analysis_message)
            logger.debug(f"Sent frame analysis to secondary channel for frame {self.processor.frame_count if self.processor else 0}")
            
        except Exception as e:
            logger.error(f"Error sending frame analysis to secondary channel: {e}")
    
    async def send_processing_event_to_secondary(self, event_type: str, event_data: Dict[str, Any]):
        """Example: Send processing events to secondary channel.
        
        This can be used to send real-time processing events like:
        - Node execution started/completed
        - Model loading events
        - Error conditions
        - Performance warnings
        
        Args:
            event_type: Type of event (e.g., "node_execution", "model_load", "error")
            event_data: Dictionary containing event-specific data
        """
        if not hasattr(self.client, 'send_secondary_text'):
            logger.debug(f"Secondary text channel not available for {self.request_id}")
            return
            
        try:
            processing_event = {
                "type": "processing_event",
                "event_type": event_type,
                "request_id": self.request_id,
                "timestamp": int(time.time() * 1000),
                "frame_context": {
                    "current_frame": self.processor.frame_count if self.processor else 0,
                    "pipeline_ready": self.processor.pipeline_ready if self.processor else True
                },
                "event_data": event_data
            }
            
            await self.client.send_secondary_text(processing_event)
            logger.debug(f"Sent processing event '{event_type}' to secondary channel for {self.request_id}")
            
        except Exception as e:
            logger.error(f"Error sending processing event to secondary channel: {e}")
    
    async def demonstrate_secondary_channel_usage(self, stream_config: Optional[Dict[str, Any]] = None):
        """Example: Demonstrate various secondary channel use cases.
        
        This method shows how to use the secondary channel for different
        types of metadata and real-time information during processing.
        
        Args:
            stream_config: Optional stream configuration info
        """
        if not hasattr(self.client, 'send_secondary_text'):
            logger.info(f"Secondary text channel not available for {self.request_id}")
            return
            
        logger.info(f"Starting secondary channel demonstrations for {self.request_id}")
        
        try:
            # Send initial pipeline status
            await self.send_processing_metadata_to_secondary()
            
            # Send stream start event
            await self.send_processing_event_to_secondary("stream_started", {
                "configuration": stream_config or {},
                "capabilities": {
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
                }
            })
            
            # Example frame analysis (simulated)
            sample_analysis = {
                "objects_detected": ["person", "car"],
                "confidence_scores": [0.95, 0.87],
                "processing_time_ms": 45,
                "model_inference": {
                    "model_name": "comfyui_workflow",
                    "inference_time_ms": 32,
                    "post_processing_time_ms": 13
                }
            }
            await self.send_frame_analysis_to_secondary(sample_analysis)
            
        except Exception as e:
            logger.error(f"Error in secondary channel demonstration: {e}")


# Standalone helper functions for easy import and use
async def send_comfyui_status_to_secondary(client, request_id: str, pipeline=None, processor=None):
    """Standalone function to send ComfyUI status to secondary channel."""
    examples = SecondaryChannelExamples(client, request_id, processor)
    await examples.send_processing_metadata_to_secondary(pipeline)


async def send_frame_analysis_to_secondary(client, request_id: str, frame_analysis: Dict[str, Any], processor=None):
    """Standalone function to send frame analysis to secondary channel."""
    examples = SecondaryChannelExamples(client, request_id, processor)
    await examples.send_frame_analysis_to_secondary(frame_analysis)


async def send_processing_event_to_secondary(client, request_id: str, event_type: str, event_data: Dict[str, Any], processor=None):
    """Standalone function to send processing events to secondary channel."""
    examples = SecondaryChannelExamples(client, request_id, processor)
    await examples.send_processing_event_to_secondary(event_type, event_data)


async def demonstrate_secondary_channel(client, request_id: str, stream_config: Optional[Dict[str, Any]] = None, processor=None):
    """Standalone function to demonstrate secondary channel capabilities."""
    examples = SecondaryChannelExamples(client, request_id, processor)
    await examples.demonstrate_secondary_channel_usage(stream_config) 