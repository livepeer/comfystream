"""
ComfyStream Trickle Client

Provides functionality to start streaming requests to ComfyStream using
TricklePublisher and TrickleSubscriber from the trickle-app package.
"""

import asyncio
import logging
import json
import aiohttp
import torch
from typing import Optional, Dict, Any, Callable

try:
    from trickle_app import TricklePublisher, TrickleSubscriber, VideoFrame, VideoOutput
except ImportError:
    raise ImportError("trickle-app package not found. Install with: pip install git+https://github.com/eliteprox/py-trickle.git")

logger = logging.getLogger(__name__)


class ComfyStreamTrickleClient:
    """Client for streaming to ComfyStream using trickle protocol."""
    
    def __init__(
        self,
        comfystream_host: str = "localhost",
        comfystream_port: int = 9876,
        subscribe_url: Optional[str] = None,
        publish_url: Optional[str] = None
    ):
        """
        Initialize ComfyStream Trickle Client.
        
        Args:
            comfystream_host: ComfyStream server host
            comfystream_port: ComfyStream server port  
            subscribe_url: URL to subscribe to incoming stream (optional)
            publish_url: URL to publish outgoing stream (optional)
        """
        self.comfystream_host = comfystream_host
        self.comfystream_port = comfystream_port
        self.base_url = f"http://{comfystream_host}:{comfystream_port}"
        
        # Default trickle URLs if not provided
        self.subscribe_url = subscribe_url or f"{self.base_url}/input/stream"
        self.publish_url = publish_url or f"{self.base_url}/output/stream"
        
        self.subscriber: Optional[TrickleSubscriber] = None
        self.publisher: Optional[TricklePublisher] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        
    async def start(self):
        """Start the ComfyStream trickle client."""
        if self.running:
            logger.warning("Client already running")
            return
            
        logger.info(f"Starting ComfyStream trickle client: {self.base_url}")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Initialize subscriber and publisher
        self.subscriber = TrickleSubscriber(self.subscribe_url)
        self.publisher = TricklePublisher(self.publish_url, "video/mp4")
        
        # Start subscriber and publisher
        await self.subscriber.start()
        await self.publisher.start()
        
        self.running = True
        logger.info("ComfyStream trickle client started successfully")
        
    async def stop(self):
        """Stop the ComfyStream trickle client."""
        if not self.running:
            return
            
        logger.info("Stopping ComfyStream trickle client")
        
        if self.subscriber:
            await self.subscriber.close()
            
        if self.publisher:
            await self.publisher.close()
            
        if self.session:
            await self.session.close()
            
        self.running = False
        logger.info("ComfyStream trickle client stopped")
        
    async def start_stream(
        self,
        prompts: list,
        width: int = 512,
        height: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Start a stream processing request to ComfyStream.
        
        Args:
            prompts: List of ComfyUI workflow prompts
            width: Video width
            height: Video height
            **kwargs: Additional parameters for the stream
            
        Returns:
            Dict containing stream response data
        """
        if not self.running:
            await self.start()
            
        # Prepare stream request payload
        payload = {
            "prompts": prompts,
            "width": width,
            "height": height,
            "subscribe_url": self.subscribe_url,
            "publish_url": self.publish_url,
            **kwargs
        }
        
        try:
            # Make request to ComfyStream to start processing
            async with self.session.post(
                f"{self.base_url}/api/stream/start",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Stream started successfully: {result}")
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to start stream: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            raise
            
    async def update_prompts(self, prompts: list) -> Dict[str, Any]:
        """
        Update prompts for the current stream.
        
        Args:
            prompts: New list of prompts
            
        Returns:
            Dict containing update response
        """
        if not self.running:
            raise Exception("Client not running. Call start() first.")
            
        payload = {"prompts": prompts}
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/prompts/update",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Prompts updated successfully: {result}")
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to update prompts: {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error updating prompts: {e}")
            raise
            
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of ComfyStream.
        
        Returns:
            Dict containing status information
        """
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return {"status": "healthy", "running": self.running}
                else:
                    return {"status": "unhealthy", "running": self.running}
                    
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"status": "error", "error": str(e), "running": self.running}
            
    async def stream_video_frames(
        self,
        frame_processor: Callable[[VideoFrame], VideoOutput],
        max_frames: Optional[int] = None
    ):
        """
        Stream and process video frames through the trickle connection.
        
        Args:
            frame_processor: Function to process each video frame
            max_frames: Maximum number of frames to process (None for unlimited)
        """
        if not self.running:
            raise Exception("Client not running. Call start() first.")
            
        frames_processed = 0
        
        try:
            # Subscribe to incoming frames
            await self.subscriber.subscribe()
            
            while self.running and (max_frames is None or frames_processed < max_frames):
                # Get next segment from subscriber
                segment = await self.subscriber.next()
                if segment is None:
                    break
                    
                # Read segment data
                frame_data = await segment.read()
                if frame_data:
                    # Convert to VideoFrame (this would need proper video decoding)
                    # For now, we'll create a mock frame
                    frame = VideoFrame.from_tensor(
                        torch.zeros((3, 512, 512)),  # Mock tensor
                        timestamp=frames_processed
                    )
                    
                    # Process frame
                    processed_output = frame_processor(frame)
                    
                    # Publish processed frame
                    await self._publish_frame(processed_output)
                    
                    frames_processed += 1
                    
                # Close segment
                await segment.close()
                
                # Check if end of stream
                if segment.eos():
                    break
                    
        except Exception as e:
            logger.error(f"Error in video frame streaming: {e}")
            raise
        finally:
            await self.subscriber.unsubscribe()
            
    async def _publish_frame(self, output: VideoOutput):
        """
        Publish a processed video frame.
        
        Args:
            output: Processed video output to publish
        """
        try:
            # Get next segment writer
            segment_writer = await self.publisher.next()
            
            # Convert VideoOutput to bytes (this would need proper video encoding)
            # For now, we'll create mock data
            frame_bytes = b"mock_frame_data"
            
            async with segment_writer:
                await segment_writer.write(frame_bytes)
                
        except Exception as e:
            logger.error(f"Error publishing frame: {e}")
            raise


# Example usage function
async def example_usage():
    """Example of how to use ComfyStreamTrickleClient."""
    
    # Example ComfyUI workflow
    example_workflow = {
        "3": {
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        # ... more workflow nodes
    }
    
    # Frame processor function
    def process_frame(frame: VideoFrame) -> VideoOutput:
        """Simple frame processor that passes through frames."""
        return VideoOutput(frame, "example-request")
    
    # Use the client
    async with ComfyStreamTrickleClient() as client:
        # Start streaming with ComfyUI workflow
        stream_info = await client.start_stream(
            prompts=[example_workflow],
            width=512,
            height=512
        )
        print(f"Stream started: {stream_info}")
        
        # Get status
        status = await client.get_status()
        print(f"Status: {status}")
        
        # Process some frames
        await client.stream_video_frames(process_frame, max_frames=10)


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
