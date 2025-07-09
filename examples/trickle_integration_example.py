#!/usr/bin/env python3
"""
Example: Starting ComfyStream requests using trickle-app package

This example demonstrates how to use TricklePublisher and TrickleSubscriber
from the trickle-app package to interact with ComfyStream.

Install trickle-app first:
pip install git+https://github.com/eliteprox/py-trickle.git
"""

import asyncio
import logging
import json
import aiohttp
from typing import Optional, Dict, Any

# Import from the installed trickle-app package
try:
    from trickle_app import TricklePublisher, TrickleSubscriber, VideoFrame, VideoOutput
    print("✓ Successfully imported trickle-app components")
except ImportError as e:
    print(f"✗ Failed to import trickle-app: {e}")
    print("Install with: pip install git+https://github.com/eliteprox/py-trickle.git")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComfyStreamController:
    """Controller for starting and managing ComfyStream requests via HTTP API."""
    
    def __init__(self, host: str = "localhost", port: int = 9876):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_health(self) -> bool:
        """Check if ComfyStream server is healthy."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def set_prompts(self, prompts: list) -> Dict[str, Any]:
        """Set prompts via ComfyStream API."""
        try:
            async with self.session.post(
                f"{self.base_url}/api/set_prompt",
                json=prompts,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return {"success": True, "status": "prompts_set"}
                else:
                    error = await response.text()
                    return {"success": False, "error": error}
        except Exception as e:
            logger.error(f"Failed to set prompts: {e}")
            return {"success": False, "error": str(e)}


class TrickleStreamManager:
    """Manager for trickle streaming with ComfyStream."""
    
    def __init__(
        self,
        subscribe_url: str = "http://localhost:5678/input",
        publish_url: str = "http://localhost:5679/output"
    ):
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.subscriber: Optional[TrickleSubscriber] = None
        self.publisher: Optional[TricklePublisher] = None
        self.running = False
        
    async def start(self):
        """Start trickle streaming components."""
        if self.running:
            return
            
        logger.info("Starting trickle stream manager")
        
        # Initialize trickle components
        self.subscriber = TrickleSubscriber(self.subscribe_url)
        self.publisher = TricklePublisher(self.publish_url, "video/mp4")
        
        # Start them (using context managers)
        await self.subscriber.__aenter__()
        await self.publisher.__aenter__()
        
        self.running = True
        logger.info("Trickle stream manager started")
        
    async def stop(self):
        """Stop trickle streaming components."""
        if not self.running:
            return
            
        logger.info("Stopping trickle stream manager")
        
        if self.subscriber:
            await self.subscriber.__aexit__(None, None, None)
        if self.publisher:
            await self.publisher.__aexit__(None, None, None)
            
        self.running = False
        logger.info("Trickle stream manager stopped")
        
    async def process_stream(self, frame_processor, max_frames: Optional[int] = None):
        """Process streaming frames through trickle."""
        if not self.running:
            raise RuntimeError("Stream manager not started")
            
        frames_processed = 0
        
        try:
            # Start subscribing to the stream
            success = await self.subscriber.subscribe()
            if not success:
                raise RuntimeError("Failed to start subscription")
                
            logger.info("Started processing stream")
            
            while self.running and (max_frames is None or frames_processed < max_frames):
                try:
                    # Get next segment
                    segment = await self.subscriber.next()
                    if segment is None:
                        logger.info("No more segments available")
                        break
                        
                    # Read segment data
                    data = await segment.read()
                    if data:
                        # Create a mock VideoFrame for demonstration
                        # In real usage, you'd decode the data to a proper frame
                        import torch
                        mock_tensor = torch.randn(3, 512, 512)  # CHW format
                        frame = VideoFrame.from_tensor(mock_tensor, frames_processed)
                        
                        # Process the frame
                        output = frame_processor(frame)
                        
                        # Publish the result
                        await self._publish_output(output)
                        
                        frames_processed += 1
                        logger.info(f"Processed frame {frames_processed}")
                        
                    await segment.close()
                    
                    # Check for end of stream
                    if segment.eos():
                        logger.info("End of stream reached")
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    break
                    
        finally:
            await self.subscriber.unsubscribe()
            logger.info(f"Stream processing complete. Processed {frames_processed} frames")
            
    async def _publish_output(self, output: VideoOutput):
        """Publish processed output via trickle."""
        try:
            # Get next segment writer
            segment_writer = await self.publisher.next()
            
            # In real usage, you'd encode the VideoOutput to bytes
            # For demo, we'll use mock data
            mock_data = f"processed_frame_{output.request_id}".encode()
            
            async with segment_writer:
                await segment_writer.write(mock_data)
                
        except Exception as e:
            logger.error(f"Failed to publish output: {e}")


async def example_comfyui_workflow():
    """Create an example ComfyUI workflow."""
    return {
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
        "4": {
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint - Base"}
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"}
        },
        "6": {
            "inputs": {
                "text": "beautiful landscape, high quality, detailed",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"}
        },
        "7": {
            "inputs": {
                "text": "low quality, blurry, distorted",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"}
        }
    }


def frame_processor(frame: VideoFrame) -> VideoOutput:
    """Example frame processor function."""
    # In real usage, this would apply ComfyUI processing
    # For demo, we just pass through with a request ID
    return VideoOutput(frame, f"processed_{frame.timestamp}")


async def main():
    """Main example function."""
    print("🚀 ComfyStream Trickle Integration Example")
    print("=" * 50)
    
    # Create example workflow
    workflow = await example_comfyui_workflow()
    print(f"📋 Created workflow with {len(workflow)} nodes")
    
    # Test the controller
    async with ComfyStreamController() as controller:
        print("🔍 Checking ComfyStream health...")
        healthy = await controller.check_health()
        
        if healthy:
            print("✅ ComfyStream is healthy")
            
            # Set prompts
            print("📝 Setting ComfyUI prompts...")
            result = await controller.set_prompts([workflow])
            
            if result.get("success"):
                print("✅ Prompts set successfully")
            else:
                print(f"❌ Failed to set prompts: {result.get('error')}")
                
        else:
            print("❌ ComfyStream is not healthy or not running")
            print("   Make sure ComfyStream server is running on localhost:9876")
            
    # Test trickle streaming (mock)
    print("\n🌊 Testing Trickle Streaming Components")
    stream_manager = TrickleStreamManager()
    
    try:
        await stream_manager.start()
        print("✅ Trickle stream manager started")
        
        # Note: This would require actual trickle endpoints to work
        # For demo, we'll just show the setup
        print("🎥 Stream processing would start here...")
        print("   (Requires actual trickle endpoints)")
        
    except Exception as e:
        print(f"ℹ️  Stream demo (expected to fail without endpoints): {e}")
    finally:
        await stream_manager.stop()
        print("✅ Trickle stream manager stopped")
        
    print("\n🎯 Integration Example Complete!")
    print("To use with real ComfyStream:")
    print("1. Start ComfyStream relay server")
    print("2. Configure proper trickle endpoints")
    print("3. Run stream processing with real video data")


if __name__ == "__main__":
    asyncio.run(main())
