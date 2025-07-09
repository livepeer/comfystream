#!/usr/bin/env python3
"""
ComfyStream Trickle Integration Example

This example shows how to use TrickleSubscriber and TricklePublisher
to start streaming requests to ComfyStream.
"""

import asyncio
import json
import logging
import aiohttp
from typing import Optional, Dict, Any

# Use the comfystream conda environment
import sys
sys.path.insert(0, '/workspace/miniconda3/envs/comfystream/lib/python3.11/site-packages')

try:
    from trickle_app import TrickleSubscriber, TricklePublisher, VideoFrame, VideoOutput
    print("✅ Successfully imported trickle-app components")
except ImportError as e:
    print(f"❌ Failed to import trickle-app: {e}")
    # Fallback to local implementation
    from ai_runner.runner.app.live.trickle.trickle_subscriber import TrickleSubscriber
    print("✅ Using local TrickleSubscriber implementation")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComfyStreamTrickleManager:
    """Manager for ComfyStream trickle streaming integration."""
    
    def __init__(
        self,
        comfystream_host: str = "localhost",
        comfystream_port: int = 9876,
        input_port: int = 5678,
        output_port: int = 5679
    ):
        self.comfystream_host = comfystream_host
        self.comfystream_port = comfystream_port
        self.base_url = f"http://{comfystream_host}:{comfystream_port}"
        
        # Trickle streaming URLs
        self.input_url = f"http://{comfystream_host}:{input_port}"
        self.output_url = f"http://{comfystream_host}:{output_port}"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.subscriber: Optional[TrickleSubscriber] = None
        self.publisher: Optional[TricklePublisher] = None
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        
    async def start(self):
        """Start the trickle manager."""
        logger.info("Starting ComfyStream trickle manager")
        
        # Create HTTP session for API calls
        self.session = aiohttp.ClientSession()
        
        # Initialize trickle components
        self.subscriber = TrickleSubscriber(self.input_url)
        if 'TricklePublisher' in globals():
            self.publisher = TricklePublisher(self.output_url, "video/mp4")
        
        logger.info("ComfyStream trickle manager started")
        
    async def stop(self):
        """Stop the trickle manager."""
        logger.info("Stopping ComfyStream trickle manager")
        
        if self.subscriber:
            await self.subscriber.close()
            
        if self.publisher:
            await self.publisher.close()
            
        if self.session:
            await self.session.close()
            
        logger.info("ComfyStream trickle manager stopped")
        
    async def check_comfystream_health(self) -> bool:
        """Check if ComfyStream is running and healthy."""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
            
    async def set_workflow_prompts(self, prompts: list) -> Dict[str, Any]:
        """Set ComfyUI workflow prompts via API."""
        try:
            async with self.session.post(
                f"{self.base_url}/api/set_prompt",
                json=prompts,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return {"success": True, "message": "Prompts set successfully"}
                else:
                    error_text = await response.text()
                    return {"success": False, "error": error_text}
        except Exception as e:
            logger.error(f"Failed to set prompts: {e}")
            return {"success": False, "error": str(e)}
            
    async def start_streaming(self, workflow: dict, width: int = 512, height: int = 512):
        """Start streaming with a ComfyUI workflow."""
        logger.info("Starting ComfyStream with trickle streaming")
        
        # First, set the workflow prompts
        result = await self.set_workflow_prompts([workflow])
        if not result.get("success"):
            raise Exception(f"Failed to set workflow: {result.get('error')}")
            
        logger.info("✅ Workflow prompts set successfully")
        
        # Start trickle subscriber to listen for processed frames
        await self.subscriber.__aenter__()
        logger.info("✅ Trickle subscriber started")
        
        # Start publisher if available
        if self.publisher:
            await self.publisher.__aenter__()
            logger.info("✅ Trickle publisher started")
            
        return {"success": True, "message": "Streaming started"}
        
    async def process_stream(self, max_frames: Optional[int] = None):
        """Process streaming frames through trickle."""
        if not self.subscriber:
            raise Exception("Subscriber not initialized")
            
        frames_processed = 0
        logger.info("Starting frame processing...")
        
        try:
            while max_frames is None or frames_processed < max_frames:
                # Get next segment from subscriber
                segment = await self.subscriber.next()
                if segment is None:
                    logger.info("No more segments available")
                    break
                    
                # Read segment data
                data = await segment.read()
                if data:
                    logger.info(f"Received segment data: {len(data)} bytes")
                    
                    # In a real implementation, you would:
                    # 1. Decode the video data to frames
                    # 2. Process through ComfyUI
                    # 3. Encode and publish results
                    
                    frames_processed += 1
                    
                await segment.close()
                
                # Check for end of stream
                if segment.eos():
                    logger.info("End of stream reached")
                    break
                    
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            raise
        finally:
            logger.info(f"Processed {frames_processed} frames")


def create_example_comfyui_workflow():
    """Create an example ComfyUI workflow for image generation."""
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
            "_meta": {"title": "CLIP Text Encode (Negative)"}
        }
    }


async def main():
    """Main example function."""
    print("🚀 ComfyStream + Trickle Integration")
    print("=" * 40)
    
    # Create the workflow
    workflow = create_example_comfyui_workflow()
    print(f"📋 Created ComfyUI workflow with {len(workflow)} nodes")
    
    # Test the integration
    async with ComfyStreamTrickleManager() as manager:
        # Check ComfyStream health
        print("🔍 Checking ComfyStream health...")
        healthy = await manager.check_comfystream_health()
        
        if healthy:
            print("✅ ComfyStream is healthy")
            
            # Start streaming
            print("🌊 Starting trickle streaming...")
            result = await manager.start_streaming(workflow)
            
            if result.get("success"):
                print("✅ Streaming started successfully")
                
                # Process a few frames (mock)
                print("🎥 Processing stream frames...")
                await manager.process_stream(max_frames=5)
                
            else:
                print(f"❌ Failed to start streaming: {result.get('error')}")
                
        else:
            print("❌ ComfyStream is not healthy")
            print("   Make sure ComfyStream relay server is running:")
            print("   python /workspace/comfystream/server/main.py --host=0.0.0.0 --port=9876 --media-ports=5678,5679,5680,5681")
    
    print("\n🎯 Integration example complete!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Detailed error:")
