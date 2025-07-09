#!/usr/bin/env python3
"""
Example usage of ComfyStream with Trickle API for video pipeline processing.

This script demonstrates how to:
1. Start a ComfyStream server with trickle support
2. Create a stream with a ComfyUI workflow
3. Process video frames through the pipeline
4. Monitor stream status and cleanup
"""

import asyncio
import json
import logging
import aiohttp
import time
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComfyStreamTrickleExample:
    """Example class for using ComfyStream with Trickle."""
    
    def __init__(self, server_url="http://localhost:8889"):
        self.server_url = server_url
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def start_stream(self, request_id: str, subscribe_url: str, publish_url: str, 
                          width: int = 512, height: int = 512, prompt_dict: Optional[Dict[str, Any]] = None):
        """Start a new trickle stream with ComfyUI workflow."""
        
        # Default simple workflow if none provided
        if prompt_dict is None:
            prompt_dict = {
                "1": {
                    "inputs": {
                        "images": ["2", 0]
                    },
                    "class_type": "SaveTensor"
                },
                "2": {
                    "inputs": {},
                    "class_type": "LoadTensor"
                }
            }
        
        payload = {
            "subscribe_url": subscribe_url,
            "publish_url": publish_url,
            "gateway_request_id": request_id,
            "params": {
                "width": width,
                "height": height,
                "prompt": json.dumps(prompt_dict)
            }
        }
        
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        async with self.session.post(f"{self.server_url}/stream/start", json=payload) as response:
            result = await response.json()
            
            if response.status == 200 and result.get('status') == 'success':
                logger.info(f"✅ Stream {request_id} started successfully")
                return True
            else:
                logger.error(f"❌ Failed to start stream {request_id}: {result}")
                return False
    
    async def get_stream_status(self, request_id: str):
        """Get the status of a stream."""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        async with self.session.get(f"{self.server_url}/stream/{request_id}/status") as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error(f"Failed to get status for stream {request_id}")
                return None
    
    async def stop_stream(self, request_id: str):
        """Stop a stream."""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        async with self.session.post(f"{self.server_url}/stream/{request_id}/stop") as response:
            result = await response.json()
            
            if response.status == 200 and result.get('status') == 'success':
                logger.info(f"✅ Stream {request_id} stopped successfully")
                return True
            else:
                logger.error(f"❌ Failed to stop stream {request_id}: {result}")
                return False
    
    async def list_streams(self):
        """List all active streams."""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        async with self.session.get(f"{self.server_url}/streams") as response:
            if response.status == 200:
                return await response.json()
            else:
                logger.error("Failed to list streams")
                return None

async def example_usage():
    """Example usage of the trickle API."""
    
    logger.info("🚀 Starting ComfyStream Trickle API Example")
    
    # Example stream configuration
    request_id = f"example-stream-{int(time.time())}"
    subscribe_url = "http://192.168.10.61:3389/input-stream"
    publish_url = "http://192.168.10.61:3389/output-stream"
    
    # Example ComfyUI workflow for image processing
    workflow = {
        "1": {
            "inputs": {
                "images": ["2", 0]
            },
            "class_type": "SaveTensor"
        },
        "2": {
            "inputs": {},
            "class_type": "LoadTensor"
        },
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
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": "beautiful landscape, high quality",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": "low quality, blurry",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        }
    }
    
    async with ComfyStreamTrickleExample() as client:
        try:
            # Start the stream
            logger.info(f"Starting stream {request_id}...")
            success = await client.start_stream(
                request_id=request_id,
                subscribe_url=subscribe_url,
                publish_url=publish_url,
                width=512,
                height=512,
                prompt_dict=workflow
            )
            
            if not success:
                logger.error("Failed to start stream")
                return
            
            # Monitor the stream for a while
            logger.info("Monitoring stream status...")
            for i in range(10):
                await asyncio.sleep(2)
                
                status = await client.get_stream_status(request_id)
                if status:
                    logger.info(f"Stream status: running={status.get('running', False)}, "
                              f"frames={status.get('frame_count', 0)}")
                else:
                    logger.warning("Could not get stream status")
                    break
            
            # List all streams
            logger.info("Listing all streams...")
            streams = await client.list_streams()
            if streams:
                logger.info(f"Active streams: {list(streams.get('streams', {}).keys())}")
            
            # Stop the stream
            logger.info(f"Stopping stream {request_id}...")
            success = await client.stop_stream(request_id)
            
            if success:
                logger.info("✅ Example completed successfully!")
            else:
                logger.error("❌ Failed to stop stream properly")
                
        except Exception as e:
            logger.error(f"❌ Example failed: {e}")
            import traceback
            traceback.print_exc()

async def test_server_health():
    """Test if the ComfyStream server is running and healthy."""
    
    logger.info("🔍 Checking server health...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8889/health") as response:
                if response.status == 200:
                    logger.info("✅ ComfyStream server is healthy")
                    return True
                else:
                    logger.error(f"❌ Server returned status {response.status}")
                    return False
    except aiohttp.ClientError as e:
        logger.error(f"❌ Cannot connect to server: {e}")
        logger.info("💡 Make sure ComfyStream server is running on port 8889")
        logger.info("   Start with: python server/app.py --workspace /path/to/comfy --port 8889")
        return False

def main():
    """Main function."""
    print("🎬 ComfyStream + Trickle Pipeline Example")
    print("=" * 50)
    
    async def run():
        # Check server health first
        if not await test_server_health():
            return 1
        
        # Run the example
        await example_usage()
        return 0
    
    # Run the async example
    try:
        result = asyncio.run(run())
        return result
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Example failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
