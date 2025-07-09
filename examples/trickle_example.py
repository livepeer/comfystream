#!/usr/bin/env python3
"""
Example script demonstrating ComfyStream Trickle Integration.

This script shows how to use the trickle API endpoints to process
video streams through ComfyStream using the exact curl format specified.
"""

import asyncio
import json
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_trickle_stream():
    """Example of using ComfyStream trickle integration."""
    
    # ComfyStream server URL
    base_url = "http://127.0.0.1:8889"
    
    # This matches exactly the curl request format from the requirements
    stream_request = {
        "subscribe_url": "http://192.168.10.61:3389/sample",
        "publish_url": "http://192.168.10.61:3389/sample-output",
        "gateway_request_id": "sample2",
        "params": {
            "width": 512,
            "height": 512,
            "prompt": json.dumps({
                "1": {
                    "inputs": {
                        "images": ["2", 0]
                    },
                    "class_type": "SaveTensor",
                    "_meta": {
                        "title": "SaveTensor"
                    }
                },
                "2": {
                    "inputs": {},
                    "class_type": "LoadTensor",
                    "_meta": {
                        "title": "LoadTensor"
                    }
                },
                "3": {
                    "inputs": {
                        "width": 900,
                        "height": 384,
                        "batch_size": 1
                    },
                    "class_type": "EmptyLatentImage",
                    "_meta": {
                        "title": "Empty Latent Image"
                    }
                }
            })
        }
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            # 1. Start the stream
            logger.info("🚀 Starting trickle stream...")
            async with session.post(
                f"{base_url}/stream/start",
                json=stream_request,
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"✅ Stream started: {result['message']}")
                    logger.info(f"📊 Config: {result['config']}")
                else:
                    error = await resp.text()
                    logger.error(f"❌ Failed to start stream: {error}")
                    return
            
            request_id = stream_request['gateway_request_id']
            
            # 2. Monitor stream for a few seconds
            logger.info("📺 Monitoring stream status...")
            for i in range(3):
                await asyncio.sleep(2)
                
                async with session.get(f"{base_url}/stream/{request_id}/status") as resp:
                    if resp.status == 200:
                        status = await resp.json()
                        logger.info(f"📈 Stream running: {status['frame_count']} frames processed")
                    else:
                        logger.warning("⚠️ Stream may have stopped")
                        break
            
            # 3. List all active streams
            logger.info("📋 Listing all active streams...")
            async with session.get(f"{base_url}/streams") as resp:
                if resp.status == 200:
                    streams = await resp.json()
                    logger.info(f"🔄 Active streams: {streams['count']}")
                    for stream_id, stream_info in streams['streams'].items():
                        logger.info(f"   - {stream_id}: {stream_info['frame_count']} frames")
            
            # 4. Stop the stream
            logger.info("🛑 Stopping stream...")
            async with session.post(f"{base_url}/stream/{request_id}/stop") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"✅ Stream stopped: {result['message']}")
                else:
                    error = await resp.text()
                    logger.error(f"❌ Failed to stop stream: {error}")
            
            # 5. Verify stream is cleaned up
            await asyncio.sleep(1)
            async with session.get(f"{base_url}/stream/{request_id}/status") as resp:
                if resp.status == 404:
                    logger.info("🧹 Stream successfully cleaned up")
                else:
                    logger.warning("⚠️ Stream may still be running")
                    
        except aiohttp.ClientConnectorError:
            logger.error("❌ Could not connect to ComfyStream server.")
            logger.error("💡 Make sure ComfyStream is running: python server/app.py --workspace=/workspace/ComfyUI")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")

async def demonstrate_curl_equivalent():
    """Show the curl equivalent of our API call."""
    
    logger.info("🌐 This is equivalent to the following curl command:")
    logger.info("")
    logger.info("curl --location 'http://127.0.0.1:8889/stream/start' \\")
    logger.info("--header 'Content-Type: application/json' \\")
    logger.info("--data '{")
    logger.info('    "subscribe_url": "http://192.168.10.61:3389/sample",')
    logger.info('    "publish_url": "http://192.168.10.61:3389/sample-output",')
    logger.info('    "gateway_request_id": "sample2",')
    logger.info('    "params": {')
    logger.info('        "width": 512,')
    logger.info('        "height": 512,')
    logger.info('        "prompt": "{\\"1\\":{\\"inputs\\":{\\"images\\":[\\"2\\",0]},\\"class_type\\":\\"SaveTensor\\"},...}"')
    logger.info('    }')
    logger.info("}'")
    logger.info("")

async def main():
    """Main example function."""
    
    print("🎬 ComfyStream Trickle Integration Example")
    print("=" * 50)
    
    await demonstrate_curl_equivalent()
    await example_trickle_stream()
    
    print("\n✨ Example completed!")
    print("\n📚 For more information, see TRICKLE_INTEGRATION.md")

if __name__ == "__main__":
    asyncio.run(main())
