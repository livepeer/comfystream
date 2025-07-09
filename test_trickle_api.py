#!/usr/bin/env python3
"""
Test script for ComfyStream Trickle API integration.

This script tests the trickle API endpoints to ensure they work correctly
with the curl request format specified in the requirements.
"""

import asyncio
import json
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_trickle_api():
    """Test the trickle API endpoints."""
    
    base_url = "http://127.0.0.1:8889"
    
    # Test data matching the curl request format
    test_request = {
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
            # Test 1: Start a stream
            logger.info("Testing stream start...")
            async with session.post(
                f"{base_url}/stream/start",
                json=test_request,
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"Stream start successful: {result}")
                else:
                    error_text = await resp.text()
                    logger.error(f"Stream start failed ({resp.status}): {error_text}")
                    return
            
            request_id = test_request['gateway_request_id']
            
            # Test 2: Get stream status
            logger.info("Testing stream status...")
            async with session.get(f"{base_url}/stream/{request_id}/status") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"Stream status: {result}")
                else:
                    error_text = await resp.text()
                    logger.error(f"Stream status failed ({resp.status}): {error_text}")
            
            # Test 3: List all streams
            logger.info("Testing stream list...")
            async with session.get(f"{base_url}/streams") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"Stream list: {result}")
                else:
                    error_text = await resp.text()
                    logger.error(f"Stream list failed ({resp.status}): {error_text}")
            
            # Wait a bit to let the stream run
            await asyncio.sleep(2)
            
            # Test 4: Stop the stream
            logger.info("Testing stream stop...")
            async with session.post(f"{base_url}/stream/{request_id}/stop") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"Stream stop successful: {result}")
                else:
                    error_text = await resp.text()
                    logger.error(f"Stream stop failed ({resp.status}): {error_text}")
            
            # Test 5: Verify stream is stopped
            logger.info("Verifying stream is stopped...")
            async with session.get(f"{base_url}/stream/{request_id}/status") as resp:
                if resp.status == 404:
                    logger.info("Stream successfully stopped and removed")
                else:
                    result = await resp.json()
                    logger.warning(f"Stream still exists: {result}")
                    
        except aiohttp.ClientConnectorError:
            logger.error("Could not connect to ComfyStream server. Make sure it's running on port 8889.")
        except Exception as e:
            logger.error(f"Test failed with error: {e}")

async def test_health_check():
    """Test that the server is running."""
    base_url = "http://127.0.0.1:8889"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    logger.info("Server health check passed")
                    return True
                else:
                    logger.error(f"Health check failed: {resp.status}")
                    return False
    except aiohttp.ClientConnectorError:
        logger.error("Could not connect to server")
        return False

async def main():
    """Main test function."""
    logger.info("Starting ComfyStream Trickle API tests...")
    
    # First check if server is running
    if not await test_health_check():
        logger.error("Server is not running. Please start ComfyStream server first.")
        return
    
    # Run trickle API tests
    await test_trickle_api()
    
    logger.info("Tests completed.")

if __name__ == "__main__":
    asyncio.run(main())
