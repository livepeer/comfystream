#!/usr/bin/env python3
"""
Test script for ComfyStream BYOC server

This script demonstrates how to test the BYOC server endpoints manually.
"""

import asyncio
import json
import base64
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_health_check():
    """Test the health check endpoint"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get('http://localhost:5000/health') as response:
                data = await response.json()
                logger.info(f"Health check: {data}")
                return response.status == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

async def test_text_reversal():
    """Test the text reversal capability (BYOC example)"""
    # Create the Livepeer header as specified in BYOC docs
    header_data = {
        "request": "{\"run\":\"echo\"}", 
        "capability": "text-reversal", 
        "timeout_seconds": 30
    }
    
    # Encode as base64
    header_json = json.dumps(header_data)
    livepeer_header = base64.b64encode(header_json.encode()).decode()
    
    # Request payload
    payload = {"text": "Hello, ComfyStream BYOC!"}
    
    headers = {
        'Content-Type': 'application/json',
        'Livepeer': livepeer_header
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                'http://localhost:5000/process/request/text-reversal',
                json=payload,
                headers=headers
            ) as response:
                data = await response.json()
                logger.info(f"Text reversal response: {data}")
                
                expected_reversed = payload["text"][::-1]
                if data.get("reversed") == expected_reversed:
                    logger.info("✓ Text reversal test passed")
                    return True
                else:
                    logger.error("✗ Text reversal test failed")
                    return False
                    
        except Exception as e:
            logger.error(f"Text reversal test failed: {e}")
            return False

async def test_stream_management():
    """Test stream start/stop/list functionality"""
    
    # Test data for starting a stream
    stream_data = {
        "prompts": [{"prompt": "a beautiful landscape"}],
        "stream_url": "http://test-trickle-endpoint/stream",
        "width": 512,
        "height": 512
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            # Start a stream
            async with session.post(
                'http://localhost:5000/stream/start',
                json=stream_data
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to start stream: {response.status}")
                    return False
                    
                start_data = await response.json()
                manifest_id = start_data.get("manifest_id")
                
                if not manifest_id:
                    logger.error("No manifest ID returned when starting stream")
                    return False
                    
                logger.info(f"✓ Stream started with manifest ID: {manifest_id}")
            
            # List streams
            async with session.get('http://localhost:5000/streams') as response:
                if response.status != 200:
                    logger.error(f"Failed to list streams: {response.status}")
                    return False
                    
                list_data = await response.json()
                logger.info(f"✓ Stream list: {list_data.get('count')} active streams")
            
            # Get stream status
            async with session.get(f'http://localhost:5000/stream/{manifest_id}/status') as response:
                if response.status != 200:
                    logger.error(f"Failed to get stream status: {response.status}")
                    return False
                    
                status_data = await response.json()
                logger.info(f"✓ Stream status: {status_data.get('stream', {}).get('status')}")
            
            # Stop the stream
            async with session.delete(f'http://localhost:5000/stream/{manifest_id}') as response:
                if response.status != 200:
                    logger.error(f"Failed to stop stream: {response.status}")
                    return False
                    
                stop_data = await response.json()
                logger.info(f"✓ Stream stopped: {stop_data.get('message')}")
                
            return True
            
        except Exception as e:
            logger.error(f"Stream management test failed: {e}")
            return False

async def main():
    """Run all tests"""
    logger.info("Starting ComfyStream BYOC server tests...")
    logger.info("Make sure the BYOC server is running on localhost:5000")
    logger.info("Start it with: python example_byoc_server.py --workspace /path/to/workspace")
    logger.info("")
    
    # Give user time to start server if needed
    await asyncio.sleep(2)
    
    tests = [
        ("Health Check", test_health_check),
        ("Text Reversal", test_text_reversal),
        ("Stream Management", test_stream_management),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} test PASSED")
            else:
                logger.error(f"✗ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} test ERROR: {e}")
            results.append((test_name, False))
        
        logger.info("")
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    logger.info("=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name}: {status}")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"Test error: {e}")
        exit(1) 