#!/usr/bin/env python3
"""
Test script for ComfyStream Trickle Protocol Implementation
"""

import asyncio
import json
import time
import base64
import httpx
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:9876"
STREAM_ID = "test_stream_123"

async def test_webrtc_start():
    """Test WebRTC start (backward compatibility)"""
    print("Testing WebRTC start (backward compatibility)...")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/stream/start",
            content=STREAM_ID,
            headers={"Content-Type": "text/plain"}
        )
        
    print(f"WebRTC Start Response: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error: {response.text}")
    
    # Stop processing
    async with httpx.AsyncClient() as client:
        await client.post(f"{BASE_URL}/stream/stop")

async def test_trickle_start_without_input():
    """Test trickle start without input URL"""
    print("\nTesting Trickle start without input URL...")
    
    payload = {"stream_id": STREAM_ID}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/stream/start",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-Protocol": "trickle"
            }
        )
    
    print(f"Trickle Start Response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result}")
        
        # Check status
        await test_status()
        
    else:
        print(f"Error: {response.text}")
    
    # Stop processing
    async with httpx.AsyncClient() as client:
        await client.post(f"{BASE_URL}/stream/stop")

async def test_trickle_start_with_input():
    """Test trickle start with input URL"""
    print("\nTesting Trickle start with input URL...")
    
    payload = {
        "stream_id": STREAM_ID,
        "input_url": "http://example-source:8080"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/stream/start",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-Protocol": "trickle"
            }
        )
    
    print(f"Trickle Start with Input Response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result}")
        
        # Check status
        await test_status()
        
    else:
        print(f"Error: {response.text}")
    
    # Stop processing
    async with httpx.AsyncClient() as client:
        await client.post(f"{BASE_URL}/stream/stop")

async def test_status():
    """Test status endpoint"""
    print("\nTesting status endpoint...")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/stream/status")
    
    print(f"Status Response: {response.status_code}")
    if response.status_code == 200:
        status = response.json()
        print(f"Status: {json.dumps(status, indent=2)}")
    else:
        print(f"Error: {response.text}")

async def test_trickle_endpoints():
    """Test trickle-specific endpoints"""
    print("\nTesting trickle endpoints...")
    
    # Start trickle processing first
    payload = {"stream_id": STREAM_ID}
    
    async with httpx.AsyncClient() as client:
        start_response = await client.post(
            f"{BASE_URL}/stream/start",
            json=payload,
            headers={
                "Content-Type": "application/json", 
                "X-Protocol": "trickle"
            }
        )
    
    if start_response.status_code != 200:
        print(f"Failed to start trickle processing: {start_response.text}")
        return
    
    print("Trickle processing started successfully")
    
    # Test publish endpoint
    print("Testing publish endpoint...")
    
    # Create dummy frame data
    dummy_frame_data = {
        "frame_type": "video",
        "data": base64.b64encode(b"dummy_h264_frame_data").decode(),
        "timestamp": time.time(),
        "width": 512,
        "height": 512,
        "encoding": "h264",
        "pts": 1000
    }
    
    async with httpx.AsyncClient() as client:
        publish_response = await client.post(
            f"{BASE_URL}/trickle/publish",
            json=dummy_frame_data,
            headers={
                "Content-Type": "application/json",
                "X-Stream-Id": STREAM_ID
            }
        )
    
    print(f"Publish Response: {publish_response.status_code}")
    if publish_response.status_code in [200, 201]:
        print(f"Publish Result: {publish_response.json()}")
    else:
        print(f"Publish Error: {publish_response.text}")
    
    # Test control endpoint
    print("Testing control endpoint...")
    
    control_message = {
        "control_type": "resolution_change",
        "width": 1024,
        "height": 768
    }
    
    async with httpx.AsyncClient() as client:
        control_response = await client.post(
            f"{BASE_URL}/trickle/control",
            json=control_message,
            headers={
                "Content-Type": "application/json",
                "X-Stream-Id": STREAM_ID
            }
        )
    
    print(f"Control Response: {control_response.status_code}")
    if control_response.status_code in [200, 201]:
        print(f"Control Result: {control_response.json()}")
    else:
        print(f"Control Error: {control_response.text}")
    
    # Test subscribe endpoint (briefly)
    print("Testing subscribe endpoint...")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            async with client.stream(
                "GET",
                f"{BASE_URL}/trickle/subscribe",
                headers={"X-Stream-Id": STREAM_ID}
            ) as response:
                print(f"Subscribe Response: {response.status_code}")
                
                if response.status_code == 200:
                    print("Subscribe connected successfully")
                    # Read a few chunks
                    chunk_count = 0
                    async for chunk in response.aiter_bytes():
                        if chunk_count >= 2:  # Just test a couple chunks
                            break
                        print(f"Received chunk: {len(chunk)} bytes")
                        try:
                            message = json.loads(chunk.decode())
                            print(f"Message type: {message.get('type')}")
                        except:
                            print("Non-JSON chunk received")
                        chunk_count += 1
                else:
                    print(f"Subscribe Error: {response.text}")
                    
    except Exception as e:
        print(f"Subscribe test error (expected for timeout): {e}")
    
    # Stop processing
    async with httpx.AsyncClient() as client:
        await client.post(f"{BASE_URL}/stream/stop")
    print("Trickle processing stopped")

async def test_error_cases():
    """Test error cases"""
    print("\nTesting error cases...")
    
    # Test missing stream_id for trickle
    print("Testing missing stream_id for trickle...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/stream/start",
            json={"input_url": "http://example.com"},
            headers={
                "Content-Type": "application/json",
                "X-Protocol": "trickle"
            }
        )
    print(f"Missing stream_id response: {response.status_code} - {response.text}")
    
    # Test trickle endpoints without server initialization
    print("Testing trickle endpoints without initialization...")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/trickle/subscribe",
            headers={"X-Stream-Id": "nonexistent"}
        )
    print(f"Subscribe without init response: {response.status_code} - {response.text}")

async def main():
    """Run all tests"""
    print("ComfyStream Trickle Protocol Test Suite")
    print("=" * 50)
    
    # Test server availability
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print(f"Server not available at {BASE_URL}")
            return
        print(f"Server available: {response.json()}")
    except Exception as e:
        print(f"Server not available: {e}")
        return
    
    # Run tests
    await test_webrtc_start()
    await test_trickle_start_without_input()
    await test_trickle_start_with_input()
    await test_trickle_endpoints()
    await test_error_cases()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")

if __name__ == "__main__":
    asyncio.run(main()) 