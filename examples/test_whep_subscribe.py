#!/usr/bin/env python3
"""
Simple WHEP subscription test script.

This is a minimal script to test WHEP subscription and track reception from ComfyStream.
"""

import asyncio
import logging
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_whep_subscribe():
    """Simple test of WHEP subscription."""
    whep_url = "http://localhost:8889/whep"
    
    # Create peer connection
    pc = RTCPeerConnection()
    
    # Track counter for testing
    tracks_received = []
    
    # Set up track handler
    @pc.on("track")
    def on_track(track):
        logger.info(f"✅ Received {track.kind} track from WHEP!")
        tracks_received.append(track)
        
        # Simple frame counter for testing
        frame_count = 0
        
        async def count_frames():
            nonlocal frame_count
            try:
                while True:
                    frame = await track.recv()
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        logger.info(f"📺 {track.kind}: received {frame_count} frames")
            except Exception as e:
                logger.info(f"📺 {track.kind} track ended after {frame_count} frames: {e}")
        
        # Start frame counting
        asyncio.create_task(count_frames())
    
    # Create transceivers for receiving
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")
    
    # Create offer
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    # Send WHEP request
    logger.info(f"🔄 Sending WHEP subscription request to {whep_url}")
    
    headers = {'Content-Type': 'application/sdp'}
    response = requests.post(
        whep_url,
        data=pc.localDescription.sdp,
        headers=headers
    )
    
    if response.status_code == 201:
        resource_url = response.headers.get('Location')
        logger.info(f"✅ WHEP subscription created: {resource_url}")
        
        # Set remote description
        answer = RTCSessionDescription(sdp=response.text, type='answer')
        await pc.setRemoteDescription(answer)
        
        # Wait for tracks and test for 30 seconds
        logger.info("⏳ Waiting for tracks... (will test for 30 seconds)")
        
        for i in range(30):
            await asyncio.sleep(1)
            if i == 5 and not tracks_received:
                logger.warning("⚠️  No tracks received yet after 5 seconds")
            elif i == 10 and tracks_received:
                logger.info(f"🎉 Successfully receiving {len(tracks_received)} track(s)!")
        
        # Cleanup
        logger.info("🧹 Cleaning up...")
        if resource_url:
            try:
                requests.delete(resource_url)
                logger.info("✅ WHEP session terminated")
            except:
                pass
        
        await pc.close()
        
        # Test results
        if tracks_received:
            logger.info(f"🎉 TEST PASSED: Received {len(tracks_received)} track(s)")
            return True
        else:
            logger.error("❌ TEST FAILED: No tracks received")
            return False
    
    else:
        logger.error(f"❌ WHEP request failed: {response.status_code} - {response.text}")
        return False


if __name__ == "__main__":
    print("Simple WHEP Subscription Test")
    print("============================")
    print("Testing WHEP subscription to ComfyStream processed streams...")
    print()
    
    success = asyncio.run(test_whep_subscribe())
    
    print()
    print("Test Result:", "✅ PASSED" if success else "❌ FAILED")
    exit(0 if success else 1) 