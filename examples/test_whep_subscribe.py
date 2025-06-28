#!/usr/bin/env python3
"""
Enhanced WHEP subscription test script with processing readiness checking.

This script checks if processed streams are ready before attempting subscription,
providing better reliability and user feedback.
"""

import asyncio
import logging
import requests
import time
from aiortc import RTCPeerConnection, RTCSessionDescription

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_processing_readiness(base_url: str, max_wait: int = 30):
    """Check if ComfyStream processing is ready for WHEP subscription."""
    status_url = f"{base_url}/processing/status"
    
    logger.info("🔍 Checking processing readiness...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(status_url, timeout=5)
            if response.status_code == 200:
                status = response.json()
                
                logger.info(f"📊 Status: {status.get('message', 'Unknown')}")
                logger.info(f"   • WHIP sessions: {status.get('whip_sessions', 0)}")
                logger.info(f"   • Active pipelines: {status.get('active_pipelines', 0)}")
                logger.info(f"   • Frames available: {status.get('frames_available', False)}")
                logger.info(f"   • WHEP sessions: {status.get('whep_sessions', 0)}")
                
                if status.get('processing_ready', False):
                    logger.info("✅ Processing ready for WHEP subscription!")
                    return True
                elif status.get('whip_sessions', 0) == 0:
                    logger.warning("⚠️  No WHIP sessions active - start publishing first")
                    return False
                else:
                    logger.info("⏳ Processing warming up... waiting")
                    
            else:
                logger.warning(f"⚠️  Status check failed: {response.status_code}")
                
        except requests.RequestException as e:
            logger.warning(f"⚠️  Status check error: {e}")
        
        await asyncio.sleep(2)
    
    logger.warning(f"⏰ Timeout waiting for processing readiness after {max_wait}s")
    return False


async def test_whep_subscribe_with_status_check():
    """Enhanced WHEP subscription test with status checking."""
    base_url = "http://localhost:8889"
    whep_url = f"{base_url}/whep"
    
    # Step 1: Check processing readiness
    logger.info("🎯 Enhanced WHEP Subscription Test")
    logger.info("==================================")
    
    if not await check_processing_readiness(base_url):
        logger.error("❌ Processing not ready - test aborted")
        logger.info("💡 Try: python examples/whip_client_example.py")
        return False
    
    # Step 2: Proceed with WHEP subscription
    logger.info("\n🔄 Proceeding with WHEP subscription...")
    
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
        logger.info("⏳ Testing stream reception... (30 seconds)")
        
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


async def quick_status_check():
    """Quick status check without subscription."""
    base_url = "http://localhost:8889"
    
    logger.info("🔍 Quick Processing Status Check")
    logger.info("==============================")
    
    try:
        response = requests.get(f"{base_url}/processing/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            
            print(f"📊 Processing Status: {'✅ READY' if status.get('processing_ready') else '❌ NOT READY'}")
            print(f"📝 Message: {status.get('message', 'Unknown')}")
            print(f"🔄 WHIP Sessions: {status.get('whip_sessions', 0)}")
            print(f"⚙️  Active Pipelines: {status.get('active_pipelines', 0)}")
            print(f"🖼️  Frames Available: {status.get('frames_available', False)}")
            print(f"📡 WHEP Sessions: {status.get('whep_sessions', 0)}")
            
            return status.get('processing_ready', False)
        else:
            logger.error(f"❌ Status check failed: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"❌ Connection error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    print("Enhanced WHEP Subscription Test")
    print("==============================")
    print("This script checks processing readiness before WHEP subscription.")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        # Quick status check only
        ready = asyncio.run(quick_status_check())
        exit(0 if ready else 1)
    else:
        # Full test with subscription
        print("💡 Tip: Use 'python test_whep_subscribe.py status' for quick status check")
        print()
        
        success = asyncio.run(test_whep_subscribe_with_status_check())
        
        print()
        print("Test Result:", "✅ PASSED" if success else "❌ FAILED")
        
        if not success:
            print("💡 Troubleshooting:")
            print("   1. Ensure ComfyStream server is running: python -m comfystream.server")
            print("   2. Start WHIP publishing: python examples/whip_client_example.py")
            print("   3. Check status: python examples/test_whep_subscribe.py status")
        
        exit(0 if success else 1) 