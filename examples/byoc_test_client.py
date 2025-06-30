#!/usr/bin/env python3
"""
BYOC Test Client for ComfyStream

This script demonstrates how to use the BYOC process/request/{capability} endpoints
for video processing, WHIP ingest, and WHEP subscriptions.
"""

import asyncio
import json
import logging
import requests
import time
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BYOCTestClient:
    """Client for testing BYOC endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8889"):
        self.base_url = base_url.rstrip('/')
        
    def test_video_processing_capability(self, prompts: Optional[Dict] = None) -> Dict[str, Any]:
        """Test the comfystream-video capability."""
        logger.info("🎬 Testing comfystream-video capability...")
        
        # Prepare request data
        request_data = {
            "input": {
                "prompts": prompts or [
                    {
                        "text": "a beautiful sunset over mountains",
                        "weight": 1.0
                    }
                ],
                "width": 512,
                "height": 512
            },
            "workspace": "/workspace/ComfyUI"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/process/request/comfystream-video",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Video processing capability test successful!")
                logger.info(f"   Session ID: {result.get('session_id')}")
                logger.info(f"   Status: {result.get('status')}")
                return result
            else:
                logger.error(f"❌ Video processing test failed: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except Exception as e:
            logger.error(f"❌ Error testing video processing: {e}")
            return {"error": str(e)}
    
    def test_whip_ingest_capability(self, sdp_offer: str, prompts: Optional[Dict] = None) -> Dict[str, Any]:
        """Test the whip-ingest capability."""
        logger.info("📡 Testing whip-ingest capability...")
        
        # Prepare request data
        request_data = {
            "sdp_offer": sdp_offer,
            "prompts": prompts or [
                {
                    "text": "a beautiful sunset over mountains",
                    "weight": 1.0
                }
            ],
            "width": 512,
            "height": 512
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/process/request/whip-ingest",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ WHIP ingest capability test successful!")
                logger.info(f"   Session ID: {result.get('session_id')}")
                logger.info(f"   Status: {result.get('status')}")
                return result
            else:
                logger.error(f"❌ WHIP ingest test failed: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except Exception as e:
            logger.error(f"❌ Error testing WHIP ingest: {e}")
            return {"error": str(e)}
    
    def test_whep_subscribe_capability(self, sdp_offer: str) -> Dict[str, Any]:
        """Test the whep-subscribe capability."""
        logger.info("📺 Testing whep-subscribe capability...")
        
        # Prepare request data
        request_data = {
            "sdp_offer": sdp_offer
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/process/request/whep-subscribe",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ WHEP subscribe capability test successful!")
                logger.info(f"   Session ID: {result.get('session_id')}")
                logger.info(f"   Status: {result.get('status')}")
                return result
            else:
                logger.error(f"❌ WHEP subscribe test failed: {response.status_code} - {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except Exception as e:
            logger.error(f"❌ Error testing WHEP subscribe: {e}")
            return {"error": str(e)}
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get the status of a BYOC session."""
        try:
            response = requests.get(f"{self.base_url}/byoc/session/{session_id}/status")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.text, "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup_session(self, session_id: str) -> Dict[str, Any]:
        """Clean up a BYOC session."""
        try:
            response = requests.delete(f"{self.base_url}/byoc/session/{session_id}")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.text, "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e)}
    
    def get_byoc_stats(self) -> Dict[str, Any]:
        """Get BYOC statistics."""
        try:
            response = requests.get(f"{self.base_url}/byoc-stats")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.text, "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e)}


def generate_mock_sdp_offer() -> str:
    """Generate a mock SDP offer for testing purposes."""
    return """v=0
o=- 0 0 IN IP4 127.0.0.1
s=-
c=IN IP4 127.0.0.1
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 96
a=rtpmap:96 H264/90000
a=fmtp:96 profile-level-id=42e01f
a=sendonly
a=setup:actpass
a=mid:0
a=ice-ufrag:test
a=ice-pwd:test
m=audio 9 UDP/TLS/RTP/SAVPF 111
a=rtpmap:111 opus/48000/2
a=sendonly
a=setup:actpass
a=mid:1
a=ice-ufrag:test
a=ice-pwd:test"""


def main():
    """Main test function."""
    print("🚀 BYOC Test Client for ComfyStream")
    print("=" * 50)
    
    client = BYOCTestClient()
    
    # Test 1: Video Processing Capability
    print("\n1. Testing Video Processing Capability")
    video_result = client.test_video_processing_capability()
    if "session_id" in video_result:
        session_id = video_result["session_id"]
        
        # Check session status
        print(f"   Checking session status for {session_id[:8]}...")
        status = client.get_session_status(session_id)
        print(f"   Status: {status}")
        
        # Clean up session
        print(f"   Cleaning up session {session_id[:8]}...")
        cleanup_result = client.cleanup_session(session_id)
        print(f"   Cleanup result: {cleanup_result}")
    
    # Test 2: WHIP Ingest Capability (with mock SDP)
    print("\n2. Testing WHIP Ingest Capability")
    mock_sdp = generate_mock_sdp_offer()
    whip_result = client.test_whip_ingest_capability(mock_sdp)
    if "session_id" in whip_result:
        session_id = whip_result["session_id"]
        print(f"   WHIP session created: {session_id[:8]}...")
        
        # Note: In a real scenario, you would establish the WebRTC connection
        # using the returned SDP answer before cleaning up
        
        # Clean up session
        print(f"   Cleaning up WHIP session {session_id[:8]}...")
        cleanup_result = client.cleanup_session(session_id)
        print(f"   Cleanup result: {cleanup_result}")
    
    # Test 3: WHEP Subscribe Capability (with mock SDP)
    print("\n3. Testing WHEP Subscribe Capability")
    whep_result = client.test_whep_subscribe_capability(mock_sdp)
    if "session_id" in whep_result:
        session_id = whep_result["session_id"]
        print(f"   WHEP session created: {session_id[:8]}...")
        
        # Clean up session
        print(f"   Cleaning up WHEP session {session_id[:8]}...")
        cleanup_result = client.cleanup_session(session_id)
        print(f"   Cleanup result: {cleanup_result}")
    
    # Test 4: Get BYOC Statistics
    print("\n4. Getting BYOC Statistics")
    stats = client.get_byoc_stats()
    print(f"   Active sessions: {len(stats) if isinstance(stats, dict) else 'N/A'}")
    print(f"   Stats: {stats}")
    
    print("\n✨ BYOC tests completed!")


if __name__ == "__main__":
    main() 