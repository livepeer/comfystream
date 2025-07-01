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

from comfystream.utils import DEFAULT_SD_PROMPT

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
                    json.loads(DEFAULT_SD_PROMPT),
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
                json.loads(DEFAULT_SD_PROMPT)
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
o=- 123456789 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
a=msid-semantic: WMS
m=video 54400 UDP/TLS/RTP/SAVPF 96 97 98 99 100 101 127 124 125
c=IN IP4 127.0.0.1
a=rtcp:9 IN IP4 127.0.0.1
a=ice-ufrag:test123
a=ice-pwd:testpassword123
a=ice-options:trickle
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF
a=setup:actpass
a=mid:0
a=sendonly
a=rtcp-mux
a=rtcp-rsize
a=rtpmap:96 VP8/90000
a=rtcp-fb:96 goog-remb
a=rtcp-fb:96 transport-cc
a=rtcp-fb:96 ccm fir
a=rtcp-fb:96 nack
a=rtcp-fb:96 nack pli
a=rtpmap:97 rtx/90000
a=fmtp:97 apt=96
a=rtpmap:98 VP9/90000
a=rtcp-fb:98 goog-remb
a=rtcp-fb:98 transport-cc
a=rtcp-fb:98 ccm fir
a=rtcp-fb:98 nack
a=rtcp-fb:98 nack pli
a=rtpmap:99 rtx/90000
a=fmtp:99 apt=98
a=rtpmap:100 H264/90000
a=rtcp-fb:100 goog-remb
a=rtcp-fb:100 transport-cc
a=rtcp-fb:100 ccm fir
a=rtcp-fb:100 nack
a=rtcp-fb:100 nack pli
a=fmtp:100 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f
a=rtpmap:101 rtx/90000
a=fmtp:101 apt=100
a=rtpmap:127 red/90000
a=rtpmap:124 rtx/90000
a=fmtp:124 apt=127
a=rtpmap:125 ulpfec/90000
a=ssrc-group:FID 1234567890 1234567891
a=ssrc:1234567890 cname:test-stream
a=ssrc:1234567890 msid:test-stream video0
a=ssrc:1234567891 cname:test-stream
a=ssrc:1234567891 msid:test-stream video0
m=audio 54401 UDP/TLS/RTP/SAVPF 111 103 9 102 0 8 105 13 110 113 126
c=IN IP4 127.0.0.1
a=rtcp:9 IN IP4 127.0.0.1
a=ice-ufrag:test123
a=ice-pwd:testpassword123
a=ice-options:trickle
a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF
a=setup:actpass
a=mid:1
a=sendonly
a=rtcp-mux
a=rtcp-rsize
a=rtpmap:111 opus/48000/2
a=rtcp-fb:111 transport-cc
a=fmtp:111 minptime=10;useinbandfec=1
a=rtpmap:103 ISAC/16000
a=rtpmap:9 G722/8000
a=rtpmap:102 ILBC/8000
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=rtpmap:105 CN/16000
a=rtpmap:13 CN/8000
a=rtpmap:110 telephone-event/48000
a=rtpmap:113 telephone-event/16000
a=rtpmap:126 telephone-event/8000
a=ssrc:1234567892 cname:test-stream
a=ssrc:1234567892 msid:test-stream audio0"""


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