#!/usr/bin/env python3
"""
WHIP/WHEP Integration Test for go-livepeer + ComfyStream

This script demonstrates the complete SDP exchange flow:
1. Client -> go-livepeer Gateway -> go-livepeer Orchestrator -> ComfyStream BYOC
2. ComfyStream processes and returns SDP answer back through the chain
3. Establishes WebRTC connections for real-time AI processing
"""

import asyncio
import json
import logging
import requests
import time
import base64
from typing import Optional
from aiortc import RTCPeerConnection, RTCSessionDescription
from comfystream.utils import DEFAULT_SD_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationTestClient:
    """Test client for WHIP/WHEP integration between go-livepeer and ComfyStream."""
    
    def __init__(self, 
                 gateway_url: str = "http://192.168.10.61:8937",
                 orchestrator_url: str = "https://192.168.10.61:8936",
                 comfystream_url: str = "http://localhost:8889"):
        self.gateway_url = gateway_url.rstrip('/')
        self.orchestrator_url = orchestrator_url.rstrip('/')
        self.comfystream_url = comfystream_url.rstrip('/')
        
    def create_job_request(self, capability: str, stream_id: str, sdp_offer: str, 
                          timeout: int = 30) -> dict:
        """Create a job request for go-livepeer."""
        
        # Create job request details
        job_request_details = {
            "start_stream": capability == "whip-ingest",
            "start_stream_output": capability == "whep-subscribe", 
            "stream_id": stream_id
        }
        
        # Create the main job request
        job_request = {
            "id": f"job_{int(time.time())}",
            "request": json.dumps(job_request_details),
            "parameters": json.dumps({
                "prompts": [json.loads(DEFAULT_SD_PROMPT)],
                "width": 512,
                "height": 512
            }),
            "capability": capability,
            "timeout_seconds": timeout
        }
        
        return job_request
    
    async def test_whip_integration(self) -> dict:
        """Test WHIP integration through go-livepeer to ComfyStream."""
        logger.info("🔄 Testing WHIP integration...")
        
        try:
            # Create WebRTC peer connection for WHIP
            pc = RTCPeerConnection()
            
            # Add video track (mock)
            from aiortc.contrib.media import MediaPlayer
            player = MediaPlayer('/dev/video0')  # or use a test file
            video_track = player.video
            if video_track:
                pc.addTrack(video_track)
            
            # Create offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            # Create job request
            stream_id = f"test_stream_{int(time.time())}"
            job_request = self.create_job_request("whip-ingest", stream_id, offer.sdp)
            
            # Encode job request for header
            job_header = base64.b64encode(json.dumps(job_request).encode()).decode()
            
            # Send request to go-livepeer Gateway/Orchestrator
            headers = {
                'Content-Type': 'application/sdp',
                'Livepeer': job_header
            }
            
            logger.info(f"📡 Sending WHIP request to {self.gateway_url}/process/request/whip-ingest")
            response = requests.post(
                f"{self.gateway_url}/process/request/whip-ingest",
                data=offer.sdp,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 201:
                # Success - parse SDP answer
                sdp_answer = response.text
                resource_url = response.headers.get('Location')
                
                logger.info(f"✅ WHIP session created: {resource_url}")
                
                # Set remote description
                answer = RTCSessionDescription(sdp=sdp_answer, type='answer')
                await pc.setRemoteDescription(answer)
                
                return {
                    "success": True,
                    "stream_id": stream_id,
                    "resource_url": resource_url,
                    "sdp_answer": sdp_answer,
                    "connection_state": pc.connectionState
                }
            else:
                logger.error(f"❌ WHIP request failed: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            logger.error(f"❌ WHIP integration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_whep_integration(self, stream_id: Optional[str] = None) -> dict:
        """Test WHEP integration through go-livepeer to ComfyStream."""
        logger.info("📺 Testing WHEP integration...")
        
        try:
            # Create WebRTC peer connection for WHEP
            pc = RTCPeerConnection()
            
            # Set up for receiving video and audio
            pc.addTransceiver("video", direction="recvonly")
            pc.addTransceiver("audio", direction="recvonly")
            
            # Track for received streams
            tracks_received = []
            
            @pc.on("track")
            def on_track(track):
                logger.info(f"📺 Received {track.kind} track from WHEP")
                tracks_received.append(track)
            
            # Create offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            # Create job request
            if not stream_id:
                stream_id = f"test_output_{int(time.time())}"
                
            job_request = self.create_job_request("whep-subscribe", stream_id, offer.sdp)
            
            # Encode job request for header
            job_header = base64.b64encode(json.dumps(job_request).encode()).decode()
            
            # Send request to go-livepeer Gateway/Orchestrator
            headers = {
                'Content-Type': 'application/sdp',
                'Livepeer': job_header
            }
            
            logger.info(f"📡 Sending WHEP request to {self.gateway_url}/process/request/whep-subscribe")
            response = requests.post(
                f"{self.gateway_url}/process/request/whep-subscribe",
                data=offer.sdp,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 201:
                # Success - parse SDP answer
                sdp_answer = response.text
                resource_url = response.headers.get('Location')
                
                logger.info(f"✅ WHEP session created: {resource_url}")
                
                # Set remote description
                answer = RTCSessionDescription(sdp=sdp_answer, type='answer')
                await pc.setRemoteDescription(answer)
                
                # Wait a bit for potential tracks
                await asyncio.sleep(5)
                
                return {
                    "success": True,
                    "stream_id": stream_id,
                    "resource_url": resource_url,
                    "sdp_answer": sdp_answer,
                    "tracks_received": len(tracks_received),
                    "connection_state": pc.connectionState
                }
            else:
                logger.error(f"❌ WHEP request failed: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            logger.error(f"❌ WHEP integration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_direct_byoc_endpoints(self) -> dict:
        """Test direct BYOC endpoints on ComfyStream."""
        logger.info("🔧 Testing direct BYOC endpoints...")
        
        results = {}
        
        # Test whip-ingest capability
        try:
            mock_sdp = self.generate_mock_sdp_offer()
            whip_data = {
                "sdp_offer": mock_sdp,
                "prompts": [{"text": "test prompt", "weight": 1.0}],
                "width": 512,
                "height": 512
            }
            
            response = requests.post(
                f"{self.comfystream_url}/process/request/whip-ingest",
                json=whip_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            results["whip_ingest"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else response.text
            }
            
        except Exception as e:
            results["whip_ingest"] = {"success": False, "error": str(e)}
        
        # Test whep-subscribe capability  
        try:
            mock_sdp = self.generate_mock_sdp_offer()
            whep_data = {
                "sdp_offer": mock_sdp
            }
            
            response = requests.post(
                f"{self.comfystream_url}/process/request/whep-subscribe",
                json=whep_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            results["whep_subscribe"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response": response.json() if response.status_code == 200 else response.text
            }
            
        except Exception as e:
            results["whep_subscribe"] = {"success": False, "error": str(e)}
        
        return results
    
    def generate_mock_sdp_offer(self) -> str:
        """Generate a mock SDP offer for testing."""
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
    
    def check_service_availability(self) -> dict:
        """Check if all services are available."""
        logger.info("🔍 Checking service availability...")
        
        services = {}
        
        # Check ComfyStream
        try:
            response = requests.get(f"{self.comfystream_url}/health", timeout=5)
            services["comfystream"] = {
                "available": response.status_code == 200,
                "url": self.comfystream_url
            }
        except:
            services["comfystream"] = {
                "available": False,
                "url": self.comfystream_url
            }
        
        # Check go-livepeer Gateway
        try:
            response = requests.get(f"{self.gateway_url}/health", timeout=5)
            services["gateway"] = {
                "available": response.status_code == 200,
                "url": self.gateway_url
            }
        except:
            services["gateway"] = {
                "available": False,
                "url": self.gateway_url
            }
        
        # Check go-livepeer Orchestrator  
        try:
            response = requests.get(f"{self.orchestrator_url}/health", timeout=5)
            services["orchestrator"] = {
                "available": response.status_code == 200,
                "url": self.orchestrator_url
            }
        except:
            services["orchestrator"] = {
                "available": False,
                "url": self.orchestrator_url
            }
        
        return services


async def run_integration_tests():
    """Run comprehensive integration tests."""
    print("🚀 WHIP/WHEP Integration Test Suite")
    print("=" * 50)
    
    client = IntegrationTestClient()
    
    # Check service availability
    print("\n1. Checking Service Availability")
    services = client.check_service_availability()
    for service, info in services.items():
        status = "✅ Available" if info["available"] else "❌ Unavailable"
        print(f"   {service}: {status} ({info['url']})")
    
    # Test direct BYOC endpoints
    print("\n2. Testing Direct BYOC Endpoints")
    byoc_results = client.test_direct_byoc_endpoints()
    for endpoint, result in byoc_results.items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"   {endpoint}: {status}")
        if not result["success"]:
            print(f"      Error: {result.get('error', 'Unknown error')}")
    
    # Test WHIP integration (commented out as it requires actual video device)
    print("\n3. WHIP Integration Test")
    print("   ⚠️  Skipped - requires video device (uncomment code to test)")
    # whip_result = await client.test_whip_integration()
    # print(f"   WHIP: {'✅ Success' if whip_result['success'] else '❌ Failed'}")
    
    # Test WHEP integration
    print("\n4. WHEP Integration Test") 
    print("   ⚠️  Skipped - requires active WHIP stream (uncomment code to test)")
    # whep_result = await client.test_whep_integration()
    # print(f"   WHEP: {'✅ Success' if whep_result['success'] else '❌ Failed'}")
    
    print("\n✨ Integration tests completed!")
    print("\n💡 To run full tests:")
    print("   1. Start ComfyStream: python -m comfystream.server.app")
    print("   2. Start go-livepeer orchestrator and gateway")
    print("   3. Register capabilities: python byoc/register_capability.py")
    print("   4. Uncomment the WHIP/WHEP test code above")


if __name__ == "__main__":
    asyncio.run(run_integration_tests()) 