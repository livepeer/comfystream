#!/usr/bin/env python3
"""
Simple WHEP client example for ComfyStream.

This example demonstrates how to use the WHEP (WebRTC-HTTP Egress Protocol)
endpoint to subscribe to processed media streams from ComfyStream using standard HTTP requests.
"""

import asyncio
import json
import logging
import requests
import sys
from typing import Optional
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder, MediaPlayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WHEPClient:
    """Simple WHEP client implementation for subscribing to streams."""
    
    def __init__(self, whep_url: str, stream_config: Optional[dict] = None):
        self.whep_url = whep_url
        self.stream_config = stream_config or {}
        self.pc = None
        self.resource_url = None
        self.output_file = None
        self.tracks_received = []
        
    async def subscribe(self, stream_id: Optional[str] = None, output_file: Optional[str] = None):
        """Subscribe to processed stream from WHEP endpoint."""
        try:
            self.output_file = output_file
            
            # Create peer connection
            self.pc = RTCPeerConnection()
            
            # Create transceiver for receiving video and audio
            video_transceiver = self.pc.addTransceiver("video", direction="recvonly")
            audio_transceiver = self.pc.addTransceiver("audio", direction="recvonly")
            
            # Set up track handlers
            @self.pc.on("track")
            def on_track(track):
                logger.info(f"Received {track.kind} track from WHEP")
                self.tracks_received.append(track)
                
                # Handle track ending
                @track.on("ended")
                async def on_ended():
                    logger.info(f"{track.kind} track ended")
            
            # Create offer for receiving
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            # Prepare WHEP request
            headers = {'Content-Type': 'application/sdp'}
            
            # Add query parameters for configuration
            params = {}
            if stream_id:
                params['streamId'] = stream_id
            if self.stream_config:
                params['config'] = json.dumps(self.stream_config)
            
            # Send WHEP request
            logger.info(f"Sending WHEP subscription request to {self.whep_url}")
            response = requests.post(
                self.whep_url,
                data=self.pc.localDescription.sdp,
                headers=headers,
                params=params
            )
            
            if response.status_code == 201:
                # Success - get resource URL and SDP answer
                self.resource_url = response.headers.get('Location')
                answer_sdp = response.text
                
                logger.info(f"WHEP subscription created: {self.resource_url}")
                
                # Set remote description
                answer = RTCSessionDescription(sdp=answer_sdp, type='answer')
                await self.pc.setRemoteDescription(answer)
                
                # Log ICE server configuration if provided
                if 'Link' in response.headers:
                    logger.info(f"ICE servers provided: {response.headers['Link']}")
                
                return True
            else:
                logger.error(f"WHEP request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during WHEP subscription: {e}")
            return False
    
    async def unsubscribe(self):
        """Terminate WHEP subscription session."""
        # Terminate WHEP session
        if self.resource_url:
            try:
                logger.info(f"Terminating WHEP subscription: {self.resource_url}")
                response = requests.delete(self.resource_url)
                if response.status_code == 200:
                    logger.info("WHEP subscription terminated successfully")
                else:
                    logger.warning(f"WHEP termination returned: {response.status_code}")
            except Exception as e:
                logger.error(f"Error terminating WHEP subscription: {e}")
        
        # Close peer connection
        if self.pc:
            await self.pc.close()
            logger.info("Peer connection closed")


class WHEPViewer:
    """WHEP client that displays the stream in real-time using OpenCV."""
    
    def __init__(self, whep_url: str, stream_config: Optional[dict] = None):
        self.whep_url = whep_url
        self.stream_config = stream_config or {}
        self.pc = None
        self.resource_url = None
        self.running = False
        
    async def view_stream(self, stream_id: Optional[str] = None):
        """Subscribe and display the stream in real-time."""
        try:
            import cv2
            import numpy as np
            from av import VideoFrame
        except ImportError:
            logger.error("OpenCV and/or PyAV not installed. Install with: pip install opencv-python PyAV")
            return False
            
        try:
            # Create peer connection
            self.pc = RTCPeerConnection()
            
            # Create transceiver for receiving video only
            video_transceiver = self.pc.addTransceiver("video", direction="recvonly")
            
            # Set up track handlers
            @self.pc.on("track")
            def on_track(track):
                if track.kind == "video":
                    logger.info("Received video track from WHEP - starting viewer")
                    
                    # Create task to handle video frames
                    asyncio.create_task(self.handle_video_track(track))
                
                @track.on("ended")
                async def on_ended():
                    logger.info("Video track ended")
                    self.running = False
            
            # Create offer for receiving video only
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            # Prepare WHEP request
            headers = {'Content-Type': 'application/sdp'}
            params = {}
            if stream_id:
                params['streamId'] = stream_id
            if self.stream_config:
                params['config'] = json.dumps(self.stream_config)
            
            # Send WHEP request
            logger.info(f"Sending WHEP viewer request to {self.whep_url}")
            response = requests.post(
                self.whep_url,
                data=self.pc.localDescription.sdp,
                headers=headers,
                params=params
            )
            
            if response.status_code == 201:
                self.resource_url = response.headers.get('Location')
                answer_sdp = response.text
                
                logger.info(f"WHEP viewer session created: {self.resource_url}")
                
                # Set remote description
                answer = RTCSessionDescription(sdp=answer_sdp, type='answer')
                await self.pc.setRemoteDescription(answer)
                
                self.running = True
                
                # Wait for viewer to complete
                while self.running:
                    await asyncio.sleep(0.1)
                
                return True
            else:
                logger.error(f"WHEP viewer request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during WHEP viewing: {e}")
            return False
    
    async def handle_video_track(self, track):
        """Handle incoming video frames and display them."""
        try:
            import cv2
            import numpy as np
            
            logger.info("Starting video display loop")
            
            while self.running:
                try:
                    frame = await track.recv()
                    
                    # Convert frame to numpy array
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Display frame
                    cv2.imshow('ComfyStream WHEP Viewer', img)
                    
                    # Check for quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        logger.info("User requested quit")
                        self.running = False
                        break
                        
                except Exception as e:
                    if "MediaStreamError" in str(type(e)):
                        logger.info("Media stream ended")
                        break
                    else:
                        logger.error(f"Error processing video frame: {e}")
                        break
            
            cv2.destroyAllWindows()
            logger.info("Video display stopped")
            
        except Exception as e:
            logger.error(f"Error in video track handler: {e}")
    
    async def stop_viewing(self):
        """Stop viewing and cleanup."""
        self.running = False
        
        if self.resource_url:
            try:
                response = requests.delete(self.resource_url)
                logger.info(f"WHEP viewer session terminated: {response.status_code}")
            except Exception as e:
                logger.error(f"Error terminating WHEP viewer session: {e}")
        
        if self.pc:
            await self.pc.close()


async def main():
    """Main example function."""
    # Configure WHEP endpoint
    whep_url = "http://localhost:8889/whep"

    # Example stream configuration for subscribing to specific processed streams
    example_stream_config = {
        "quality": "high",
        "format": "h264",
        "processing_type": "depth_control",  # Could specify which processing pipeline
        "resolution": "512x512"
    }

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "view"
    
    if mode == "record":
        # Example 1: Record processed stream (simplified without MediaRecorder complexities)
        client = WHEPClient(whep_url, example_stream_config)
        
        try:
            success = await client.subscribe(stream_id="default")
            
            if success:
                logger.info("WHEP subscription started! Receiving processed stream...")
                
                # Keep subscription active for 30 seconds
                logger.info("Receiving stream for 30 seconds...")
                await asyncio.sleep(30)
            else:
                logger.error("Failed to start WHEP subscription")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            await client.unsubscribe()
            
    elif mode == "view":
        # Example 2: View processed stream in real-time
        viewer = WHEPViewer(whep_url, example_stream_config)
        
        try:
            logger.info("Starting WHEP viewer (press 'q' or ESC to quit)")
            success = await viewer.view_stream(stream_id="default")
            
            if not success:
                logger.error("Failed to start WHEP viewer")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            await viewer.stop_viewing()
    
    else:
        print("Usage: python whep_client_example.py [record|view]")
        print("  record - Subscribe to processed stream (track reception)")
        print("  view   - View processed stream in real-time (default)")


if __name__ == "__main__":
    print("ComfyStream WHEP Client Example")
    print("===============================")
    print()
    print("This example demonstrates how to use WHEP to subscribe to processed streams from ComfyStream.")
    print("Make sure ComfyStream server is running at http://localhost:8889")
    print("and that there's an active WHIP stream being processed.")
    print()
    
    asyncio.run(main()) 