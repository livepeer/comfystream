#!/usr/bin/env python3
"""
Simple WHIP client example for ComfyStream.

This example demonstrates how to use the WHIP (WebRTC-HTTP Ingestion Protocol)
endpoint to ingest media streams to ComfyStream using standard HTTP requests.
"""

import asyncio
import json
import logging
import requests
import os
from typing import Optional
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer
from av.video.frame import VideoFrame
from fractions import Fraction
from comfystream.utils import DEFAULT_PROMPT
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageStreamTrack(MediaStreamTrack):
    """
    A video track that streams a single image frame endlessly.
    """
    kind = "video"

    def __init__(self, image_path: str, fps: int = 30):
        super().__init__()
        self.image_path = image_path
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.start_time = time.time()
        self.frame_count = 0
        self._frame = None
        
        # Load and prepare the image
        self._load_image()

    def _load_image(self):
        """Load the image and convert to VideoFrame."""
        try:
            import cv2
            import numpy as np
            
            # Load image using OpenCV
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError(f"Could not load image: {self.image_path}")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create VideoFrame from numpy array
            self._frame = VideoFrame.from_ndarray(img_rgb, format="rgb24")
            
            logger.info(f"Loaded image: {self.image_path} ({img_rgb.shape[1]}x{img_rgb.shape[0]})")
            
        except ImportError:
            logger.error("OpenCV not installed. Install with: pip install opencv-python")
            raise
        except Exception as e:
            logger.error(f"Error loading image {self.image_path}: {e}")
            raise

    async def recv(self):
        """Return the same frame repeatedly at the specified fps."""
        if self._frame is None:
            raise Exception("No frame available")
        
        # Calculate timing for consistent framerate
        expected_time = self.start_time + (self.frame_count * self.frame_time)
        current_time = time.time()
        
        if current_time < expected_time:
            await asyncio.sleep(expected_time - current_time)
        
        self.frame_count += 1
        
        # Set the frame timestamp
        pts = int(self.frame_count * (1 / self.frame_time) * 1000)  # milliseconds
        self._frame.pts = pts
        self._frame.time_base = Fraction(1, 1000)
        
        return self._frame


class WHIPClient:
    """Simple WHIP client implementation."""
    
    def __init__(self, whip_url: str, prompts=None):
        self.whip_url = whip_url
        self.prompts = prompts or []
        self.pc = None
        self.resource_url = None
        
    def _is_image_file(self, path: str) -> bool:
        """Check if the path is an image file."""
        if not os.path.isfile(path):
            return False
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        _, ext = os.path.splitext(path.lower())
        return ext in image_extensions
        
    def _is_video_file(self, path: str) -> bool:
        """Check if the path is a video file."""
        if not os.path.isfile(path):
            return False
            
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        _, ext = os.path.splitext(path.lower())
        return ext in video_extensions

    async def publish(self, media_source: Optional[str] = None):
        """Publish media to WHIP endpoint."""
        try:
            # Create peer connection
            self.pc = RTCPeerConnection()
            
            # Add media tracks if source provided
            if media_source is not None:
                if self._is_image_file(media_source):
                    # Stream single image endlessly
                    logger.info(f"Streaming image: {media_source}")
                    video_track = ImageStreamTrack(media_source, fps=30)
                    self.pc.addTrack(video_track)
                    logger.info("Added image stream track")
                    
                elif media_source in ["testsrc", "0"] or self._is_video_file(media_source):
                    # Use MediaPlayer for video files, webcam, or test sources
                    player = MediaPlayer(media_source)
                    if hasattr(player, 'video') and player.video:
                        self.pc.addTrack(player.video)
                        logger.info(f"Added video track from: {media_source}")
                    if hasattr(player, 'audio') and player.audio:
                        self.pc.addTrack(player.audio)
                        logger.info(f"Added audio track from: {media_source}")
                        
                else:
                    logger.warning(f"Unknown media source type: {media_source}")
                    logger.info("Supported sources:")
                    logger.info("  - Image files: .jpg, .png, .bmp, etc.")
                    logger.info("  - Video files: .mp4, .avi, .mov, etc.")
                    logger.info("  - 'testsrc' for synthetic test pattern")
                    logger.info("  - '0' for webcam")
            else:
                logger.info("No media source provided - signaling only")
            
            # Create offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            # Prepare WHIP request
            headers = {'Content-Type': 'application/sdp'}
            
            # Add query parameters for configuration
            params = {}
            if self.prompts:
                params['prompts'] = json.dumps(self.prompts)
            
            # Send WHIP request
            logger.info(f"Sending WHIP request to {self.whip_url}")
            response = requests.post(
                self.whip_url,
                data=self.pc.localDescription.sdp,
                headers=headers,
                params=params
            )
            
            if response.status_code == 201:
                # Success - get resource URL and SDP answer
                self.resource_url = response.headers.get('Location')
                answer_sdp = response.text
                
                logger.info(f"WHIP session created: {self.resource_url}")
                
                # Set remote description
                answer = RTCSessionDescription(sdp=answer_sdp, type='answer')
                await self.pc.setRemoteDescription(answer)
                
                # Log ICE server configuration if provided
                if 'Link' in response.headers:
                    logger.info(f"ICE servers provided: {response.headers['Link']}")
                
                return True
            else:
                logger.error(f"WHIP request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during WHIP publish: {e}")
            return False
    
    async def unpublish(self):
        """Terminate WHIP session."""
        if self.resource_url:
            try:
                logger.info(f"Terminating WHIP session: {self.resource_url}")
                response = requests.delete(self.resource_url)
                if response.status_code == 200:
                    logger.info("WHIP session terminated successfully")
                else:
                    logger.warning(f"WHIP termination returned: {response.status_code}")
            except Exception as e:
                logger.error(f"Error terminating WHIP session: {e}")
        
        if self.pc:
            await self.pc.close()
            logger.info("Peer connection closed")


async def main():
    """Main example function."""
    # Configure WHIP endpoint
    whip_url = "http://localhost:8889/whip"

    # Example ComfyUI prompts
    example_prompts = [json.loads(DEFAULT_PROMPT)]

    # Create WHIP client
    client = WHIPClient(whip_url, example_prompts)

    try:
        # Option 1: Publish webcam (requires camera)
        # success = await client.publish("0")  # Camera device 0

        # Option 2: Publish file
        success = await client.publish("test/example-512x512.png")

        # Option 3: Test signaling only (no media)
        # success = await client.publish()

        if success:
            logger.info("WHIP publishing started successfully!")

            # Keep session active for 30 seconds
            logger.info("Keeping session active for 30 seconds...")
            await asyncio.sleep(30)
        else:
            logger.error("Failed to start WHIP publishing")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        await client.unpublish()


if __name__ == "__main__":
    print("ComfyStream WHIP Client Example")
    print("===============================")
    print()
    print("This example demonstrates how to use WHIP to ingest streams to ComfyStream.")
    print("Make sure ComfyStream server is running at http://localhost:8889")
    print()
    
    asyncio.run(main()) 
