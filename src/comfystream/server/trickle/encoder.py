"""
Trickle Encoder Module

Provides frame encoding functionality for trickle streaming.
Encodes processed video frames into segments with proper metadata,
timebase, and container format for trickle publisher.
"""

import io
import av
import logging
import numpy as np
from typing import Optional, Dict, Any
from fractions import Fraction

logger = logging.getLogger(__name__)

class TrickleSegmentEncoder:
    """
    Encodes video frames into trickle segments with proper container format.
    
    Handles encoding of av.VideoFrame objects into bytes suitable for
    trickle streaming, with proper metadata and timebase preservation.
    """
    
    def __init__(self, width: int = 512, height: int = 512, fps: int = 30, 
                 format: str = "mp4", video_codec: str = "libx264"):
        self.width = width
        self.height = height
        self.fps = fps
        self.format = format
        self.video_codec = video_codec
        self.time_base = Fraction(1, fps)
        
    def encode_frame(self, frame: av.VideoFrame, frame_number: int = 0) -> bytes:
        """
        Encode a single video frame into segment bytes.
        
        Args:
            frame: av.VideoFrame to encode
            frame_number: Frame sequence number for PTS calculation
            
        Returns:
            Encoded segment data as bytes
        """
        try:
            # Create output buffer for segment
            output_buffer = io.BytesIO()
            
            # Create output container in memory
            output_container = av.open(output_buffer, mode='w', format=self.format)
            
            # Add video stream with proper codec and settings
            video_stream = output_container.add_stream(self.video_codec, rate=self.fps)
            video_stream.width = self.width
            video_stream.height = self.height
            video_stream.pix_fmt = 'yuv420p'
            video_stream.time_base = self.time_base
            
            # Set proper PTS and time_base on the frame
            frame.pts = frame_number
            frame.time_base = self.time_base
            
            # Ensure frame dimensions match stream settings
            if frame.width != self.width or frame.height != self.height:
                frame = frame.reformat(width=self.width, height=self.height)
            
            # Encode the frame
            for packet in video_stream.encode(frame):
                output_container.mux(packet)
            
            # Flush the encoder
            for packet in video_stream.encode():
                output_container.mux(packet)
            
            # Finalize container
            output_container.close()
            
            # Get encoded data
            segment_data = output_buffer.getvalue()
            output_buffer.close()
            
            logger.debug(f"Encoded frame {frame_number} to {len(segment_data)} bytes")
            return segment_data
            
        except Exception as e:
            logger.error(f"Error encoding frame {frame_number}: {e}")
            return b""
    
    def encode_frames_batch(self, frames: list[av.VideoFrame], start_frame_number: int = 0) -> bytes:
        """
        Encode multiple video frames into a single segment.
        
        Args:
            frames: List of av.VideoFrame objects to encode
            start_frame_number: Starting frame number for PTS calculation
            
        Returns:
            Encoded segment data containing all frames as bytes
        """
        try:
            if not frames:
                logger.warning("No frames provided for batch encoding")
                return b""
                
            # Create output buffer for segment
            output_buffer = io.BytesIO()
            
            # Create output container in memory
            output_container = av.open(output_buffer, mode='w', format=self.format)
            
            # Add video stream with proper codec and settings
            video_stream = output_container.add_stream(self.video_codec, rate=self.fps)
            video_stream.width = self.width
            video_stream.height = self.height
            video_stream.pix_fmt = 'yuv420p'
            video_stream.time_base = self.time_base
            
            # Encode all frames
            for i, frame in enumerate(frames):
                frame_number = start_frame_number + i
                
                # Set proper PTS and time_base on each frame
                frame.pts = frame_number
                frame.time_base = self.time_base
                
                # Ensure frame dimensions match stream settings
                if frame.width != self.width or frame.height != self.height:
                    frame = frame.reformat(width=self.width, height=self.height)
                
                # Encode the frame
                for packet in video_stream.encode(frame):
                    output_container.mux(packet)
            
            # Flush the encoder
            for packet in video_stream.encode():
                output_container.mux(packet)
            
            # Finalize container
            output_container.close()
            
            # Get encoded data
            segment_data = output_buffer.getvalue()
            output_buffer.close()
            
            logger.debug(f"Encoded {len(frames)} frames to {len(segment_data)} bytes (frames {start_frame_number}-{start_frame_number + len(frames) - 1})")
            return segment_data
            
        except Exception as e:
            logger.error(f"Error encoding frame batch: {e}")
            return b""

class TrickleMetadataExtractor:
    """
    Extracts and manages metadata for trickle segments.
    """
    
    @staticmethod
    def extract_segment_metadata(segment_data: bytes) -> Dict[str, Any]:
        """
        Extract metadata from trickle segment data.
        
        Args:
            segment_data: Raw segment bytes
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            if not segment_data:
                return {}
                
            # Create container from segment data
            input_buffer = io.BytesIO(segment_data)
            container = av.open(input_buffer, mode='r')
            
            metadata = {}
            
            # Extract video stream metadata if available
            if container.streams.video:
                video_stream = container.streams.video[0]
                metadata.update({
                    'video_codec': video_stream.codec_context.name,
                    'width': video_stream.width,
                    'height': video_stream.height,
                    'fps': float(video_stream.average_rate),
                    'time_base': [video_stream.time_base.numerator, video_stream.time_base.denominator],
                    'duration': float(container.duration / av.time_base) if container.duration else None,
                    'frame_count': video_stream.frames if hasattr(video_stream, 'frames') else None
                })
            
            # Extract audio stream metadata if available
            if container.streams.audio:
                audio_stream = container.streams.audio[0]
                metadata.update({
                    'audio_codec': audio_stream.codec_context.name,
                    'sample_rate': audio_stream.sample_rate,
                    'channels': audio_stream.channels,
                    'audio_layout': audio_stream.layout.name if audio_stream.layout else None
                })
            
            container.close()
            input_buffer.close()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from segment: {e}")
            return {}
    
    @staticmethod
    def create_segment_headers(metadata: Dict[str, Any], segment_index: int) -> Dict[str, str]:
        """
        Create HTTP headers for trickle segment based on metadata.
        
        Args:
            metadata: Segment metadata dictionary
            segment_index: Segment sequence number
            
        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            'Lp-Trickle-Seq': str(segment_index),
            'Content-Type': 'video/mp4'  # Default to mp4
        }
        
        # Add video-specific headers
        if 'width' in metadata and 'height' in metadata:
            headers['Lp-Video-Resolution'] = f"{metadata['width']}x{metadata['height']}"
        
        if 'fps' in metadata:
            headers['Lp-Video-FPS'] = str(metadata['fps'])
        
        if 'video_codec' in metadata:
            headers['Lp-Video-Codec'] = metadata['video_codec']
        
        # Add audio-specific headers if present
        if 'audio_codec' in metadata:
            headers['Lp-Audio-Codec'] = metadata['audio_codec']
            
        if 'sample_rate' in metadata:
            headers['Lp-Audio-SampleRate'] = str(metadata['sample_rate'])
        
        return headers 