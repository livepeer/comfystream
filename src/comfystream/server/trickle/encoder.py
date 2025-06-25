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
from typing import Optional, Dict, Any, List
from fractions import Fraction

logger = logging.getLogger(__name__)

class TrickleSegmentEncoder:
    """
    Encodes video frames into trickle segments with proper container format.
    
    Handles encoding of av.VideoFrame objects into bytes suitable for
    trickle streaming, with proper metadata and timebase preservation.
    Uses fragmented MP4 (fMP4) format for streaming compatibility.
    """
    
    def __init__(self, width: int = 512, height: int = 512, fps: int = 24, 
                 format: str = "mpegts", video_codec: str = "libx264"):
        self.width = width
        self.height = height
        self.fps = fps
        self.format = format  # Use MPEG-TS for better streaming compatibility
        self.video_codec = video_codec
        # Use consistent time base for both container and codec to fix DTS issues
        self.time_base = Fraction(1, self.fps)  # Consistent 1/24 time base for 24fps
        
        # Track segments for metadata AND wall-clock time-based timestamps
        self.segment_count = 0
        self.frames_per_second = fps
        self.start_time = None  # Wall-clock start time for consistent timing
        self.last_segment_end_time = 0.0  # End time of last segment for continuity
        self.last_segment_time = None  # Wall-clock time of last segment for duration calculation
    
    def _get_stream_fps(self, video_stream) -> float:
        """
        Get frame rate from video stream using PyAV version-compatible approach.
        
        Args:
            video_stream: PyAV video stream object
            
        Returns:
            Frame rate as float, defaults to self.fps if not found
        """
        try:
            # Try different attributes based on PyAV version
            
            # PyAV 8.x and newer: average_rate
            if hasattr(video_stream, 'average_rate') and video_stream.average_rate:
                return float(video_stream.average_rate)
            
            # PyAV 7.x and some versions: rate  
            if hasattr(video_stream, 'rate') and video_stream.rate:
                return float(video_stream.rate)
            
            # Fallback: try to get from time_base
            if hasattr(video_stream, 'time_base') and video_stream.time_base:
                # Convert time_base to fps (time_base is typically 1/fps)
                if video_stream.time_base.denominator > 0:
                    calculated_fps = float(video_stream.time_base.denominator) / float(video_stream.time_base.numerator)
                    if 1.0 <= calculated_fps <= 120.0:  # Reasonable fps range
                        return calculated_fps
            
            # Try codec context
            if hasattr(video_stream, 'codec_context'):
                codec_ctx = video_stream.codec_context
                
                # Try various codec context attributes
                if hasattr(codec_ctx, 'framerate') and codec_ctx.framerate:
                    return float(codec_ctx.framerate)
                
                if hasattr(codec_ctx, 'time_base') and codec_ctx.time_base:
                    if codec_ctx.time_base.denominator > 0:
                        calculated_fps = float(codec_ctx.time_base.denominator) / float(codec_ctx.time_base.numerator)
                        if 1.0 <= calculated_fps <= 120.0:
                            return calculated_fps
            
            # Last resort: check stream metadata
            if hasattr(video_stream, 'metadata'):
                metadata = video_stream.metadata
                if 'r_frame_rate' in metadata:
                    try:
                        rate_str = metadata['r_frame_rate']
                        if '/' in rate_str:
                            num, den = rate_str.split('/')
                            if int(den) > 0:
                                return float(num) / float(den)
                    except (ValueError, ZeroDivisionError):
                        pass
                        
                if 'avg_frame_rate' in metadata:
                    try:
                        rate_str = metadata['avg_frame_rate']
                        if '/' in rate_str:
                            num, den = rate_str.split('/')
                            if int(den) > 0:
                                return float(num) / float(den)
                    except (ValueError, ZeroDivisionError):
                        pass
            
            logger.warning(f"Could not determine frame rate from video stream, using encoder fps {self.fps}")
            return float(self.fps)
            
        except Exception as e:
            logger.warning(f"Error getting stream fps: {e}, using encoder fps {self.fps}")
            return float(self.fps)
        
    def encode_frame(self, frame: av.VideoFrame, frame_number: int = 0) -> bytes:
        """
        Encode a single video frame into segment bytes.
        
        Args:
            frame: av.VideoFrame to encode
            frame_number: Frame sequence number for PTS calculation
            
        Returns:
            Encoded segment data as bytes
        """
        return self.encode_frames_batch([frame], frame_number)
    
    def encode_frames_batch(self, frames: List[av.VideoFrame], start_frame_number: int = 0) -> bytes:
        """
        Encode multiple video frames into a single segment with variable frame count compensation.
        
        CRITICAL: Adjusts timing metadata to maintain consistent 3-second playback duration
        regardless of actual frame count (36-108 frames), preventing timing gaps.
        
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
            
            # WALL-CLOCK TIME BASED TIMESTAMPS: Use real elapsed time for DTS continuity
            actual_frame_count = len(frames)
            import time
            current_time = time.time()
            
            # Initialize start time on first segment
            if self.start_time is None:
                self.start_time = current_time
            
            # Always use the same time_base (1/fps) for all segments to prevent DTS corruption
            segment_time_base = self.time_base  # Consistent 1/24 for 24fps
            
            # Calculate segment timing based on actual processing time
            segment_start_time = self.last_segment_end_time
            
            # Calculate actual processing duration since last segment
            if self.segment_count > 0 and self.last_segment_time is not None:
                actual_processing_duration = current_time - self.last_segment_time
                # Use actual processing time as segment duration (stretch to fill processing time)
                segment_duration = max(3.0, actual_processing_duration)  # Minimum 3 seconds
            else:
                segment_duration = 3.0  # First segment uses default
            
            # Update last segment time for next calculation
            self.last_segment_time = current_time
            
            logger.debug(f"Segment timing: processing_duration={segment_duration:.1f}s, "
                        f"stream_start={segment_start_time:.1f}s")
            
            logger.debug(f"Segment {self.segment_count}: {actual_frame_count} frames, "
                        f"time_base={segment_time_base}, stream_time={segment_start_time:.3f}s")
                
            # Create output buffer for segment
            output_buffer = io.BytesIO()
            
            # Use MPEG-TS format for streaming compatibility (no special options needed)
            output_container = av.open(
                output_buffer,
                mode='w',
                format=self.format
            )
            
            # Add video stream with proper codec and settings
            video_stream = output_container.add_stream(self.video_codec, rate=self.fps)
            # Configure stream properties (using type: ignore for PyAV typing issues)
            video_stream.width = self.width  # type: ignore
            video_stream.height = self.height  # type: ignore
            video_stream.pix_fmt = 'yuv420p'  # type: ignore
            video_stream.time_base = segment_time_base  # type: ignore
            # Set frame rate using the rate property (compatible with all PyAV versions)
            video_stream.rate = self.fps  # type: ignore
            
            # Configure codec context with explicit parameters to fix ffplay issues
            codec_ctx = video_stream.codec_context
            # Set video-specific codec parameters (using type: ignore for PyAV typing issues)
            codec_ctx.width = self.width  # type: ignore
            codec_ctx.height = self.height  # type: ignore
            codec_ctx.pix_fmt = 'yuv420p'  # type: ignore
            codec_ctx.time_base = segment_time_base  # Use segment-specific time base for consistent duration
            
            # Set H.264 specific parameters for streaming compatibility - FIXED for consistency
            codec_ctx.options = {
                'preset': 'fast',       # Changed from 'ultrafast' for better consistency
                'tune': 'zerolatency',
                'profile': 'baseline',  # Baseline profile for maximum compatibility
                'level': '3.1',         # Level 3.1 for 512x512 resolution
                'g': str(self.fps),     # GOP size = fps for 1 second keyframe interval
                'keyint_max': str(self.fps),  # Maximum keyframe interval
                'bf': '0',              # No B-frames for low latency
                'refs': '1',            # Single reference frame
                'sc_threshold': '0',    # Disable scene change detection
                'keyint_min': str(self.fps),  # Min keyframe interval
                'flags': '+cgop',       # Closed GOP
                'force_key_frames': f'expr:gte(t,n_forced*3)',  # Force keyframes every 3 seconds
                'pix_fmt': 'yuv420p',   # Explicit pixel format
                'r': str(self.fps),     # Explicit frame rate for proper detection
                # Additional stability options for consistency
                'threads': '1',         # Single thread for consistent timing
                'slices': '1',          # Single slice for minimal overhead
            }
            
            logger.debug(f"Encoder setup: {self.width}x{self.height}@{self.fps}fps, "
                        f"time_base={self.time_base}, codec={self.video_codec}")
            
            # CRITICAL FIX: Use per-segment timestamps for consistent 3-second duration
            segment_duration_frames = len(frames)
            
            logger.debug(f"Segment {self.segment_count}: {segment_duration_frames} frames")
            
            # Encode all frames with STREAM TIMELINE based timestamps
            for i, frame in enumerate(frames):
                
                # Ensure frame has proper format and dimensions
                if (frame.format.name != 'yuv420p' or 
                    frame.width != self.width or 
                    frame.height != self.height):
                    frame = frame.reformat(
                        format='yuv420p',
                        width=self.width,
                        height=self.height
                    )
                
                # STREAM TIMELINE TIMESTAMPS: Calculate frame time in continuous stream timeline
                frame_time_in_segment = i * segment_time_base  # Time within this segment
                absolute_stream_time = segment_start_time + float(frame_time_in_segment)  # Absolute time in stream
                frame_pts = int(absolute_stream_time / float(segment_time_base))  # Convert back to PTS units
                
                frame.pts = frame_pts  # Stream timeline based PTS
                frame.time_base = segment_time_base  # Consistent time_base for all segments
                
                if i < 3:  # Log first few frames for debugging
                    logger.debug(f"Frame {i}: pts={frame_pts}, stream_time={absolute_stream_time:.3f}s, "
                               f"segment_start={segment_start_time:.3f}s")
                
                # Encode the frame
                try:
                    packets = video_stream.encode(frame)  # type: ignore
                    for packet in packets:
                        # Ensure packet has proper timestamp continuity
                        if packet.dts is None:
                            packet.dts = packet.pts
                        output_container.mux(packet)
                except Exception as e:
                    logger.error(f"Error encoding frame {i}: {e}")
                    continue
            
            # Update last segment end time for next segment's continuity (using actual duration)
            self.last_segment_end_time = segment_start_time + segment_duration
            
            logger.debug(f"Stream timeline: segment {self.segment_count} spans {segment_start_time:.1f}s - {self.last_segment_end_time:.1f}s (duration: {segment_duration:.1f}s)")
            
            # Flush the encoder to get remaining packets
            try:
                flush_packets = video_stream.encode()  # type: ignore
                for packet in flush_packets:
                    # Ensure flushed packets have proper timestamp
                    if packet.dts is None:
                        packet.dts = packet.pts
                    output_container.mux(packet)
                logger.debug(f"Flushed {len(flush_packets)} packets from encoder")
            except Exception as e:
                logger.error(f"Error flushing encoder: {e}")
            
            # Finalize container
            try:
                output_container.close()
            except Exception as e:
                logger.error(f"Error closing container: {e}")
            
            # Get encoded data
            segment_data = output_buffer.getvalue()
            output_buffer.close()
            
            if segment_data:
                # Calculate actual duration based on frame count and time_base
                actual_duration = (len(frames) - 1) * float(segment_time_base) if len(frames) > 1 else 0.0
                logger.debug(f"Encoded {len(frames)} frames to {len(segment_data)} bytes "
                               f"(segment {self.segment_count}, duration: {actual_duration:.2f}s, time_base: {segment_time_base})")
                
                # Validate the encoded segment for MPEG-TS structure
                if self._validate_segment(segment_data):
                    # Increment segment counter after successful encoding
                    self.segment_count += 1
                    return segment_data
                else:
                    logger.error("Generated segment failed validation")
                    return b""
            else:
                logger.error("No data generated from encoding")
                return b""
            
        except Exception as e:
            logger.error(f"Error encoding frame batch: {e}", exc_info=True)
            return b""
    
    def _validate_segment(self, segment_data: bytes) -> bool:
        """
        Validate that the encoded segment is properly formatted MPEG-TS.
        
        Args:
            segment_data: Encoded segment bytes to validate
            
        Returns:
            True if segment is valid, False otherwise
        """
        try:
            if not segment_data or len(segment_data) < 188:  # Minimum TS packet size
                logger.warning("Segment too small for MPEG-TS validation")
                return False
            
            # Check for MPEG-TS sync bytes (0x47) at proper intervals
            sync_bytes_found = 0
            for i in range(0, min(len(segment_data), 1880), 188):  # Check first 10 packets
                if i < len(segment_data) and segment_data[i] == 0x47:
                    sync_bytes_found += 1
            
            if sync_bytes_found == 0:
                logger.error("No MPEG-TS sync bytes found in segment")
                return False
            
            # Try to open with av to verify it's readable and has valid timestamps
            try:
                input_buffer = io.BytesIO(segment_data)
                container = av.open(input_buffer, mode='r')
                
                if not container.streams.video:
                    logger.error("No video stream found in MPEG-TS segment")
                    container.close()
                    input_buffer.close()
                    return False
                
                # Check video stream parameters
                video_stream = container.streams.video[0]
                if (video_stream.width != self.width or 
                    video_stream.height != self.height):
                    logger.error(f"Video stream dimensions mismatch: "
                               f"expected {self.width}x{self.height}, "
                               f"got {video_stream.width}x{video_stream.height}")
                    container.close()
                    input_buffer.close()
                    return False
                
                # Verify frame rate is properly detected using robust PyAV compatibility
                detected_fps = self._get_stream_fps(video_stream)
                if abs(detected_fps - self.fps) > 1.0:  # Allow 1 FPS tolerance
                    logger.warning(f"Frame rate mismatch: expected {self.fps}, detected {detected_fps}")
                
                # Check that we can decode at least one frame to verify timestamps
                frame_count = 0
                for packet in container.demux(video_stream):
                    for frame in packet.decode():
                        frame_count += 1
                        if frame_count >= 1:  # Just check first frame
                            if frame.pts is None:
                                logger.warning("Frame has no PTS timestamp")
                            break
                    if frame_count >= 1:
                        break
                
                container.close()
                input_buffer.close()
                
                logger.debug(f"MPEG-TS segment validation passed: {sync_bytes_found} sync bytes, {frame_count} frames, {detected_fps} fps")
                return True
                
            except Exception as e:
                logger.error(f"MPEG-TS segment validation failed with av error: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error validating MPEG-TS segment: {e}")
            return False

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
                                            'fps': TrickleMetadataExtractor._get_stream_fps_static(video_stream),
                    'time_base': [video_stream.time_base.numerator, video_stream.time_base.denominator] if video_stream.time_base else [1, 24],
                    'duration': float(container.duration / av.time_base) if container.duration else None,
                    'frame_count': video_stream.frames if hasattr(video_stream, 'frames') else None,
                    'pixel_format': video_stream.codec_context.pix_fmt
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
            'Content-Type': 'video/mp2t'  # MPEG-TS format
        }
        
        # Add video-specific headers
        if 'width' in metadata and 'height' in metadata:
            headers['Lp-Video-Resolution'] = f"{metadata['width']}x{metadata['height']}"
        
        if 'fps' in metadata:
            headers['Lp-Video-FPS'] = str(metadata['fps'])
        
        if 'video_codec' in metadata:
            headers['Lp-Video-Codec'] = metadata['video_codec']
        
        if 'pixel_format' in metadata:
            headers['Lp-Video-PixelFormat'] = metadata['pixel_format']
        
        # Add audio-specific headers if present
        if 'audio_codec' in metadata:
            headers['Lp-Audio-Codec'] = metadata['audio_codec']
            
        if 'sample_rate' in metadata:
            headers['Lp-Audio-SampleRate'] = str(metadata['sample_rate'])
        
        return headers
    
    @staticmethod
    def _get_stream_fps_static(video_stream) -> float:
        """
        Static version of frame rate detection for use in metadata extraction.
        
        Args:
            video_stream: PyAV video stream object
            
        Returns:
            Frame rate as float, defaults to 24.0 if not found
        """
        try:
            # Try different attributes based on PyAV version
            
            # PyAV 8.x and newer: average_rate
            if hasattr(video_stream, 'average_rate') and video_stream.average_rate:
                return float(video_stream.average_rate)
            
            # PyAV 7.x and some versions: rate  
            if hasattr(video_stream, 'rate') and video_stream.rate:
                return float(video_stream.rate)
            
            # Fallback: try to get from time_base
            if hasattr(video_stream, 'time_base') and video_stream.time_base:
                # Convert time_base to fps (time_base is typically 1/fps)
                if video_stream.time_base.denominator > 0:
                    calculated_fps = float(video_stream.time_base.denominator) / float(video_stream.time_base.numerator)
                    if 1.0 <= calculated_fps <= 120.0:  # Reasonable fps range
                        return calculated_fps
            
            # Try codec context
            if hasattr(video_stream, 'codec_context'):
                codec_ctx = video_stream.codec_context
                
                # Try various codec context attributes
                if hasattr(codec_ctx, 'framerate') and codec_ctx.framerate:
                    return float(codec_ctx.framerate)
                
                if hasattr(codec_ctx, 'time_base') and codec_ctx.time_base:
                    if codec_ctx.time_base.denominator > 0:
                        calculated_fps = float(codec_ctx.time_base.denominator) / float(codec_ctx.time_base.numerator)
                        if 1.0 <= calculated_fps <= 120.0:
                            return calculated_fps
            
            # Last resort: check stream metadata
            if hasattr(video_stream, 'metadata'):
                metadata = video_stream.metadata
                if 'r_frame_rate' in metadata:
                    try:
                        rate_str = metadata['r_frame_rate']
                        if '/' in rate_str:
                            num, den = rate_str.split('/')
                            if int(den) > 0:
                                return float(num) / float(den)
                    except (ValueError, ZeroDivisionError):
                        pass
                        
                if 'avg_frame_rate' in metadata:
                    try:
                        rate_str = metadata['avg_frame_rate']
                        if '/' in rate_str:
                            num, den = rate_str.split('/')
                            if int(den) > 0:
                                return float(num) / float(den)
                    except (ValueError, ZeroDivisionError):
                        pass
            
            logger.warning(f"Could not determine frame rate from video stream, using default 24.0 fps")
            return 24.0
            
        except Exception as e:
            logger.warning(f"Error getting stream fps: {e}, using default 24.0")
            return 24.0 