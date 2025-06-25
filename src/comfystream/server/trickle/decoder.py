"""
Trickle Decoder Module

Provides frame decoding functionality for trickle streaming.
Decodes trickle segments into individual video frames for processing
through ComfyStream pipeline.
"""

import io
import av
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Iterator
from fractions import Fraction

logger = logging.getLogger(__name__)

class TrickleSegmentDecoder:
    """
    Decodes trickle segments into individual video frames.
    
    Handles decoding of trickle segment bytes into av.VideoFrame objects
    suitable for ComfyStream pipeline processing.
    """
    
    def __init__(self):
        self.last_width = None
        self.last_height = None
        self.last_fps = None
    
    def _get_stream_fps(self, video_stream) -> float:
        """
        Get frame rate from video stream using PyAV version-compatible approach.
        
        Args:
            video_stream: PyAV video stream object
            
        Returns:
            Frame rate as float, defaults to 30.0 if not found
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
        
    def decode_segment(self, segment_data: bytes) -> List[av.VideoFrame]:
        """
        Decode a trickle segment into individual video frames.
        
        Args:
            segment_data: Raw segment bytes to decode
            
        Returns:
            List of av.VideoFrame objects extracted from segment
        """
        try:
            if not segment_data:
                logger.warning("Empty segment data provided for decoding")
                return []
                
            # Create container from segment data
            input_buffer = io.BytesIO(segment_data)
            container = av.open(input_buffer, mode='r')
            
            frames = []
            
            # Find video stream
            video_stream = None
            if container.streams.video:
                video_stream = container.streams.video[0]
                
                # Get frame rate using PyAV version-compatible approach
                fps = self._get_stream_fps(video_stream)
                
                logger.debug(f"Found video stream: {video_stream.codec_context.name}, "
                           f"{video_stream.width}x{video_stream.height}, "
                           f"fps: {fps}")
                
                # Store metadata for consistency
                self.last_width = video_stream.width
                self.last_height = video_stream.height
                self.last_fps = fps
            else:
                logger.warning("No video stream found in segment")
                container.close()
                input_buffer.close()
                return []
            
            # Decode all video frames from the segment
            frame_count = 0
            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    frame_count += 1
                    logger.debug(f"Decoded frame {frame_count}: {frame.width}x{frame.height}, "
                               f"pts={frame.pts}, time_base={frame.time_base}")
                    frames.append(frame)
            
            container.close()
            input_buffer.close()
            
            logger.debug(f"Successfully decoded {len(frames)} frames from segment "
                        f"(size: {len(segment_data)} bytes)")
            return frames
            
        except Exception as e:
            logger.error(f"Error decoding segment: {e}")
            return []
    
    def decode_segment_generator(self, segment_data: bytes) -> Iterator[av.VideoFrame]:
        """
        Decode a trickle segment yielding frames one by one (memory efficient).
        
        Args:
            segment_data: Raw segment bytes to decode
            
        Yields:
            av.VideoFrame objects extracted from segment
        """
        try:
            if not segment_data:
                logger.warning("Empty segment data provided for decoding")
                return
                
            # Create container from segment data
            input_buffer = io.BytesIO(segment_data)
            container = av.open(input_buffer, mode='r')
            
            # Find video stream
            video_stream = None
            if container.streams.video:
                video_stream = container.streams.video[0]
                
                # Get frame rate using PyAV version-compatible approach
                fps = self._get_stream_fps(video_stream)
                
                logger.debug(f"Found video stream: {video_stream.codec_context.name}, "
                           f"{video_stream.width}x{video_stream.height}")
                
                # Store metadata for consistency
                self.last_width = video_stream.width
                self.last_height = video_stream.height
                self.last_fps = fps
            else:
                logger.warning("No video stream found in segment")
                container.close()
                input_buffer.close()
                return
            
            # Decode and yield frames one by one
            frame_count = 0
            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    frame_count += 1
                    logger.debug(f"Yielding frame {frame_count}: {frame.width}x{frame.height}")
                    yield frame
            
            container.close()
            input_buffer.close()
            
            logger.debug(f"Successfully decoded {frame_count} frames from segment generator")
            
        except Exception as e:
            logger.error(f"Error in segment generator: {e}")
            return
    
    def decode_single_frame(self, segment_data: bytes, frame_index: int = 0) -> Optional[av.VideoFrame]:
        """
        Decode a specific frame from a trickle segment.
        
        Args:
            segment_data: Raw segment bytes to decode
            frame_index: Index of frame to extract (0-based)
            
        Returns:
            av.VideoFrame at specified index, or None if not found
        """
        try:
            frames = self.decode_segment(segment_data)
            
            if frame_index < len(frames):
                return frames[frame_index]
            else:
                logger.warning(f"Frame index {frame_index} out of range (segment has {len(frames)} frames)")
                return None
                
        except Exception as e:
            logger.error(f"Error decoding single frame: {e}")
            return None
    
    def get_segment_info(self, segment_data: bytes) -> Dict[str, Any]:
        """
        Get information about a trickle segment without full decoding.
        
        Args:
            segment_data: Raw segment bytes to analyze
            
        Returns:
            Dictionary containing segment information
        """
        try:
            if not segment_data:
                return {}
                
            # Create container from segment data
            input_buffer = io.BytesIO(segment_data)
            container = av.open(input_buffer, mode='r')
            
            info = {
                'format': container.format.name,
                'duration': float(container.duration / av.time_base) if container.duration else None,
                'size_bytes': len(segment_data)
            }
            
            # Get video stream info
            if container.streams.video:
                video_stream = container.streams.video[0]
                
                # Get frame rate using PyAV version-compatible approach
                fps = self._get_stream_fps(video_stream)
                
                info.update({
                    'video_codec': video_stream.codec_context.name,
                    'width': video_stream.width,
                    'height': video_stream.height,
                    'fps': fps,
                    'time_base': [video_stream.time_base.numerator, video_stream.time_base.denominator] if video_stream.time_base else [1, 30],
                    'estimated_frames': video_stream.frames if hasattr(video_stream, 'frames') else None
                })
            
            # Get audio stream info
            if container.streams.audio:
                audio_stream = container.streams.audio[0]
                info.update({
                    'audio_codec': audio_stream.codec_context.name,
                    'sample_rate': audio_stream.sample_rate,
                    'channels': audio_stream.channels
                })
            
            container.close()
            input_buffer.close()
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting segment info: {e}")
            return {}

class TrickleFrameConverter:
    """
    Converts between trickle frames and ComfyStream pipeline formats.
    """
    
    @staticmethod
    def av_frame_to_pipeline_format(av_frame: av.VideoFrame, target_width: int = 512, 
                                   target_height: int = 512) -> av.VideoFrame:
        """
        Convert av.VideoFrame to format suitable for ComfyStream pipeline.
        
        Args:
            av_frame: Source av.VideoFrame
            target_width: Target width for pipeline
            target_height: Target height for pipeline
            
        Returns:
            av.VideoFrame formatted for pipeline
        """
        try:
            # Ensure the frame has proper dimensions
            if av_frame.width != target_width or av_frame.height != target_height:
                # Resize frame to target dimensions
                resized_frame = av_frame.reformat(width=target_width, height=target_height)
                logger.debug(f"Resized frame from {av_frame.width}x{av_frame.height} "
                           f"to {target_width}x{target_height}")
                return resized_frame
            
            return av_frame
            
        except Exception as e:
            logger.error(f"Error converting frame to pipeline format: {e}")
            return av_frame
    
    @staticmethod
    def pipeline_frame_to_av_format(processed_frame: av.VideoFrame, 
                                   original_width: int = None, original_height: int = None) -> av.VideoFrame:
        """
        Convert processed frame from pipeline back to original dimensions.
        
        Args:
            processed_frame: Frame from ComfyStream pipeline
            original_width: Original width to restore (optional)
            original_height: Original height to restore (optional)
            
        Returns:
            av.VideoFrame in original format
        """
        try:
            if original_width and original_height:
                if (processed_frame.width != original_width or 
                    processed_frame.height != original_height):
                    # Restore original dimensions
                    restored_frame = processed_frame.reformat(
                        width=original_width, height=original_height
                    )
                    logger.debug(f"Restored frame from {processed_frame.width}x{processed_frame.height} "
                               f"to {original_width}x{original_height}")
                    return restored_frame
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error converting pipeline frame back: {e}")
            return processed_frame

class TrickleStreamDecoder:
    """
    High-level decoder for continuous trickle stream processing.
    Manages frame extraction across multiple segments.
    """
    
    def __init__(self, target_width: int = 512, target_height: int = 512):
        self.decoder = TrickleSegmentDecoder()
        self.converter = TrickleFrameConverter()
        self.target_width = target_width
        self.target_height = target_height
        self.total_frames_processed = 0
        
    def process_segment(self, segment_data: bytes) -> List[av.VideoFrame]:
        """
        Process a complete segment and return frames ready for pipeline.
        
        Args:
            segment_data: Raw segment bytes
            
        Returns:
            List of av.VideoFrame objects ready for ComfyStream pipeline
        """
        try:
            # Decode segment into frames
            raw_frames = self.decoder.decode_segment(segment_data)
            
            if not raw_frames:
                logger.warning("No frames decoded from segment")
                return []
            
            # Convert frames for pipeline processing
            pipeline_frames = []
            for frame in raw_frames:
                converted_frame = self.converter.av_frame_to_pipeline_format(
                    frame, self.target_width, self.target_height
                )
                pipeline_frames.append(converted_frame)
            
            self.total_frames_processed += len(pipeline_frames)
            logger.debug(f"Processed segment: {len(raw_frames)} frames → {len(pipeline_frames)} pipeline frames "
                        f"(total processed: {self.total_frames_processed})")
            
            return pipeline_frames
            
        except Exception as e:
            logger.error(f"Error processing segment: {e}")
            return []
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the decoded stream.
        
        Returns:
            Dictionary containing stream statistics
        """
        return {
            'total_frames_processed': self.total_frames_processed,
            'target_resolution': f"{self.target_width}x{self.target_height}",
            'last_segment_info': {
                'width': self.decoder.last_width,
                'height': self.decoder.last_height,
                'fps': self.decoder.last_fps
            }
        } 