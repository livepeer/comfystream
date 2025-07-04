"""
Trickle Protocol Implementation for ComfyStream

This module implements the trickle protocol for streaming video/audio frames
using HTTP-based subscribe/publish patterns with GPU-accelerated encoding/decoding.
"""

import asyncio
import json
import logging
import numpy as np
import torch
import av
import httpx
from typing import Dict, Any, Optional, Callable, List, Set
from dataclasses import dataclass
from enum import Enum
import time
import base64
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)

class TrickleMessageType(Enum):
    """Trickle message types"""
    SUBSCRIBE = "subscribe"
    PUBLISH = "publish"
    CONTROL = "control"
    FRAME = "frame"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class TrickleFrame:
    """Trickle frame data structure"""
    frame_type: str  # 'video' or 'audio'
    data: bytes
    timestamp: float
    width: Optional[int] = None
    height: Optional[int] = None
    encoding: str = "h264"  # Default encoding
    pts: Optional[int] = None
    sample_rate: Optional[int] = None

class TrickleEncoder:
    """GPU-accelerated encoder for trickle frames"""
    
    def __init__(self, width: int = 512, height: int = 512, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._video_encoder = None
        self._audio_encoder = None
        self._nvenc_available = None  # Cache NVENC availability check
        self._init_encoders()
        
    def _init_encoders(self):
        """Initialize GPU-accelerated encoders"""
        try:
            # Initialize video encoder with memory buffer - use mpegts for streaming
            import io
            self._video_buffer = io.BytesIO()
            self._video_encoder = av.open(self._video_buffer, mode="w", format="mpegts")
            
            # Add video stream with appropriate codec - robust fallback for NVENC issues
            video_stream = None
            
            # Check NVENC availability only once and cache the result
            if self._nvenc_available is None:
                self._nvenc_available = False
                if torch.cuda.is_available():
                    try:
                        # Test NVENC availability with a dummy encoder
                        test_buffer = io.BytesIO()
                        test_encoder = av.open(test_buffer, mode="w", format="mpegts")
                        test_stream = test_encoder.add_stream("h264_nvenc", rate=self.fps)
                        test_encoder.close()
                        self._nvenc_available = True
                        logger.info("NVENC encoder available")
                    except Exception as nvenc_error:
                        logger.info(f"NVENC not available, using software encoder: {nvenc_error}")
                        self._nvenc_available = False
                else:
                    logger.info("CUDA not available, using software encoder")
            
            # Use cached NVENC availability check
            if self._nvenc_available:
                try:
                    video_stream = self._video_encoder.add_stream("h264_nvenc", rate=self.fps)
                    logger.debug("Using NVENC encoder")
                except Exception as nvenc_error:
                    logger.warning(f"NVENC failed unexpectedly: {nvenc_error}")
                    # Mark as unavailable and fallback
                    self._nvenc_available = False
                    video_stream = None
            
            # Fallback to software encoder
            if video_stream is None:
                try:
                    video_stream = self._video_encoder.add_stream("libx264", rate=self.fps)
                    logger.debug("Using software H.264 encoder")
                except Exception as sw_error:
                    logger.error(f"Software encoder also failed: {sw_error}")
                    # Final fallback - don't set video stream, handle gracefully
                    video_stream = None
                
            if video_stream:
                try:
                    video_stream.width = self.width  # type: ignore
                    video_stream.height = self.height  # type: ignore
                    video_stream.pix_fmt = "yuv420p"  # type: ignore
                    
                    # Set proper time_base for streaming (1/fps for frame-based timing)
                    from fractions import Fraction
                    video_stream.time_base = Fraction(1, self.fps)  # type: ignore
                    
                except AttributeError:
                    logger.warning("Could not set video stream attributes")
                
                # Set encoder options based on available encoder
                try:
                    # Different options for NVENC vs software encoder
                    if self._nvenc_available:
                        video_stream.options = {  # type: ignore
                            "preset": "fast",
                            "profile": "baseline",
                            "rc": "cbr",  # Constant bitrate
                            "b:v": "2M",   # 2Mbps bitrate
                            "g": str(self.fps * 2),  # GOP size
                            "keyint_min": str(self.fps)  # Min keyframe interval
                        }
                    else:
                        video_stream.options = {  # type: ignore
                            "preset": "fast",
                            "tune": "zerolatency",
                            "profile": "baseline",
                            "g": str(self.fps * 2),  # GOP size
                            "keyint_min": str(self.fps)  # Min keyframe interval
                        }
                except:
                    pass  # Ignore if options not supported
            
            # Initialize audio encoder - fix the channel setting issue
            self._audio_buffer = io.BytesIO()
            self._audio_encoder = av.open(self._audio_buffer, mode="w", format="mpegts")
            audio_stream = self._audio_encoder.add_stream("aac", rate=48000)
            
            # Set channel layout properly instead of directly setting channels
            try:
                audio_stream.layout = 'stereo'  # type: ignore # This sets channels to 2
            except AttributeError:
                # Fallback: set codec context directly
                try:
                    audio_stream.codec_context.channel_layout = 'stereo'  # type: ignore
                except:
                    logger.warning("Could not set audio channel layout, audio encoding may not work")
            
            try:
                audio_stream.sample_rate = 48000  # type: ignore
            except AttributeError:
                logger.warning("Could not set audio sample rate")
            
        except Exception as e:
            logger.error(f"Failed to initialize encoders: {e}")
            # Fallback to CPU encoding
            self._init_cpu_encoders()
    
    def _init_cpu_encoders(self):
        """Fallback CPU encoder initialization"""
        try:
            import io
            self._video_buffer = io.BytesIO()
            self._video_encoder = av.open(self._video_buffer, mode="w", format="mpegts")
            video_stream = self._video_encoder.add_stream("libx264", rate=self.fps)
            
            if video_stream:
                try:
                    video_stream.width = self.width  # type: ignore
                    video_stream.height = self.height  # type: ignore
                    video_stream.pix_fmt = "yuv420p"  # type: ignore
                    
                    # Set proper time_base for streaming
                    from fractions import Fraction
                    video_stream.time_base = Fraction(1, self.fps)  # type: ignore
                    
                except AttributeError:
                    logger.warning("Could not set video stream attributes")
                
                # Set software encoder options
                try:
                    video_stream.options = {  # type: ignore
                        "preset": "fast",
                        "tune": "zerolatency",
                        "profile": "baseline",
                        "g": str(self.fps * 2),  # GOP size
                        "keyint_min": str(self.fps)  # Min keyframe interval
                    }
                except:
                    pass  # Ignore if options not supported
            
            self._audio_buffer = io.BytesIO()
            self._audio_encoder = av.open(self._audio_buffer, mode="w", format="mpegts")
            audio_stream = self._audio_encoder.add_stream("aac", rate=48000)
            
            # Set channel layout properly instead of directly setting channels
            try:
                audio_stream.layout = 'stereo'  # type: ignore # This sets channels to 2
            except AttributeError:
                # Fallback: set codec context directly
                try:
                    audio_stream.codec_context.channel_layout = 'stereo'  # type: ignore
                except:
                    logger.warning("Could not set audio channel layout, audio encoding may not work")
            
            try:
                audio_stream.sample_rate = 48000  # type: ignore
            except AttributeError:
                logger.warning("Could not set audio sample rate")
            
        except Exception as e:
            logger.error(f"CPU encoder initialization failed: {e}")
            # Set to None to indicate encoding is not available
            self._video_encoder = None
            self._audio_encoder = None

    def encode_video_frame(self, frame: av.VideoFrame) -> bytes:
        """Encode video frame to bytes with proper timing"""
        try:
            if not self._video_encoder:
                logger.error("Video encoder not initialized")
                return b""
            
            # Resize frame if needed
            if frame.width != self.width or frame.height != self.height:
                frame = frame.reformat(width=self.width, height=self.height)
            
            # Reset buffer position for fresh encoding
            self._video_buffer.seek(0)
            self._video_buffer.truncate(0)
            
            # Get the video stream
            video_stream = self._video_encoder.streams.video[0]
            
            # Fix frame timing to match encoder's time_base
            from fractions import Fraction
            encoder_time_base = Fraction(1, self.fps)
            
            # Ensure frame has proper time_base and PTS
            if frame.time_base != encoder_time_base:
                # Convert PTS to encoder's time_base if needed
                if frame.pts is not None and frame.time_base is not None:
                    # Convert from frame's time_base to encoder's time_base
                    frame.pts = int(frame.pts * float(frame.time_base) / float(encoder_time_base))
                frame.time_base = encoder_time_base
            
            # Ensure frame has a valid PTS
            if frame.pts is None:
                # Use current time as fallback
                import time
                frame.pts = int(time.time() * self.fps) % (2**31)  # Avoid overflow
            
            # Encode frame - don't flush after every frame to avoid EOF errors
            packets = []
            try:
                for packet in video_stream.encode(frame):
                    packets.append(packet)
                    self._video_encoder.mux(packet)
                
                # Only get data if we actually have packets
                if packets:
                    encoded_data = self._video_buffer.getvalue()
                    return encoded_data
                else:
                    # Return empty bytes if no packets generated (normal for some encoders)
                    return b""
                    
            except Exception as encode_error:
                # Handle encoding errors more gracefully
                if "End of file" in str(encode_error) or "AVERROR_EOF" in str(encode_error):
                    logger.debug(f"Encoder EOF (normal): {encode_error}")
                    return b""
                else:
                    logger.error(f"Video encoding error: {encode_error}")
                    return b""
            
        except Exception as e:
            logger.error(f"Video encoding error: {e}")
            return b""
    
    def encode_audio_frame(self, frame: av.AudioFrame) -> bytes:
        """Encode audio frame to bytes"""
        try:
            if not self._audio_encoder:
                logger.error("Audio encoder not initialized")
                return b""
            
            # Reset buffer position
            self._audio_buffer.seek(0)
            self._audio_buffer.truncate(0)
            
            # Get the audio stream
            audio_stream = self._audio_encoder.streams.audio[0]
            
            # Encode frame
            for packet in audio_stream.encode(frame):
                self._audio_encoder.mux(packet)
            
            # Flush encoder
            for packet in audio_stream.encode():
                self._audio_encoder.mux(packet)
            
            # Get encoded data
            encoded_data = self._audio_buffer.getvalue()
            
            return encoded_data
            
        except Exception as e:
            logger.error(f"Audio encoding error: {e}")
            return b""
    
    def update_resolution(self, width: int, height: int):
        """Update encoder resolution"""
        if self.width != width or self.height != height:
            self.width = width
            self.height = height
            # Re-initialize encoders with new resolution
            self._init_encoders()
            logger.info(f"Trickle encoder resolution updated to {width}x{height}")

class TrickleDecoder:
    """Pipe-based decoder for continuous trickle streams (ai-runner compatible)"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._stream_buffer = None
        self._decoder_task = None
        self._frame_callback: Optional[Callable[[av.VideoFrame], None]] = None
        self._running = False
        self._write_pipe = None
        self._read_pipe = None
        self._container = None
        self._main_loop = None
        
        # Suppress FFmpeg logging to prevent spam
        import os
        os.environ['AV_LOG_LEVEL'] = 'panic'
        os.environ['FFMPEG_LOG_LEVEL'] = 'panic'
        
    def start_continuous_decoding(self, frame_callback: Callable[[av.VideoFrame], None]):
        """Start continuous decoding using ai-runner's pipe approach"""
        if self._running:
            return
        
        self._frame_callback = frame_callback
        self._running = True
        
        # Capture the current event loop for use in the decoder thread
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = asyncio.get_event_loop()
        
        # Create pipe for continuous stream (ai-runner approach)
        import os
        self._read_pipe, self._write_pipe = os.pipe()
        
        # Start decoder task that reads from pipe
        self._decoder_task = asyncio.create_task(self._decode_continuous_stream())
        
        logger.info("Started continuous trickle decoding (ai-runner compatible)")
    
    async def _decode_continuous_stream(self):
        """Decode continuous stream from pipe (ai-runner method)"""
        try:
            # Create executor for blocking decode operation
            executor = ThreadPoolExecutor(max_workers=1)
            
            # Run decoder in thread (ai-runner approach)
            await asyncio.get_event_loop().run_in_executor(
                executor, self._decode_stream_thread
            )
            
        except Exception as e:
            logger.error(f"Continuous decode error: {e}")
        finally:
            self._cleanup_pipes()
    
    def _decode_stream_thread(self):
        """Thread-based continuous stream decoder (ai-runner pattern)"""
        try:
            # Open container from pipe (ai-runner approach)
            container = av.open(f"pipe:{self._read_pipe}", mode='r')
            
            # Find video stream
            video_stream = None
            if container.streams.video:
                video_stream = container.streams.video[0]
            
            if not video_stream:
                logger.warning("No video stream found in continuous decode")
                return
            
            # Process packets continuously (ai-runner method)
            for packet in container.demux(video_stream):
                if not self._running:
                    break
                    
                try:
                    # Decode frames from packet
                    for frame in packet.decode():
                        if not self._running:
                            break
                            
                        if isinstance(frame, av.VideoFrame) and self._frame_callback:
                            # Call frame callback in thread-safe manner
                            try:
                                # Check if callback is async
                                import inspect
                                if inspect.iscoroutinefunction(self._frame_callback):
                                    # Schedule async callback in main event loop using captured loop
                                    if self._main_loop:
                                        asyncio.run_coroutine_threadsafe(
                                            self._frame_callback(frame), self._main_loop
                                        )
                                    else:
                                        logger.warning("No main event loop available for async callback")
                                else:
                                    # Call sync callback directly
                                    self._frame_callback(frame)
                            except Exception as cb_error:
                                logger.error(f"Frame callback error: {cb_error}")
                                
                except Exception as decode_error:
                    logger.debug(f"Decode error in continuous stream: {decode_error}")
                    continue  # Continue processing other packets
                    
        except Exception as e:
            logger.error(f"Stream decode thread error: {e}")
        finally:
            if 'container' in locals():
                container.close()
    
    def feed_segment(self, segment_data: bytes):
        """Feed raw trickle segment to continuous stream (ai-runner approach)"""
        if not self._running or not self._write_pipe:
            return
        
        try:
            # Write raw segment data to pipe (ai-runner method)
            import os
            os.write(self._write_pipe, segment_data)
            
        except Exception as e:
            logger.debug(f"Error feeding segment to pipe: {e}")
    
    def stop_continuous_decoding(self):
        """Stop continuous decoding"""
        self._running = False
        
        if self._decoder_task:
            self._decoder_task.cancel()
        
        self._cleanup_pipes()
    
    def _cleanup_pipes(self):
        """Clean up pipes"""
        import os
        
        if self._write_pipe:
            try:
                os.close(self._write_pipe)
            except:
                pass
            self._write_pipe = None
        
        if self._read_pipe:
            try:
                os.close(self._read_pipe)
            except:
                pass
            self._read_pipe = None
    
    # Keep old methods for backward compatibility but mark them as deprecated
    def decode_video_frame(self, data: bytes) -> Optional[av.VideoFrame]:
        """Deprecated: Use continuous decoding instead"""
        logger.warning("decode_video_frame is deprecated - use continuous decoding")
        return None
    
    def decode_audio_frame(self, data: bytes) -> Optional[av.AudioFrame]:
        """Deprecated: Use continuous decoding instead"""
        logger.warning("decode_audio_frame is deprecated - use continuous decoding")
        return None

class TrickleServer:
    """Server-side trickle protocol handler"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.subscribers = defaultdict(set)  # stream_id -> set of subscriber IDs
        self.publishers = defaultdict(set)   # stream_id -> set of publisher IDs
        self.frame_queues = defaultdict(asyncio.Queue)  # stream_id -> frame queue
        self.control_queues = defaultdict(asyncio.Queue)  # stream_id -> control queue
        self.encoders = {}  # stream_id -> TrickleEncoder
        self.decoders = {}  # stream_id -> TrickleDecoder
        self.running = False
        self.processing_active = False
        self.stream_id = ""
        self.caller_ip = ""
        self.current_resolution = {"width": 512, "height": 512}
        self.frame_processor_task = None
        self.input_client: Optional[TrickleClient] = None
        self.input_task = None
        self.output_client: Optional[TrickleClient] = None  # Client for publishing processed frames
        self.output_stream_id = ""  # Output stream name with -out suffix
        self.pipeline_producing_frames = False  # Flag to track when pipeline is actually producing frames
        
        # Segment accumulation for proper trickle behavior
        self.segment_accumulator = None  # Accumulate frames into segments
        self.last_segment_time = 0  # Track when we last published a segment
        self.segment_duration = 2.0  # Publish segments every 2 seconds
        
    async def start_processing(self, caller_ip: str, stream_id: str, input_url: Optional[str] = None):
        """Start trickle processing pipeline"""
        try:
            self.caller_ip = caller_ip
            self.stream_id = stream_id
            self.running = True
            self.processing_active = True
            
            logger.info(f"Starting trickle processing for {caller_ip} stream {stream_id}")
            
            # Initialize encoder/decoder for this stream
            self.encoders[stream_id] = TrickleEncoder(
                width=self.current_resolution["width"],
                height=self.current_resolution["height"]
            )
            self.decoders[stream_id] = TrickleDecoder()
            
            # Initialize segment accumulator for proper trickle segments
            self._init_segment_accumulator()
            
            # Start continuous decoding with callback to feed pipeline
            async def decoded_frame_callback(frame: av.VideoFrame):
                """Callback to feed decoded frames to pipeline"""
                try:
                    await self.pipeline.put_video_frame(frame)
                    logger.debug("Fed decoded frame to pipeline")
                except Exception as e:
                    logger.error(f"Error feeding decoded frame to pipeline: {e}")
            
            self.decoders[stream_id].start_continuous_decoding(decoded_frame_callback)
            
            # Start frame processor
            self.frame_processor_task = asyncio.create_task(
                self._process_frames(stream_id)
            )
            
            # If input_url provided, connect to it for input frames
            if input_url:
                self.input_client = TrickleClient(input_url, stream_id)
                self.input_task = asyncio.create_task(
                    self._connect_to_input_source(stream_id)
                )
                logger.info(f"Connecting to input source: {input_url}")
                
                # Set up output client to publish processed frames back to trickle server
                # Extract the actual channel name from the input URL for proper output naming
                # For http-trickle protocol: input_url = http://host:port/channel-name
                if '/' in input_url:
                    # Extract channel name from URL (last path component)
                    input_channel_name = input_url.rstrip('/').split('/')[-1]
                    base_host = input_url.rsplit('/', 1)[0]
                else:
                    # Fallback if no path separator
                    input_channel_name = stream_id
                    base_host = input_url
                
                # Create output stream name based on input channel name
                self.output_stream_id = f"{input_channel_name}-out"
                
                # Construct output URL for http-trickle protocol
                # Publishers POST to /channel-name/seq, so base_url should include channel name
                output_base_url = f"{base_host}/{self.output_stream_id}"
                self.output_client = TrickleClient(output_base_url, self.output_stream_id)
                logger.info(f"Will publish processed frames to: {output_base_url}")
                
            else:
                self.input_client = None
                self.input_task = None
                self.output_client = None
                logger.info("No input URL provided - expecting frames via publish endpoint")
            
            logger.info(f"Trickle processing started for stream {stream_id}")
            return {
                "status": "started", 
                "caller_ip": caller_ip, 
                "protocol": "trickle",
                "input_url": input_url,
                "stream_id": stream_id,
                "output_stream_id": self.output_stream_id if self.output_client else None
            }
            
        except Exception as e:
            logger.error(f"Error starting trickle processing: {e}")
            await self.cleanup()
            raise
    
    async def stop_processing(self):
        """Stop trickle processing pipeline"""
        self.running = False
        self.processing_active = False
        
        if self.frame_processor_task:
            self.frame_processor_task.cancel()
            try:
                await self.frame_processor_task
            except asyncio.CancelledError:
                pass
        
        if self.input_task:
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass
        
        if self.input_client:
            await self.input_client.disconnect()
        
        await self.cleanup()
        logger.info("Trickle processing stopped")
    
    def _init_segment_accumulator(self):
        """Initialize segment accumulator for proper trickle segments"""
        try:
            import io
            self.segment_buffer = io.BytesIO()
            self.segment_container = av.open(self.segment_buffer, mode="w", format="mpegts")
            
            # Add video stream to segment accumulator
            self.segment_video_stream = self.segment_container.add_stream("libx264", rate=30)
            self.segment_video_stream.width = self.current_resolution["width"]  # type: ignore
            self.segment_video_stream.height = self.current_resolution["height"]  # type: ignore
            self.segment_video_stream.pix_fmt = "yuv420p"  # type: ignore
            
            # Set proper time_base for segment accumulator
            from fractions import Fraction
            self.segment_video_stream.time_base = Fraction(1, 30)  # type: ignore
            
            # Set encoder options for segment accumulator
            self.segment_video_stream.options = {  # type: ignore
                "preset": "fast",
                "tune": "zerolatency", 
                "profile": "baseline",
                "g": "60",  # GOP size
                "keyint_min": "30"  # Min keyframe interval
            }
            
            self.last_segment_time = time.time()
            logger.info("Segment accumulator initialized for proper trickle segments")
            
        except Exception as e:
            logger.error(f"Failed to initialize segment accumulator: {e}")
            self.segment_accumulator = None

    async def _add_frame_to_segment(self, frame: av.VideoFrame, stream_id: str):
        """Add frame to segment accumulator and publish segment when ready"""
        try:
            if not hasattr(self, 'segment_container') or not self.segment_container:
                logger.warning("Segment accumulator not initialized")
                return

            # Fix frame timing to match segment accumulator's time_base
            from fractions import Fraction
            segment_time_base = Fraction(1, 30)
            
            # Ensure frame has proper time_base and PTS
            if frame.time_base != segment_time_base:
                # Convert PTS to segment's time_base if needed
                if frame.pts is not None and frame.time_base is not None:
                    frame.pts = int(frame.pts * float(frame.time_base) / float(segment_time_base))
                frame.time_base = segment_time_base
            
            # Ensure frame has a valid PTS
            if frame.pts is None:
                frame.pts = int((time.time() - self.last_segment_time) * 30)
            
            # Resize frame to match segment accumulator resolution
            if frame.width != self.current_resolution["width"] or frame.height != self.current_resolution["height"]:
                frame = frame.reformat(
                    width=self.current_resolution["width"], 
                    height=self.current_resolution["height"]
                )
            
            # Add frame to segment accumulator
            for packet in self.segment_video_stream.encode(frame):
                self.segment_container.mux(packet)
            
            # Check if it's time to publish the segment
            current_time = time.time()
            if current_time - self.last_segment_time >= self.segment_duration:
                await self._finalize_and_publish_segment(stream_id)
                
        except Exception as e:
            logger.error(f"Error adding frame to segment: {e}")

    async def _finalize_and_publish_segment(self, stream_id: str):
        """Finalize current segment and publish it as a trickle segment"""
        try:
            if not hasattr(self, 'segment_container') or not self.segment_container:
                return

            # Flush the encoder to finalize the segment
            for packet in self.segment_video_stream.encode():
                self.segment_container.mux(packet)
            
            # Get the accumulated segment data
            segment_data = self.segment_buffer.getvalue()
            
            if len(segment_data) > 0:
                # Create trickle frame with the complete segment
                trickle_frame = TrickleFrame(
                    frame_type="video",
                    data=segment_data,
                    timestamp=time.time(),
                    width=self.current_resolution["width"],
                    height=self.current_resolution["height"],
                    encoding="h264"
                )
                
                # Queue segment for local subscribers 
                if self.subscribers[stream_id]:
                    await self.frame_queues[stream_id].put(trickle_frame)
                
                # Publish segment to trickle server
                if self.output_client and self.pipeline_producing_frames:
                    try:
                        await self.output_client.publish_frame(trickle_frame)
                        logger.info(f"Published {len(segment_data)} byte segment to {self.output_stream_id} (accumulated {self.segment_duration}s)")
                    except Exception as publish_error:
                        logger.error(f"Failed to publish segment: {publish_error}")
                
                # Reset accumulator for next segment
                self._init_segment_accumulator()
            else:
                logger.warning("Empty segment data, skipping publish")
                
        except Exception as e:
            logger.error(f"Error finalizing segment: {e}")

    async def cleanup(self):
        """Clean up trickle resources"""
        self.running = False
        self.processing_active = False
        
        # Clear all queues and connections
        for stream_id in list(self.subscribers.keys()):
            self.subscribers[stream_id].clear()
            self.publishers[stream_id].clear()
        
        for stream_id in list(self.frame_queues.keys()):
            while not self.frame_queues[stream_id].empty():
                try:
                    self.frame_queues[stream_id].get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        for stream_id in list(self.control_queues.keys()):
            while not self.control_queues[stream_id].empty():
                try:
                    self.control_queues[stream_id].get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        # Clean up encoders/decoders
        for decoder in self.decoders.values():
            decoder.stop_continuous_decoding()
        self.encoders.clear()
        self.decoders.clear()
        
        # Clean up segment accumulator
        if hasattr(self, 'segment_container') and self.segment_container:
            try:
                self.segment_container.close()
            except:
                pass
            self.segment_container = None
        
        # Clean up input client
        if self.input_client:
            await self.input_client.disconnect()
            self.input_client = None
        
        self.input_task = None
        
        # Clean up pipeline
        if self.pipeline:
            await self.pipeline.cleanup()
        
        logger.info("Trickle server cleanup complete")
    
    async def handle_subscribe(self, stream_id: str, subscriber_id: Optional[str] = None):
        """Handle subscribe request"""
        if not subscriber_id:
            subscriber_id = str(uuid.uuid4())
        
        self.subscribers[stream_id].add(subscriber_id)
        logger.info(f"Subscriber {subscriber_id} added to stream {stream_id}")
        
        # Generator function to yield frames
        async def frame_generator():
            try:
                while self.running and subscriber_id in self.subscribers[stream_id]:
                    try:
                        # Wait for frame in queue
                        frame = await asyncio.wait_for(
                            self.frame_queues[stream_id].get(), 
                            timeout=1.0
                        )
                        
                        # Convert frame to JSON message
                        message = {
                            "type": "frame",
                            "frame_type": frame.frame_type,
                            "data": base64.b64encode(frame.data).decode(),
                            "timestamp": frame.timestamp,
                            "encoding": frame.encoding,
                            "stream_id": stream_id
                        }
                        
                        if frame.frame_type == "video":
                            message.update({
                                "width": frame.width,
                                "height": frame.height,
                                "pts": frame.pts
                            })
                        elif frame.frame_type == "audio":
                            message.update({
                                "sample_rate": frame.sample_rate,
                                "pts": frame.pts
                            })
                        
                        yield json.dumps(message).encode() + b"\n"
                        
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        heartbeat = {
                            "type": "heartbeat",
                            "timestamp": time.time(),
                            "stream_id": stream_id
                        }
                        yield json.dumps(heartbeat).encode() + b"\n"
                        
                    except Exception as e:
                        logger.error(f"Error in frame generator: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"Frame generator error: {e}")
            finally:
                # Clean up subscriber
                self.subscribers[stream_id].discard(subscriber_id)
                logger.info(f"Subscriber {subscriber_id} removed from stream {stream_id}")
        
        return frame_generator()
    
    async def handle_publish(self, stream_id: str, frame_data: Dict[str, Any]):
        """Handle publish request"""
        try:
            # Decode frame data
            frame = TrickleFrame(
                frame_type=frame_data["frame_type"],
                data=base64.b64decode(frame_data["data"]),
                timestamp=frame_data["timestamp"],
                encoding=frame_data.get("encoding", "h264")
            )
            
            if frame.frame_type == "video":
                frame.width = frame_data.get("width")
                frame.height = frame_data.get("height")
                frame.pts = frame_data.get("pts")
            elif frame.frame_type == "audio":
                frame.sample_rate = frame_data.get("sample_rate")
                frame.pts = frame_data.get("pts")
            
            # Process frame through pipeline
            await self._process_incoming_frame(stream_id, frame)
            
            return {"status": "published", "timestamp": frame.timestamp}
            
        except Exception as e:
            logger.error(f"Publish error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def handle_control(self, stream_id: str, control_data: Dict[str, Any]):
        """Handle control message"""
        try:
            control_type = control_data.get("control_type")
            
            if control_type == "resolution_change":
                width = control_data.get("width")
                height = control_data.get("height")
                
                if width and height:
                    await self._handle_resolution_change(stream_id, width, height)
                    return {"status": "success", "message": f"Resolution updated to {width}x{height}"}
                else:
                    return {"status": "error", "message": "Missing width or height"}
            
            elif control_type == "quality_change":
                quality = control_data.get("quality")
                logger.info(f"Quality change request: {quality}")
                return {"status": "success", "message": f"Quality change acknowledged: {quality}"}
            
            elif control_type == "heartbeat_response":
                return {"status": "success", "message": "Heartbeat received"}
            
            else:
                return {"status": "error", "message": f"Unknown control type: {control_type}"}
                
        except Exception as e:
            logger.error(f"Control error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _process_incoming_frame(self, stream_id: str, frame: TrickleFrame):
        """Process incoming frame through pipeline - AI runner continuous decode approach"""
        try:
            if frame.frame_type == "video":
                # Get decoder for this stream
                decoder = self.decoders.get(stream_id)
                if not decoder:
                    return
                
                # Feed raw segment data to continuous decoder (ai-runner approach)
                decoder.feed_segment(frame.data)
                logger.debug(f"Fed {len(frame.data)} bytes to continuous decoder - no decode errors")
                
            elif frame.frame_type == "audio":
                # Audio processing - feed to continuous decoder when implemented
                logger.debug(f"Received audio segment of {len(frame.data)} bytes")
                        
        except Exception as e:
            logger.error(f"Error processing incoming frame: {e}")
    
    async def _process_frames(self, stream_id: str):
        """Process frames from pipeline and distribute to subscribers - only publish when pipeline is active"""
        try:
            logger.info("Waiting for pipeline to start producing processed frames...")
            
            while self.running and self.processing_active:
                try:
                    # Get processed frame from pipeline
                    if self.pipeline:
                        processed_frame = await self.pipeline.get_processed_video_frame()
                        
                        # Mark that pipeline is now producing frames
                        if not self.pipeline_producing_frames:
                            self.pipeline_producing_frames = True
                            logger.info("Pipeline started producing frames - beginning segment accumulation")
                        
                        # Add frame to segment accumulator instead of publishing immediately
                        await self._add_frame_to_segment(processed_frame, stream_id)
                        
                except Exception as e:
                    logger.error(f"Error processing frames: {e}")
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Frame processing task error: {e}")
    
    async def _handle_resolution_change(self, stream_id: str, width: int, height: int):
        """Handle resolution change request"""
        try:
            # Update current resolution
            self.current_resolution = {"width": width, "height": height}
            
            # Update pipeline resolution
            if self.pipeline:
                self.pipeline.width = width
                self.pipeline.height = height
                await self.pipeline.warm_video()
            
            # Update encoder resolution
            encoder = self.encoders.get(stream_id)
            if encoder:
                encoder.update_resolution(width, height)
            
            logger.info(f"Trickle resolution updated to {width}x{height}")
            
        except Exception as e:
            logger.error(f"Error handling resolution change: {e}")
    
    async def _connect_to_input_source(self, stream_id: str):
        """Connect to input trickle source and process incoming frames"""
        try:
            await self.input_client.connect()
            
            # Create a queue to pass frames from the callback to the async context
            input_frame_queue = asyncio.Queue()
            
            # Store reference to current event loop for thread-safe operations
            current_loop = asyncio.get_running_loop()
            
            def on_input_frame(frame: TrickleFrame):
                """Handle incoming frame from input source"""
                try:
                    # Put frame in queue for async processing (thread-safe)
                    asyncio.run_coroutine_threadsafe(
                        input_frame_queue.put(frame), 
                        current_loop
                    )
                except Exception as e:
                    logger.error(f"Error queuing input frame: {e}")
            
            # Start async task to process frames from queue
            async def process_frame_queue():
                """Process frames from the input queue"""
                while self.running:
                    try:
                        # Wait for frame in queue with timeout
                        frame = await asyncio.wait_for(input_frame_queue.get(), timeout=1.0)
                        await self._process_incoming_frame(stream_id, frame)
                    except asyncio.TimeoutError:
                        continue  # Keep checking for new frames
                    except Exception as e:
                        logger.error(f"Error processing queued frame: {e}")
            
            # Start the frame processing task
            frame_processor = asyncio.create_task(process_frame_queue())
            
            try:
                # Subscribe to input frames
                await self.input_client.subscribe(on_input_frame)
            finally:
                # Clean up the frame processor
                frame_processor.cancel()
                try:
                    await frame_processor
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            logger.error(f"Error connecting to input source: {e}")
    
    def get_status(self):
        """Get current trickle server status"""
        return {
            "processing_active": self.processing_active,
            "caller_ip": self.caller_ip,
            "stream_id": self.stream_id,
            "protocol": "trickle",
            "input_connected": self.input_client is not None,
            "input_url": self.input_client.base_url if self.input_client else None,
            "subscribers": {
                stream_id: len(subscribers) 
                for stream_id, subscribers in self.subscribers.items()
            },
            "publishers": {
                stream_id: len(publishers) 
                for stream_id, publishers in self.publishers.items()
            },
            "current_resolution": self.current_resolution,
            "pipeline_initialized": self.pipeline is not None
        }

class TrickleClient:
    """Client for handling trickle protocol communication following Livepeer's proven approach"""
    
    def __init__(self, base_url: str, stream_id: str):
        self.base_url = base_url.rstrip('/')
        self.stream_id = stream_id
        self.current_seq = -1  # Start with -1 to get latest segment
        self.session: Optional[httpx.AsyncClient] = None
        self.encoder = TrickleEncoder()
        self.decoder = TrickleDecoder()
        self.running = False
        self.frame_callbacks = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.discovered = False  # Track if we've discovered the latest segment
        self.segment_buffer = bytearray()  # Accumulate segments for decoding
        
    async def connect(self):
        """Connect to trickle server"""
        self.session = httpx.AsyncClient(timeout=30.0)
        logger.info(f"Trickle client connected to {self.base_url}")
        
    async def disconnect(self):
        """Disconnect from trickle server"""
        self.running = False
        if self.session:
            await self.session.aclose()
        self.executor.shutdown(wait=True)
        logger.info("Trickle client disconnected")
    
    async def subscribe(self, frame_callback: Callable[[TrickleFrame], None]):
        """Subscribe to receive frames using standard trickle protocol"""
        if not self.session:
            await self.connect()
        
        assert self.session is not None  # Help type checker understand session is not None after connect
        self.running = True
        self.frame_callbacks['default'] = frame_callback
        
        try:
            # Try to start from a recent keyframe instead of latest segment
            # First, try to get the latest segment to see current position
            latest_url = f"{self.base_url}/-1"
            subscribe_url = f"{self.base_url}/-1"  # Default fallback
            keyframe_seq = -1
            
            try:
                async with self.session.stream("GET", latest_url) as response:
                    if response.status_code == 200 and "Lp-Trickle-Seq" in response.headers:
                        latest_seq = int(response.headers["Lp-Trickle-Seq"])
                        
                        # Try different keyframe positions, starting close to latest
                        keyframe_attempts = [
                            latest_seq - 2,   # Very recent (likely available)
                            latest_seq - 5,   # Recent
                            latest_seq - 10,  # GOP boundary
                            latest_seq - 1,   # Almost latest
                        ]
                        
                        # Test each keyframe position to see if it's available
                        for attempt_seq in keyframe_attempts:
                            if attempt_seq < 0:
                                continue
                                
                            test_url = f"{self.base_url}/{attempt_seq}"
                            try:
                                # Use a quick GET request to test availability
                                test_response = await self.session.get(test_url, timeout=2.0)
                                if test_response.status_code == 200:
                                    keyframe_seq = attempt_seq
                                    subscribe_url = test_url
                                    logger.info(f"Latest sequence: {latest_seq}, using available keyframe at: {keyframe_seq}")
                                    break
                                elif test_response.status_code == 470:
                                    logger.debug(f"Sequence {attempt_seq} not available (470)")
                                    continue
                            except Exception as test_error:
                                logger.debug(f"Could not test sequence {attempt_seq}: {test_error}")
                                continue
                        
                        # If no keyframe found, use latest
                        if keyframe_seq == -1:
                            logger.info(f"No older sequences available, using latest: {latest_seq}")
                            subscribe_url = f"{self.base_url}/-1"
                            keyframe_seq = latest_seq
                            
                    else:
                        logger.warning("Could not get sequence info from latest segment")
                        
            except Exception as seq_error:
                logger.warning(f"Could not determine latest sequence: {seq_error}")
                # Start from beginning to ensure we get headers
                subscribe_url = f"{self.base_url}/0"
                keyframe_seq = 0
            
            logger.info(f"Subscribing to trickle stream: {subscribe_url}")
            
            # Try the selected URL, with fallback to latest if it fails
            for attempt_url in [subscribe_url, f"{self.base_url}/-1"]:
                try:
                    async with self.session.stream("GET", attempt_url) as response:
                        if response.status_code == 200:
                            logger.info("Trickle subscription established")
                            
                            # Get current sequence number from header if available
                            if "Lp-Trickle-Seq" in response.headers:
                                self.current_seq = int(response.headers["Lp-Trickle-Seq"])
                                logger.info(f"Current trickle sequence: {self.current_seq}")
                            else:
                                self.current_seq = keyframe_seq
                            
                            # Process incoming data (raw segment data in standard trickle)
                            async for chunk in response.aiter_bytes():
                                if not self.running:
                                    break
                                
                                try:
                                    # For standard trickle protocol, we receive raw segment data
                                    # Convert to TrickleFrame for processing
                                    frame = TrickleFrame(
                                        frame_type="video",  # Assume video for now, could be detected
                                        data=chunk,
                                        timestamp=time.time(),
                                        encoding="h264"
                                    )
                                    
                                    # Call frame callback
                                    if 'default' in self.frame_callbacks:
                                        await asyncio.get_event_loop().run_in_executor(
                                            self.executor,
                                            self.frame_callbacks['default'],
                                            frame
                                        )
                                        
                                except Exception as e:
                                    logger.error(f"Error processing trickle segment: {e}")
                            
                            # After current segment, subscribe to next sequence
                            if self.running and self.current_seq >= 0:
                                await self._subscribe_to_next_segment()
                            
                            return  # Success, exit the retry loop
                            
                        elif response.status_code == 470:
                            logger.warning(f"Sequence not available (470) for {attempt_url}")
                            if attempt_url != f"{self.base_url}/-1":
                                continue  # Try next URL (fallback to latest)
                            else:
                                logger.error("Even latest sequence unavailable")
                                return
                        else:
                            logger.error(f"Subscribe failed: {response.status_code} - {response.text}")
                            if attempt_url != f"{self.base_url}/-1":
                                continue  # Try next URL
                            else:
                                return
                                
                except Exception as stream_error:
                    logger.error(f"Stream error for {attempt_url}: {stream_error}")
                    if attempt_url != f"{self.base_url}/-1":
                        continue  # Try next URL
                    else:
                        break
                        
        except Exception as e:
            logger.error(f"Subscribe error: {e}")
    
    async def _subscribe_to_next_segment(self):
        """Subscribe to the next segment in sequence (standard trickle protocol)"""
        try:
            next_seq = self.current_seq + 1
            subscribe_url = f"{self.base_url}/{next_seq}"
            logger.info(f"Pre-connecting to next segment: {subscribe_url}")
            
            async with self.session.stream("GET", subscribe_url) as response:
                if response.status_code == 200:
                    self.current_seq = next_seq
                    
                    # Update sequence from header if available
                    if "Lp-Trickle-Seq" in response.headers:
                        self.current_seq = int(response.headers["Lp-Trickle-Seq"])
                    
                    async for chunk in response.aiter_bytes():
                        if not self.running:
                            break
                        
                        try:
                            frame = TrickleFrame(
                                frame_type="video",
                                data=chunk,
                                timestamp=time.time(),
                                encoding="h264"
                            )
                            
                            if 'default' in self.frame_callbacks:
                                await asyncio.get_event_loop().run_in_executor(
                                    self.executor,
                                    self.frame_callbacks['default'],
                                    frame
                                )
                                
                        except Exception as e:
                            logger.error(f"Error processing next segment: {e}")
                    
                    # Continue to next segment if still running
                    if self.running:
                        await self._subscribe_to_next_segment()
                else:
                    logger.warning(f"Next segment not available: {response.status_code}")
                    # Maybe wait a bit and retry for live streams
                    if self.running:
                        await asyncio.sleep(0.1)
                        await self._subscribe_to_next_segment()
                    
        except Exception as e:
            logger.error(f"Error subscribing to next segment: {e}")
            # Retry after a short delay for live streams
            if self.running:
                await asyncio.sleep(1.0)
                await self._subscribe_to_next_segment()
    
    async def publish_frame(self, frame: TrickleFrame, seq: Optional[int] = None):
        """Publish a frame using standard trickle protocol"""
        if not self.session:
            await self.connect()
        
        assert self.session is not None  # Help type checker understand session is not None after connect
        try:
            if seq is None:
                seq = self.current_seq + 1
                self.current_seq = seq
            
            # Use http-trickle protocol: POST /channel-name/seq
            # The base_url already includes the channel name
            publish_url = f"{self.base_url}/{seq}"
            
            response = await self.session.post(
                publish_url,
                content=frame.data,
                headers={
                    "Content-Type": "application/octet-stream"
                }
            )
            
            if response.status_code not in [200, 201]:
                logger.error(f"Publish failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Publish error: {e}")
    
    async def send_control_message(self, message: Dict[str, Any]):
        """Send control message (not part of standard trickle protocol)"""
        logger.warning("Control messages not supported in standard trickle protocol")
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming trickle message (legacy method for compatibility)"""
        # This method is kept for compatibility but not used in standard trickle
        pass
    
    async def _handle_control_message(self, message: Dict[str, Any]):
        """Handle control message (legacy method for compatibility)"""
        # This method is kept for compatibility but not used in standard trickle
        pass
    
    def encode_av_frame(self, frame: av.VideoFrame) -> TrickleFrame:
        """Convert av.VideoFrame to TrickleFrame"""
        encoded_data = self.encoder.encode_video_frame(frame)
        return TrickleFrame(
            frame_type="video",
            data=encoded_data,
            timestamp=time.time(),
            width=frame.width,
            height=frame.height,
            encoding="h264",
            pts=frame.pts
        )
    
    def encode_av_audio_frame(self, frame: av.AudioFrame) -> TrickleFrame:
        """Convert av.AudioFrame to TrickleFrame"""
        encoded_data = self.encoder.encode_audio_frame(frame)
        return TrickleFrame(
            frame_type="audio",
            data=encoded_data,
            timestamp=time.time(),
            sample_rate=frame.sample_rate,
            encoding="aac",
            pts=frame.pts
        )
    
    def decode_to_av_frame(self, trickle_frame: TrickleFrame) -> Optional[av.VideoFrame]:
        """Convert TrickleFrame to av.VideoFrame"""
        if trickle_frame.frame_type == "video":
            return self.decoder.decode_video_frame(trickle_frame.data)
        # Audio frames are not VideoFrames, so we return None for them
        return None 