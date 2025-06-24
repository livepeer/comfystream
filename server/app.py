import argparse
import asyncio
import json
import logging
import os
import sys
import time
import secrets
import torch
import uuid
import base64
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Initialize CUDA before any other imports to prevent core dump.
if torch.cuda.is_available():
    torch.cuda.init()


from aiohttp import web, MultipartWriter
from aiohttp_cors import setup as setup_cors, ResourceOptions
from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
# Import HTTP streaming modules
from http_streaming.routes import setup_routes
from aiortc.codecs import h264
from aiortc.rtcrtpsender import RTCRtpSender
from comfystream.pipeline import Pipeline
from twilio.rest import Client
from comfystream.server.utils import patch_loop_datagram, add_prefix_to_app_routes, FPSMeter
from comfystream.server.metrics import MetricsManager, StreamStatsManager

# Import trickle streaming components
from comfystream.server.trickle import (
    TricklePublisher, TrickleSubscriber, enhanced_segment_publisher,
    TrickleStreamDecoder, TrickleSegmentEncoder, TrickleMetadataExtractor
)
from comfystream.server.trickle.frame import VideoFrame, AudioFrame

logger = logging.getLogger(__name__)
logging.getLogger("aiortc.rtcrtpsender").setLevel(logging.WARNING)
logging.getLogger("aiortc.rtcrtpreceiver").setLevel(logging.WARNING)


MAX_BITRATE = 2000000
MIN_BITRATE = 2000000

@dataclass
class StreamManifest:
    """Represents a streaming session with manifest ID for tracking"""
    manifest_id: str
    input_stream_url: str  # Input stream URL for trickle subscription
    output_stream_url: str  # Output stream URL for trickle publishing
    created_at: datetime
    status: str  # 'starting', 'active', 'stopping', 'stopped'
    pipeline: Optional[Any] = None
    publisher_task: Optional[asyncio.Task] = None
    subscriber_task: Optional[asyncio.Task] = None
    frame_processor_task: Optional[asyncio.Task] = None
    frame_queue: Optional[asyncio.Queue] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "manifest_id": self.manifest_id,
            "input_stream_url": self.input_stream_url,
            "output_stream_url": self.output_stream_url,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "metadata": self.metadata or {}
        }


class VideoStreamTrack(MediaStreamTrack):
    """video stream track that processes video frames using a pipeline.

    Attributes:
        kind (str): The kind of media, which is "video" for this class.
        track (MediaStreamTrack): The underlying media stream track.
        pipeline (Pipeline): The processing pipeline to apply to each video frame.
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack, pipeline: Pipeline):
        """Initialize the VideoStreamTrack.

        Args:
            track: The underlying media stream track.
            pipeline: The processing pipeline to apply to each video frame.
        """
        super().__init__()
        self.track = track
        self.pipeline = pipeline
        self.fps_meter = FPSMeter(
            metrics_manager=app["metrics_manager"], track_id=track.id
        )
        self.running = True
        self.collect_task = asyncio.create_task(self.collect_frames())
        
        # Add cleanup when track ends
        @track.on("ended")
        async def on_ended():
            logger.info("Source video track ended, stopping collection")
            await cancel_collect_frames(self)

    async def collect_frames(self):
        """Collect video frames from the underlying track and pass them to
        the processing pipeline. Stops when track ends or connection closes.
        """
        try:
            while self.running:
                try:
                    frame = await self.track.recv()
                    await self.pipeline.put_video_frame(frame)
                except asyncio.CancelledError:
                    logger.info("Frame collection cancelled")
                    break
                except Exception as e:
                    if "MediaStreamError" in str(type(e)):
                        logger.info("Media stream ended")
                    else:
                        logger.error(f"Error collecting video frames: {str(e)}")
                    self.running = False
                    break
            
            # Perform cleanup outside the exception handler
            logger.info("Video frame collection stopped")
        except asyncio.CancelledError:
            logger.info("Frame collection task cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in frame collection: {str(e)}")
        finally:
            await self.pipeline.cleanup()

    async def recv(self):
        """Receive a processed video frame from the pipeline, increment the frame
        count for FPS calculation and return the processed frame to the client.
        """
        processed_frame = await self.pipeline.get_processed_video_frame()

                # Update the frame buffer with the processed frame
        try:
            from frame_buffer import FrameBuffer
            frame_buffer = FrameBuffer.get_instance()
            frame_buffer.update_frame(processed_frame)
        except Exception as e:
            # Don't let frame buffer errors affect the main pipeline
            print(f"Error updating frame buffer: {e}")

        # Increment the frame count to calculate FPS.
        await self.fps_meter.increment_frame_count()

        return processed_frame


class TrickleVideoStreamTrack(MediaStreamTrack):
    """Trickle video stream track that processes video frames from trickle input stream"""
    
    kind = "video"
    
    def __init__(self, input_stream_url: str, pipeline: Pipeline, width: int = 512, height: int = 512):
        super().__init__()
        self.input_stream_url = input_stream_url
        self.pipeline = pipeline
        self.width = width
        self.height = height
        self.running = True
        self.frame_queue = asyncio.Queue()
        self.collect_task = asyncio.create_task(self.collect_frames())
        
    async def collect_frames(self):
        """Collect frames from trickle input stream and process through pipeline"""
        try:
            # Initialize decoder for proper frame handling
            stream_decoder = TrickleStreamDecoder(target_width=self.width, target_height=self.height)
            
            logger.info(f"Starting trickle frame collection from: {self.input_stream_url}")
            
            # Use TrickleSubscriber following http-trickle protocol
            async with TrickleSubscriber(self.input_stream_url) as subscriber:
                frame_count = 0
                segment_count = 0
                current_segment = None
                no_segment_count = 0
                max_no_segment_retries = 10
                
                while self.running:
                    try:
                        # Get a new segment if we don't have one or current one is exhausted
                        if current_segment is None:
                            current_segment = await subscriber.next()
                            
                            if current_segment is None:
                                no_segment_count += 1
                                if no_segment_count <= max_no_segment_retries:
                                    logger.debug(f"No segment available yet, retry {no_segment_count}/{max_no_segment_retries}")
                                    await asyncio.sleep(0.5)
                                    continue
                                else:
                                    logger.warning(f"No segments received after {max_no_segment_retries} retries, ending stream")
                                    break
                            else:
                                no_segment_count = 0
                                segment_count += 1
                                logger.debug(f"Got new trickle segment {segment_count}")
                        
                        # Read the complete trickle segment data
                        segment_data = await self._read_complete_segment(current_segment)
                        
                        # If no segment data, the segment is exhausted - get a new one
                        if not segment_data:
                            logger.debug("Trickle segment exhausted, getting next segment...")
                            if hasattr(current_segment, 'close'):
                                try:
                                    await current_segment.close()
                                except:
                                    pass
                            current_segment = None
                            continue
                        
                        # Decode segment into individual video frames
                        try:
                            decoded_frames = stream_decoder.process_segment(segment_data)
                            
                            if not decoded_frames:
                                logger.warning(f"No frames decoded from segment {segment_count}")
                                if hasattr(current_segment, 'close'):
                                    try:
                                        await current_segment.close()
                                    except:
                                        pass
                                current_segment = None
                                continue
                            
                            logger.debug(f"Decoded {len(decoded_frames)} frames from segment {segment_count}")
                            
                            # Process each frame through ComfyStream pipeline following app.py pattern
                            for i, frame in enumerate(decoded_frames):
                                if self.pipeline is not None:
                                    # Feed frame into pipeline - same pattern as VideoStreamTrack
                                    await self.pipeline.put_video_frame(frame)
                                    frame_count += 1
                                    logger.debug(f"Fed frame {frame_count} into pipeline from segment {segment_count}")
                            
                            # Mark segment as processed, get next one
                            if current_segment:
                                if hasattr(current_segment, 'close'):
                                    try:
                                        await current_segment.close()
                                    except:
                                        pass
                            current_segment = None
                                    
                        except Exception as e:
                            logger.error(f"Error processing segment {segment_count}: {e}")
                            if current_segment:
                                if hasattr(current_segment, 'close'):
                                    try:
                                        await current_segment.close()
                                    except:
                                        pass
                            current_segment = None
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error in trickle frame processing loop: {e}")
                        if current_segment:
                            if hasattr(current_segment, 'close'):
                                try:
                                    if asyncio.iscoroutinefunction(current_segment.close):
                                        await current_segment.close()
                                    else:
                                        await current_segment.close()
                                except:
                                    pass
                            current_segment = None
                        await asyncio.sleep(0.5)
                        
                # Clean up current segment when done
                if current_segment:
                    if hasattr(current_segment, 'close'):
                        try:
                            if asyncio.iscoroutinefunction(current_segment.close):
                                current_segment.close()
                            else:
                                current_segment.close()
                        except:
                            pass
                        
            logger.info(f"Trickle frame collection finished, processed {segment_count} segments, {frame_count} total frames")
            
        except Exception as e:
            logger.error(f"Trickle frame collector error: {e}")
        finally:
            await self.pipeline.cleanup()

    async def _read_complete_segment(self, segment_reader) -> bytes:
        """Read the complete trickle segment data before processing as a frame"""
        try:
            complete_data = b""
            while True:
                # Read data in chunks
                chunk = await segment_reader.read(8192)
                if not chunk:
                    # End of segment
                    break
                complete_data += chunk
            
            logger.debug(f"Read complete segment: {len(complete_data)} bytes")
            return complete_data
            
        except Exception as e:
            logger.error(f"Error reading complete segment: {e}")
            return b""

    async def recv(self):
        """Receive a processed video frame from the pipeline - same as VideoStreamTrack"""
        processed_frame = await self.pipeline.get_processed_video_frame()

        # Update the frame buffer with the processed frame
        try:
            from frame_buffer import FrameBuffer
            frame_buffer = FrameBuffer.get_instance()
            frame_buffer.update_frame(processed_frame)
        except Exception as e:
            # Don't let frame buffer errors affect the main pipeline
            print(f"Error updating frame buffer: {e}")

        return processed_frame


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track: MediaStreamTrack, pipeline):
        super().__init__()
        self.track = track
        self.pipeline = pipeline
        self.running = True
        self.collect_task = asyncio.create_task(self.collect_frames())
        
        # Add cleanup when track ends
        @track.on("ended")
        async def on_ended():
            logger.info("Source audio track ended, stopping collection")
            await cancel_collect_frames(self)

    async def collect_frames(self):
        """Collect audio frames from the underlying track and pass them to
        the processing pipeline. Stops when track ends or connection closes.
        """
        try:
            while self.running:
                try:
                    frame = await self.track.recv()
                    await self.pipeline.put_audio_frame(frame)
                except asyncio.CancelledError:
                    logger.info("Audio frame collection cancelled")
                    break
                except Exception as e:
                    if "MediaStreamError" in str(type(e)):
                        logger.info("Media stream ended")
                    else:
                        logger.error(f"Error collecting audio frames: {str(e)}")
                    self.running = False
                    break
            
            # Perform cleanup outside the exception handler
            logger.info("Audio frame collection stopped")
        except asyncio.CancelledError:
            logger.info("Frame collection task cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in audio frame collection: {str(e)}")
        finally:
            await self.pipeline.cleanup()

    async def recv(self):
        return await self.pipeline.get_processed_audio_frame()


def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    codecPrefs = [codec for codec in codecs if codec.mimeType == forced_codec]
    transceiver.setCodecPreferences(codecPrefs)


def get_twilio_token():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if account_sid is None or auth_token is None:
        return None

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token


def get_ice_servers():
    ice_servers = []

    token = get_twilio_token()
    if token is not None:
        # Use Twilio TURN servers
        for server in token.ice_servers:
            if server["url"].startswith("turn:"):
                turn = RTCIceServer(
                    urls=[server["urls"]],
                    credential=server["credential"],
                    username=server["username"],
                )
                ice_servers.append(turn)

    return ice_servers


async def warmup_pipeline(request: web.Request) -> web.Response:
    """Warmup endpoint to prepare pipeline with specific prompts and resolution"""
    try:
        data = await request.json()
        
        prompts = data.get('prompts', [])
        width = data.get('width', 512)
        height = data.get('height', 512)
        
        if not prompts:
            return web.json_response(
                {'error': 'No prompts provided'}, 
                status=400
            )
        
        logger.info(f"Warming up pipeline with resolution {width}x{height}")
        
        # Create a temporary pipeline for warmup
        temp_pipeline = Pipeline(
            width=width,
            height=height,
            cwd=request.app["workspace"],
            disable_cuda_malloc=True,
            gpu_only=True,
            preview_method='none',
            comfyui_inference_log_level=request.app.get("comfui_inference_log_level", None),
        )
        
        try:
            # Set prompts
            await temp_pipeline.set_prompts(prompts)
            logger.info("Prompts set for warmup")
            
            # Perform warmup
            await temp_pipeline.warm_video()
            logger.info("Pipeline warmup completed successfully")
            
            return web.json_response({
                'success': True,
                'message': f'Pipeline warmed up at {width}x{height}',
                'width': width,
                'height': height
            })
            
        finally:
            await temp_pipeline.cleanup()
            
    except Exception as e:
        logger.error(f"Error during pipeline warmup: {e}")
        return web.json_response(
            {'error': f'Failed to warmup pipeline: {str(e)}'}, 
            status=500
        )


async def process_capability_request(request: web.Request) -> web.Response:
    """
    Process a capability request similar to BYOC reverse server interface.
    
    This endpoint receives requests from Livepeer orchestrators and routes them
    to the appropriate ComfyStream pipeline processing.
    """
    capability = request.match_info.get('capability', '')
    
    try:
        # Parse the Livepeer header
        livepeer_header = request.headers.get('Livepeer', '')
        if not livepeer_header:
            return web.Response(
                status=400,
                text="Missing Livepeer header"
            )
            
        # Decode base64 header
        try:
            decoded_header = base64.b64decode(livepeer_header).decode('utf-8')
            header_data = json.loads(decoded_header)
        except Exception as e:
            logger.error(f"Failed to decode Livepeer header: {e}")
            return web.Response(
                status=400,
                text="Invalid Livepeer header format"
            )
        
        # Get request body
        try:
            request_data = await request.json()
        except Exception as e:
            logger.error(f"Failed to parse request JSON: {e}")
            return web.Response(
                status=400,
                text="Invalid JSON in request body"
            )
        
        # Process the request based on capability
        if capability == "text-reversal":
            # Example text reversal service from BYOC docs
            text = request_data.get('text', '')
            reversed_text = text[::-1]
            result = {
                'original': text,
                'reversed': reversed_text
            }
        elif capability == "comfystream-video":
            # Process video through ComfyStream pipeline
            result = await _process_video_capability(request, request_data, header_data)
        elif capability == "comfystream-image":
            # Process image through ComfyStream pipeline
            result = await _process_image_capability(request, request_data, header_data)
        else:
            return web.Response(
                status=404,
                text=f"Unknown capability: {capability}"
            )
        
        return web.json_response(result)
        
    except Exception as e:
        logger.error(f"Error processing capability request: {e}")
        return web.Response(
            status=500,
            text="An internal server error has occurred. Please try again later."
        )


async def _process_video_capability(request: web.Request, request_data: Dict, header_data: Dict) -> Dict:
    """Process video through ComfyStream pipeline with trickle streaming"""
    
    # Extract prompts and configuration
    prompts = request_data.get('prompts', [])
    width = request_data.get('width', 512)
    height = request_data.get('height', 512)
    stream_url = request_data.get('stream_url', '')
    
    if not prompts:
        raise ValueError("No prompts provided for video processing")
        
    if not stream_url:
        raise ValueError("No stream URL provided for video output")
    
    # Create a pipeline for this request
    pipeline = Pipeline(
        width=width,
        height=height,
        cwd=request.app["workspace"],
        disable_cuda_malloc=True,
        gpu_only=True,
        preview_method='none',
        comfyui_inference_log_level=request.app.get("comfui_inference_log_level", None),
    )
    
    try:
        await pipeline.set_prompts(prompts)
        
        # Create manifest for tracking
        manifest_id = str(uuid.uuid4())
        stream_manifest = StreamManifest(
            manifest_id=manifest_id,
            input_stream_url="",  # Will be set based on the request
            output_stream_url=stream_url,
            created_at=datetime.now(),
            status='starting',
            pipeline=pipeline,
            frame_queue=asyncio.Queue(),
            metadata={
                'width': width,
                'height': height,
                'capability': 'comfystream-video'
            }
        )
        
        # Start the streaming pipeline
        if stream_manifest.frame_queue is not None:
            stream_manifest.publisher_task = asyncio.create_task(
                enhanced_segment_publisher(stream_url, stream_manifest.frame_queue, add_metadata_headers=True)
            )
        
        request.app["active_streams"][manifest_id] = stream_manifest
        stream_manifest.status = 'active'
        
        return {
            'success': True,
            'manifest_id': manifest_id,
            'stream_url': stream_url,
            'message': 'Video processing pipeline started'
        }
        
    except Exception as e:
        await pipeline.cleanup()
        raise e


async def _process_image_capability(request: web.Request, request_data: Dict, header_data: Dict) -> Dict:
    """Process image through ComfyStream pipeline"""
    
    # Extract prompts and configuration
    prompts = request_data.get('prompts', [])
    width = request_data.get('width', 512)
    height = request_data.get('height', 512)
    
    if not prompts:
        raise ValueError("No prompts provided for image processing")
    
    # Create a pipeline for this request
    pipeline = Pipeline(
        width=width,
        height=height,
        cwd=request.app["workspace"],
        disable_cuda_malloc=True,
        gpu_only=True,
        preview_method='none',
        comfyui_inference_log_level=request.app.get("comfui_inference_log_level", None),
    )
    
    try:
        await pipeline.set_prompts(prompts)
        
        # For single image processing, we don't need streaming
        # This would be implemented based on specific image processing needs
        
        return {
            'success': True,
            'message': 'Image processing completed',
            'width': width,
            'height': height
        }
        
    finally:
        await pipeline.cleanup()


async def _process_frames_for_trickle_output(stream_manifest: StreamManifest, trickle_track: TrickleVideoStreamTrack):
    """Process frames from trickle track and encode them for output"""
    try:
        frame_queue = stream_manifest.frame_queue
        if frame_queue is None:
            logger.error("Frame queue is None, cannot process frames for output")
            return
            
        metadata = stream_manifest.metadata or {}
        width = metadata.get('width', 512)
        height = metadata.get('height', 512)
        
        # Initialize encoder for output segments
        segment_encoder = TrickleSegmentEncoder(width=width, height=height)
        
        logger.info(f"Starting trickle output frame processing for stream {stream_manifest.manifest_id}")
        
        frame_count = 0
        processed_frames_batch = []
        batch_size = 30  # Process frames in batches for better encoding efficiency
        
        while stream_manifest.status == 'active':
            try:
                # Get processed frame from the trickle track (which gets it from pipeline)
                processed_frame = await trickle_track.recv()
                processed_frames_batch.append(processed_frame)
                frame_count += 1
                
                # When we have a batch of frames, encode them into a segment
                if len(processed_frames_batch) >= batch_size:
                    # Encode batch of processed frames into output segment
                    output_segment_data = segment_encoder.encode_frames_batch(
                        processed_frames_batch, start_frame_number=frame_count - len(processed_frames_batch)
                    )
                    
                    if output_segment_data:
                        # Put encoded segment into output queue for trickle publisher
                        await frame_queue.put(output_segment_data)
                        logger.debug(f"Encoded output segment with {len(processed_frames_batch)} frames "
                                   f"({len(output_segment_data)} bytes)")
                    
                    # Clear the batch
                    processed_frames_batch = []
                    
                    if frame_count % 150 == 0:  # Log every 150 frames (5 batches)
                        logger.info(f"Processed {frame_count} frames for output stream {stream_manifest.manifest_id}")
                
            except Exception as e:
                logger.error(f"Error processing frame for output: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retrying
        
        # Process any remaining frames in the batch
        if processed_frames_batch:
            output_segment_data = segment_encoder.encode_frames_batch(
                processed_frames_batch, start_frame_number=frame_count - len(processed_frames_batch)
            )
            
            if output_segment_data:
                await frame_queue.put(output_segment_data)
                logger.info(f"Encoded final output segment with {len(processed_frames_batch)} frames")
        
        logger.info(f"Trickle output frame processing finished for stream {stream_manifest.manifest_id}, processed {frame_count} total frames")
        
    except Exception as e:
        logger.error(f"Trickle output frame processor error for stream {stream_manifest.manifest_id}: {e}")
    finally:
        # Signal end of stream
        try:
            if stream_manifest.frame_queue is not None:
                await stream_manifest.frame_queue.put(None)
        except:
            pass


async def start_stream(request: web.Request) -> web.Response:
    """Start a new streaming session with real trickle stream processing"""
    try:
        data = await request.json()
        
        prompts = data.get('prompts', [])
        input_stream_url = data.get('stream_url', '')  # Input trickle stream URL
        width = data.get('width', 512)
        height = data.get('height', 512)
        
        # Optional output stream URL (defaults to auto-generated)
        output_stream_url = data.get('output_stream_url', '')
        
        if not prompts:
            return web.json_response(
                {'error': 'No prompts provided'}, 
                status=400
            )
            
        if not input_stream_url:
            return web.json_response(
                {'error': 'No input stream_url provided'}, 
                status=400
            )
        
        # Create pipeline using the same pattern as app.py
        pipeline = Pipeline(
            width=width,
            height=height,
            cwd=request.app["workspace"],
            disable_cuda_malloc=True,
            gpu_only=True,
            preview_method='none',
            comfyui_inference_log_level=request.app.get("comfui_inference_log_level", None),
        )
        
        # Set prompts
        await pipeline.set_prompts(prompts)
        logger.info("Pipeline prompts set successfully")
        
        # Create manifest
        manifest_id = str(uuid.uuid4())
        
        # Generate output URL if not provided - use simple "-out" suffix
        if not output_stream_url:
            if input_stream_url.endswith('/'):
                base_url = input_stream_url.rstrip('/')
            else:
                base_url = input_stream_url
            
            output_stream_url = f"{base_url}-out"
        
        stream_manifest = StreamManifest(
            manifest_id=manifest_id,
            input_stream_url=input_stream_url,
            output_stream_url=output_stream_url,
            created_at=datetime.now(),
            status='starting',
            pipeline=pipeline,
            frame_queue=asyncio.Queue(),
            metadata={
                'width': width,
                'height': height,
                'prompts': prompts,
                'input_url': input_stream_url,
                'output_url': output_stream_url
            }
        )
        
        # Store the manifest first
        request.app["active_streams"][manifest_id] = stream_manifest
        
        # Start the trickle frame processing using TrickleVideoStreamTrack
        trickle_track = TrickleVideoStreamTrack(input_stream_url, pipeline, width, height)
        
        # Start frame processor task that will get frames from pipeline and encode them for output
        stream_manifest.frame_processor_task = asyncio.create_task(
            _process_frames_for_trickle_output(stream_manifest, trickle_track)
        )
        
        # Start streaming publisher
        if stream_manifest.frame_queue is not None:
            stream_manifest.publisher_task = asyncio.create_task(
                enhanced_segment_publisher(stream_manifest.output_stream_url, stream_manifest.frame_queue, add_metadata_headers=True)
            )
        
        stream_manifest.status = 'active'
        
        logger.info(f"Started stream {manifest_id}: {input_stream_url} → {stream_manifest.output_stream_url}")
        
        return web.json_response({
            'success': True,
            'manifest_id': manifest_id,
            'input_stream_url': input_stream_url,
            'output_stream_url': stream_manifest.output_stream_url
        })
        
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        return web.json_response(
            {'error': f'Failed to start stream: {str(e)}'}, 
            status=500
        )


async def stop_stream(request: web.Request) -> web.Response:
    """Stop a streaming session by manifest ID"""
    manifest_id = request.match_info.get('manifest_id', '')
    
    if manifest_id not in request.app["active_streams"]:
        return web.json_response(
            {'error': 'Stream not found'}, 
            status=404
        )
    
    try:
        stream_manifest = request.app["active_streams"][manifest_id]
        stream_manifest.status = 'stopping'
        
        # Stop the frame processor task first (stops getting processed frames)
        if stream_manifest.frame_processor_task:
            stream_manifest.frame_processor_task.cancel()
            try:
                await stream_manifest.frame_processor_task
            except asyncio.CancelledError:
                pass
        
        # Stop the publisher task (stops output)
        if stream_manifest.publisher_task:
            stream_manifest.publisher_task.cancel()
            try:
                await stream_manifest.publisher_task
            except asyncio.CancelledError:
                pass
        
        # Stop the pipeline
        if stream_manifest.pipeline:
            await stream_manifest.pipeline.cleanup()
        
        # Signal end of stream
        if stream_manifest.frame_queue:
            await stream_manifest.frame_queue.put(None)
        
        stream_manifest.status = 'stopped'
        
        logger.info(f"Stopped stream {manifest_id}")
        
        return web.json_response({
            'success': True,
            'manifest_id': manifest_id,
            'message': 'Stream stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error stopping stream {manifest_id}: {e}")
        return web.json_response(
            {'error': f'Failed to stop stream: {str(e)}'}, 
            status=500
        )


async def get_stream_status(request: web.Request) -> web.Response:
    """Get status of a specific stream"""
    manifest_id = request.match_info.get('manifest_id', '')
    
    if manifest_id not in request.app["active_streams"]:
        return web.json_response(
            {'error': 'Stream not found'}, 
            status=404
        )
    
    stream_manifest = request.app["active_streams"][manifest_id]
    return web.json_response({
        'success': True,
        'stream': stream_manifest.to_dict()
    })


async def list_streams(request: web.Request) -> web.Response:
    """List all active streams"""
    streams = [
        stream_manifest.to_dict() 
        for stream_manifest in request.app["active_streams"].values()
    ]
    
    return web.json_response({
        'success': True,
        'streams': streams,
        'count': len(streams)
    })


async def offer(request):
    pipeline = request.app["pipeline"]
    pcs = request.app["pcs"]

    params = await request.json()

    await pipeline.set_prompts(params["prompts"])

    offer_params = params["offer"]
    offer = RTCSessionDescription(sdp=offer_params["sdp"], type=offer_params["type"])

    ice_servers = get_ice_servers()
    if len(ice_servers) > 0:
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=get_ice_servers())
        )
    else:
        pc = RTCPeerConnection()

    pcs.add(pc)

    tracks = {"video": None, "audio": None}
    
    # Flag to track if we've received resolution update
    resolution_received = {"value": False}

    # Only add video transceiver if video is present in the offer
    if "m=video" in offer.sdp:
        # Prefer h264
        transceiver = pc.addTransceiver("video")
        caps = RTCRtpSender.getCapabilities("video")
        prefs = list(filter(lambda x: x.name == "H264", caps.codecs))
        transceiver.setCodecPreferences(prefs)

        # Monkey patch max and min bitrate to ensure constant bitrate
        h264.MAX_BITRATE = MAX_BITRATE
        h264.MIN_BITRATE = MIN_BITRATE

    # Handle control channel from client
    @pc.on("datachannel")
    def on_datachannel(channel):
        if channel.label == "control":

            @channel.on("message")
            async def on_message(message):
                try:
                    params = json.loads(message)

                    if params.get("type") == "get_nodes":
                        nodes_info = await pipeline.get_nodes_info()
                        response = {"type": "nodes_info", "nodes": nodes_info}
                        channel.send(json.dumps(response))
                    elif params.get("type") == "update_prompts":
                        if "prompts" not in params:
                            logger.warning(
                                "[Control] Missing prompt in update_prompt message"
                            )
                            return
                        try:
                            await pipeline.update_prompts(params["prompts"])
                        except Exception as e:
                            logger.error(f"Error updating prompt: {str(e)}")
                        response = {"type": "prompts_updated", "success": True}
                        channel.send(json.dumps(response))
                    elif params.get("type") == "update_resolution":
                        if "width" not in params or "height" not in params:
                            logger.warning("[Control] Missing width or height in update_resolution message")
                            return
                        # Update pipeline resolution for future frames
                        pipeline.width = params["width"]
                        pipeline.height = params["height"]
                        logger.info(f"[Control] Updated resolution to {params['width']}x{params['height']}")
                        
                        # Mark that we've received resolution
                        resolution_received["value"] = True
                        
                        # Warm the video pipeline with the new resolution
                        if "m=video" in pc.remoteDescription.sdp:
                            await pipeline.warm_video()
                            
                        response = {
                            "type": "resolution_updated",
                            "success": True
                        }
                        channel.send(json.dumps(response))
                    else:
                        logger.warning(
                            "[Server] Invalid message format - missing required fields"
                        )
                except json.JSONDecodeError:
                    logger.error("[Server] Invalid JSON received")
                except Exception as e:
                    logger.error(f"[Server] Error processing message: {str(e)}")

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        if track.kind == "video":
            videoTrack = VideoStreamTrack(track, pipeline)
            tracks["video"] = videoTrack
            sender = pc.addTrack(videoTrack)

            # Store video track in app for stats.
            stream_id = track.id
            request.app["video_tracks"][stream_id] = videoTrack

            codec = "video/H264"
            force_codec(pc, sender, codec)
        elif track.kind == "audio":
            audioTrack = AudioStreamTrack(track, pipeline)
            tracks["audio"] = audioTrack
            pc.addTrack(audioTrack)

        @track.on("ended")
        async def on_ended():
            logger.info(f"{track.kind} track ended")
            request.app["video_tracks"].pop(track.id, None)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
        elif pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)

    # Only warm audio here, video warming happens after resolution update
    if "m=audio" in pc.remoteDescription.sdp:
        await pipeline.warm_audio()
    
    # We no longer warm video here - it will be warmed after receiving resolution

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def cancel_collect_frames(track):
    track.running = False
    if hasattr(track, 'collect_task') is not None and not track.collect_task.done():
        try:
            track.collect_task.cancel()
            await track.collect_task
        except (asyncio.CancelledError):
            pass

async def set_prompt(request):
    pipeline = request.app["pipeline"]

    prompt = await request.json()
    await pipeline.set_prompts(prompt)

    return web.Response(content_type="application/json", text="OK")
    

def health(_):
    return web.Response(content_type="application/json", text="OK")


async def on_startup(app: web.Application):
    if app["media_ports"]:
        patch_loop_datagram(app["media_ports"])

    app["pipeline"] = Pipeline(
        width=512,
        height=512,
        cwd=app["workspace"], 
        disable_cuda_malloc=True, 
        gpu_only=True, 
        preview_method='none',
        comfyui_inference_log_level=app.get("comfui_inference_log_level", None),
    )
    app["pcs"] = set()
    app["video_tracks"] = {}
    # Initialize active streams for trickle streaming
    app["active_streams"] = {}


async def on_shutdown(app: web.Application):
    pcs = app["pcs"]
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    
    # Cleanup active streams
    active_streams = app.get("active_streams", {})
    for manifest_id, stream_manifest in active_streams.items():
        try:
            stream_manifest.status = 'stopping'
            if stream_manifest.publisher_task:
                stream_manifest.publisher_task.cancel()
            if stream_manifest.pipeline:
                await stream_manifest.pipeline.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up stream {manifest_id}: {e}")
    active_streams.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comfystream server")
    parser.add_argument("--port", default=8889, help="Set the signaling port")
    parser.add_argument(
        "--media-ports", default=None, help="Set the UDP ports for WebRTC media"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Set the host")
    parser.add_argument(
        "--workspace", default=None, required=True, help="Set Comfy workspace"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--monitor",
        default=False,
        action="store_true",
        help="Start a Prometheus metrics endpoint for monitoring.",
    )
    parser.add_argument(
        "--stream-id-label",
        default=False,
        action="store_true",
        help="Include stream ID as a label in Prometheus metrics.",
    )
    parser.add_argument(
        "--comfyui-log-level",
        default=None,
        choices=logging._nameToLevel.keys(),
        help="Set the global logging level for ComfyUI",
    )
    parser.add_argument(
        "--comfyui-inference-log-level",
        default=None,
        choices=logging._nameToLevel.keys(),
        help="Set the logging level for ComfyUI inference",
    )
    parser.add_argument(
        "--mode",
        default="webrtc",
        choices=["webrtc", "trickle"],
        help="Set the streaming mode: webrtc (default) or trickle",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    app = web.Application()
    app["media_ports"] = args.media_ports.split(",") if args.media_ports else None
    app["workspace"] = args.workspace
    app["mode"] = args.mode
    
    # Setup CORS
    cors = setup_cors(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"]
        )
    })

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/", health)
    app.router.add_get("/health", health)

    # WebRTC signalling and control routes.
    app.router.add_post("/offer", offer)
    app.router.add_post("/prompt", set_prompt)
    
    # BYOC endpoints for Livepeer compatibility
    cors.add(app.router.add_post("/process/request/{capability}", process_capability_request))
    cors.add(app.router.add_post("/warmup", warmup_pipeline))
    
    # Trickle streaming endpoints
    cors.add(app.router.add_post("/stream/start", start_stream))
    cors.add(app.router.add_delete("/stream/{manifest_id}", stop_stream))
    cors.add(app.router.add_get("/stream/{manifest_id}/status", get_stream_status))
    cors.add(app.router.add_get("/streams", list_streams))
    
    # Setup HTTP streaming routes
    setup_routes(app, cors)
    
    # Serve static files from the public directory
    app.router.add_static("/", path=os.path.join(os.path.dirname(__file__), "public"), name="static")

    # Add routes for getting stream statistics.
    stream_stats_manager = StreamStatsManager(app)
    app.router.add_get(
        "/streams/stats", stream_stats_manager.collect_all_stream_metrics
    )
    app.router.add_get(
        "/stream/{stream_id}/stats", stream_stats_manager.collect_stream_metrics_by_id
    )

    # Add Prometheus metrics endpoint.
    app["metrics_manager"] = MetricsManager(include_stream_id=args.stream_id_label)
    if args.monitor:
        app["metrics_manager"].enable()
        logger.info(
            f"Monitoring enabled - Prometheus metrics available at: "
            f"http://{args.host}:{args.port}/metrics"
        )
        app.router.add_get("/metrics", app["metrics_manager"].metrics_handler)

    # Add hosted platform route prefix.
    # NOTE: This ensures that the local and hosted experiences have consistent routes.
    add_prefix_to_app_routes(app, "/live")

    def force_print(*args, **kwargs):
        print(*args, **kwargs, flush=True)
        sys.stdout.flush()

    # Allow overriding of ComyfUI log levels.
    if args.comfyui_log_level:
        log_level = logging._nameToLevel.get(args.comfyui_log_level.upper())
        logging.getLogger("comfy").setLevel(log_level)
    if args.comfyui_inference_log_level:
        app["comfui_inference_log_level"] = args.comfyui_inference_log_level

    # Print startup information
    mode_info = {
        "webrtc": "WebRTC mode (default) - supports real-time video streaming via WebRTC",
        "trickle": "Trickle mode - supports trickle streaming with BYOC compatibility"
    }
    
    logger.info(f"🚀 ComfyStream Server starting in {args.mode.upper()} mode")
    logger.info(f"Mode: {mode_info.get(args.mode, 'Unknown mode')}")
    logger.info(f"Server will be available at: http://{args.host}:{args.port}")
    
    if args.mode == "webrtc":
        logger.info("📡 WebRTC endpoints available:")
        logger.info(f"  - WebRTC Offer: http://{args.host}:{args.port}/offer")
        logger.info(f"  - Set Prompt: http://{args.host}:{args.port}/prompt")
    
    if args.mode == "trickle" or args.mode == "webrtc":  # Both modes have BYOC support
        logger.info("🔄 BYOC & Trickle endpoints available:")
        logger.info(f"  - BYOC Capability: http://{args.host}:{args.port}/process/request/{{capability}}")
        logger.info(f"  - Pipeline Warmup: http://{args.host}:{args.port}/warmup")
        logger.info(f"  - Start Stream: http://{args.host}:{args.port}/stream/start")
        logger.info(f"  - List Streams: http://{args.host}:{args.port}/streams")
    
    logger.info(f"📊 Health check: http://{args.host}:{args.port}/health")
    
    if args.monitor:
        logger.info(f"📈 Prometheus metrics: http://{args.host}:{args.port}/metrics")

    web.run_app(app, host=args.host, port=int(args.port), print=force_print)
