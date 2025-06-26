import argparse
import asyncio
import json
import logging
import os
import sys
import time
import av
import torch
import uuid
import base64
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

# Initialize CUDA before any other imports to prevent core dump.
if torch.cuda.is_available():
    torch.cuda.init()

from aiohttp import web
from aiohttp_cors import setup as setup_cors, ResourceOptions
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
    TricklePublisher, TrickleSubscriber, high_throughput_segment_publisher,
    TrickleStreamDecoder, TrickleSegmentEncoder
)

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
    encoder: Optional[TrickleSegmentEncoder] = None  # Persistent encoder for timestamp continuity
    
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
        
        # Convert simple text prompts to ComfyUI workflow format if needed
        if isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], str):
            # Convert simple text prompts to basic workflow format
            text_prompt = prompts[0] if prompts else "test"
            prompts = [{
                "5": {
                    "inputs": {
                        "text": text_prompt,
                        "clip": ["23", 0]
                    },
                    "class_type": "CLIPTextEncode",
                    "_meta": {"title": "CLIP Text Encode (Prompt)"}
                },
                "6": {
                    "inputs": {
                        "text": "",
                        "clip": ["23", 0]
                    },
                    "class_type": "CLIPTextEncode", 
                    "_meta": {"title": "CLIP Text Encode (Prompt)"}
                },
                "16": {
                    "inputs": {
                        "width": width,
                        "height": height,
                        "batch_size": 1
                    },
                    "class_type": "EmptyLatentImage",
                    "_meta": {"title": "Empty Latent Image"}
                },
                "23": {
                    "inputs": {
                        "clip_name": "CLIPText/model.fp16.safetensors",
                        "type": "stable_diffusion", 
                        "device": "default"
                    },
                    "class_type": "CLIPLoader",
                    "_meta": {"title": "Load CLIP"}
                }
            }]
            logger.info(f"Converted text prompt '{text_prompt}' to basic workflow format")
        
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
            return web.json_response(result)
        elif capability == "comfystream-video":
            # Process video through ComfyStream pipeline - return response directly
            return await start_stream(request)
        elif capability == "comfystream-image":
            # Process image through ComfyStream pipeline
            result = await _process_image_capability(request, request_data, header_data)
            return web.json_response(result)
        else:
            return web.Response(
                status=404,
                text=f"Unknown capability: {capability}"
            )
        
    except Exception as e:
        logger.error(f"Error processing capability request: {e}")
        return web.Response(
            status=500,
            text="An internal server error has occurred. Please try again later."
        )

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


async def _process_trickle_stream_direct(stream_manifest: StreamManifest):
    """Trickle stream processing using ai-runner pattern with FFmpeg timing management"""
    try:
        pipeline = stream_manifest.pipeline
        frame_queue = stream_manifest.frame_queue
        
        if frame_queue is None or pipeline is None:
            logger.error("Frame queue or pipeline is None, cannot process frames")
            return
            
        metadata = stream_manifest.metadata or {}
        width = metadata.get('width', 512)
        height = metadata.get('height', 512)
        
        logger.info(f"Starting ai-runner pattern trickle processing for stream {stream_manifest.manifest_id}")
        logger.info(f"Input: {stream_manifest.input_stream_url} → Output: {stream_manifest.output_stream_url}")
        
        # Use ai-runner's proven pattern: FFmpeg for decode → process frames → FFmpeg for encode
        await _run_trickle_pipeline_airunner_style(
            stream_manifest.input_stream_url,
            stream_manifest.output_stream_url, 
            pipeline,
            frame_queue,
            width,
            height,
            stream_manifest
        )
        
        logger.info(f"AI-runner pattern trickle processing finished for stream {stream_manifest.manifest_id}")
                        
    except Exception as e:
        logger.error(f"Trickle processor error for stream {stream_manifest.manifest_id}: {e}")
    finally:
        # Signal end of stream
        try:
            if stream_manifest.frame_queue is not None:
                await stream_manifest.frame_queue.put(None)
        except:
            pass


async def _run_trickle_pipeline_airunner_style(input_url: str, output_url: str, pipeline, frame_queue, width: int, height: int, stream_manifest: StreamManifest):
    """Async decoupled trickle pipeline with flow control for smooth output"""
    try:
        import av
        logger.info("Starting async decoupled trickle pipeline with flow control")
        
        # Large queues for maximum flexibility and buffering
        input_segment_queue = asyncio.Queue(maxsize=50)      # ~10+ input segments
        decoded_frame_queue = asyncio.Queue(maxsize=1000)    # ~13+ segments worth - plenty of ComfyUI buffer
        processed_frame_queue = asyncio.Queue(maxsize=800)   # ~10+ segments worth for smooth output
        output_control_queue = asyncio.Queue(maxsize=200)    # Large output buffering
        
        stream_decoder = TrickleStreamDecoder(target_width=width, target_height=height)
        
        # Metrics for monitoring flow
        flow_metrics = {
            'input_segments': 0,
            'decoded_frames': 0, 
            'processed_frames': 0,
            'published_segments': 0,
            'input_queue_depth': 0,
            'decoded_queue_depth': 0,
            'processed_queue_depth': 0,
            'output_queue_depth': 0
        }
        
        # Task 1: Async Trickle Subscriber (continuous input buffering)
        async def trickle_subscriber_task():
            """Continuously fetch and buffer trickle segments for smooth flow"""
            try:
                logger.info("Starting async trickle subscriber with enhanced buffering")
                async with TrickleSubscriber(input_url) as subscriber:
                    segment_count = 0
                    consecutive_empty = 0
                    max_consecutive_empty = 50  # Allow more retries
                    
                    while stream_manifest.status == 'active':
                        try:
                            # Fetch segment with backoff on empty
                            current_segment = await subscriber.next()
                            if current_segment is None:
                                consecutive_empty += 1
                                if consecutive_empty >= max_consecutive_empty:
                                    logger.info("No more segments available, ending subscriber")
                                    break
                                # Progressive backoff for empty segments
                                sleep_time = min(0.1 * consecutive_empty, 2.0)
                                await asyncio.sleep(sleep_time)
                                continue
                            
                            consecutive_empty = 0  # Reset counter on successful fetch
                            segment_count += 1
                            
                            # Read segment data
                            segment_data = await _read_complete_segment(current_segment)
                            if segment_data:
                                # Queue segment with priority handling
                                try:
                                    await asyncio.wait_for(
                                        input_segment_queue.put((segment_count, segment_data)), 
                                        timeout=0.5  # Longer timeout for buffering
                                    )
                                    flow_metrics['input_segments'] += 1
                                    flow_metrics['input_queue_depth'] = input_segment_queue.qsize()
                                    
                                    if segment_count % 10 == 0:
                                        logger.debug(f"Buffered segment {segment_count} (queue depth: {flow_metrics['input_queue_depth']})")
                                        
                                except asyncio.TimeoutError:
                                    # If queue is full, wait longer rather than dropping
                                    logger.warning(f"Input queue full, waiting for space...")
                                    await input_segment_queue.put((segment_count, segment_data))
                            
                            # Close segment
                            if hasattr(current_segment, 'close'):
                                try:
                                    await current_segment.close()
                                except:
                                    pass
                                    
                        except Exception as e:
                            logger.error(f"Error in subscriber: {e}")
                            await asyncio.sleep(0.2)
                            
                logger.info(f"Trickle subscriber finished: {segment_count} segments buffered")
                            
            except Exception as e:
                logger.error(f"Trickle subscriber task error: {e}")
            finally:
                # Signal end of input
                await input_segment_queue.put(None)
        
        # Task 2: Async Decoder (continuous frame decoding)
        async def decoder_task():
            """Decode segments into frames maintaining large buffer"""
            try:
                logger.info("Starting async decoder with enhanced buffering")
                total_frames = 0
                
                while True:
                    try:
                        # Get segment from queue with longer timeout
                        segment_item = await asyncio.wait_for(
                            input_segment_queue.get(), 
                            timeout=5.0  # Longer timeout for flow
                        )
                        
                        if segment_item is None:
                            break  # End of input
                        
                        segment_count, segment_data = segment_item
                        flow_metrics['input_queue_depth'] = input_segment_queue.qsize()
                        
                        # Decode segment
                        decoded_frames = stream_decoder.process_segment(segment_data)
                        
                        if decoded_frames:
                            # Queue each frame with preserved timing and segment frame count
                            total_frames_in_segment = len(decoded_frames)
                            for i, frame in enumerate(decoded_frames):
                                frame_with_timing = {
                                    'frame': frame,
                                    'original_pts': frame.pts,
                                    'original_time_base': frame.time_base,
                                    'segment_id': segment_count,
                                    'frame_index_in_segment': i,
                                    'expected_frames_in_segment': total_frames_in_segment  # Add expected count
                                }
                                
                                try:
                                    # Large queue should accommodate ComfyUI processing variations
                                    queue_size = decoded_frame_queue.qsize()
                                    if queue_size > 900:  # High usage warning for 1000-frame queue
                                        logger.warning(f"Decoded frame queue high usage ({queue_size}/1000)")
                                    
                                    await asyncio.wait_for(
                                        decoded_frame_queue.put(frame_with_timing),
                                        timeout=2.0  # Longer timeout with large queue
                                    )
                                    total_frames += 1
                                    flow_metrics['decoded_frames'] += 1
                                    flow_metrics['decoded_queue_depth'] = decoded_frame_queue.qsize()
                                    
                                except asyncio.TimeoutError:
                                    # With large queue, this should be rare - wait rather than drop
                                    logger.warning(f"Decoded frame queue full, waiting... (queue: {decoded_frame_queue.qsize()}/1000)")
                                    await decoded_frame_queue.put(frame_with_timing)
                                    total_frames += 1
                            
                            if segment_count % 5 == 0:
                                logger.debug(f"Decoded segment {segment_count}: {len(decoded_frames)} frames (total: {total_frames}, queue: {flow_metrics['decoded_queue_depth']})")
                        
                    except asyncio.TimeoutError:
                        if stream_manifest.status != 'active':
                            break
                        continue
                    except Exception as e:
                        logger.error(f"Error in decoder: {e}")
                        continue
                
                logger.info(f"Decoder finished: {total_frames} frames decoded")
                
            except Exception as e:
                logger.error(f"Decoder task error: {e}")
            finally:
                await decoded_frame_queue.put(None)
        
        # Task 3: Async Pipeline Processor (managed AI processing)
        async def pipeline_processor_task():
            """Process frames through ComfyUI with flow management and queue monitoring"""
            try:
                logger.info("Starting async pipeline processor with ComfyUI queue monitoring")
                processed_count = 0
                processing_batch_size = 3  # Smaller batches to prevent blocking
                current_batch = []
                last_queue_warning = 0
                
                while True:
                    try:
                        # Monitor queue sizes to detect ComfyUI blocking
                        decoded_queue_size = decoded_frame_queue.qsize()
                        processed_queue_size = processed_frame_queue.qsize()
                        
                        # Monitor large queues (warn at higher thresholds)
                        if decoded_queue_size > 800 and time.time() - last_queue_warning > 10.0:
                            logger.warning(f"High queue usage: decoded_queue={decoded_queue_size}/1000, "
                                         f"processed_queue={processed_queue_size}/800")
                            last_queue_warning = time.time()
                        
                        # Get frame from decoder queue with very short timeout to prevent blocking
                        frame_item = await asyncio.wait_for(
                            decoded_frame_queue.get(),
                            timeout=0.1  # Very short timeout to detect blocking immediately
                        )
                        
                        if frame_item is None:
                            # Process any remaining batch
                            if current_batch:
                                await _process_frame_batch(current_batch, pipeline, processed_frame_queue, flow_metrics)
                                processed_count += len(current_batch)
                            break
                        
                        flow_metrics['decoded_queue_depth'] = decoded_frame_queue.qsize()
                        current_batch.append(frame_item)
                        
                        # Process in smaller batches to maintain flow
                        if len(current_batch) >= processing_batch_size:
                            await _process_frame_batch(current_batch, pipeline, processed_frame_queue, flow_metrics)
                            processed_count += len(current_batch)
                            current_batch = []
                            
                            # Brief pause to allow other tasks to work
                            await asyncio.sleep(0.01)
                        
                        # Let queue fill naturally - no emergency processing that adds chaos
                        
                        # Also process batch if we have any frames and queue is not completely full
                        elif len(current_batch) > 0 and decoded_frame_queue.qsize() < 100:
                            await _process_frame_batch(current_batch, pipeline, processed_frame_queue, flow_metrics)
                            processed_count += len(current_batch)
                            current_batch = []
                        
                    except asyncio.TimeoutError:
                        # PATIENT TIMEOUT PROCESSING: Only intervene when ComfyUI is truly blocked
                        decoded_queue_size = decoded_frame_queue.qsize()
                        processed_queue_size = processed_frame_queue.qsize()
                        
                        # NATURAL QUEUE MANAGEMENT: Let ComfyUI work at its own pace
                        # Log queue status for monitoring but don't intervene with aggressive dropping
                        if decoded_queue_size > 800:
                            logger.debug(f"ComfyUI processing: decoded={decoded_queue_size}/1000, processed={processed_queue_size}/800")
                        
                        # Process current batch on timeout to maintain flow
                        if current_batch:
                            await _process_frame_batch(current_batch, pipeline, processed_frame_queue, flow_metrics)
                            processed_count += len(current_batch)
                            current_batch = []
                        
                        if stream_manifest.status != 'active':
                            break
                        continue
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        current_batch = []  # Reset batch on error
                        continue
                
                logger.info(f"Pipeline processor finished: {processed_count} frames processed")
                
            except Exception as e:
                logger.error(f"Pipeline processor task error: {e}")
            finally:
                await processed_frame_queue.put(None)
        
        # Task 4: Segment-Aware Output Controller (maintains input/output correspondence)
        async def output_flow_controller():
            """Control output with perfect segment correspondence and buffer warmup"""
            try:
                logger.info("Starting segment-aware output controller with buffer warmup strategy")
                
                # Track segments by input segment ID to maintain correspondence
                active_segments = {}  # segment_id -> {'frames': [], 'published': bool, 'start_time': float}
                completed_segments = []
                
                # No buffer warmup needed with large queues - start publishing immediately
                expected_frames_per_segment = 72  # 24 FPS * 3 seconds
                warmup_complete = True  # Always ready to publish with large queues
                
                logger.info("No buffer warmup delay - starting immediate publishing with large queue buffers")
                
                while True:
                    try:
                        # Get processed frame with metadata (very short timeout for aggressive flow)
                        frame_item = await asyncio.wait_for(
                            processed_frame_queue.get(),
                            timeout=1.0  # Very short timeout for maximum responsiveness
                        )
                        
                        if frame_item is None:
                            # Publish any remaining segments in order (only unpublished ones)
                            for segment_id in sorted(active_segments.keys()):
                                segment_info = active_segments[segment_id]
                                if segment_info['frames'] and not segment_info.get('published', False):
                                    await _segment_aware_publish(
                                        segment_info['frames'], stream_manifest.encoder,
                                        output_control_queue, segment_id
                                    )
                                    segment_info['published'] = True
                                    logger.info(f"Final segment {segment_id}: {len(segment_info['frames'])} frames")
                            break
                        
                        frame = frame_item['frame']
                        input_segment_id = frame_item['segment_id']
                        flow_metrics['processed_queue_depth'] = processed_frame_queue.qsize()
                        
                        # Initialize segment tracking if new
                        if input_segment_id not in active_segments:
                            active_segments[input_segment_id] = {
                                'frames': [],
                                'published': False,  # Track if this segment has been published
                                'start_time': time.time()  # Track when segment started for timeout publishing
                            }
                            logger.debug(f"Started tracking segment {input_segment_id}")
                        
                        # Skip if already published to prevent duplicates
                        if active_segments[input_segment_id].get('published', False):
                            logger.debug(f"Skipping frame for already published segment {input_segment_id}")
                            continue
                        
                        # Add frame to the corresponding segment
                        active_segments[input_segment_id]['frames'].append(frame)
                        current_count = len(active_segments[input_segment_id]['frames'])
                        
                        # Track expected frame count from decoder metadata
                        expected_frame_count = frame_item.get('expected_frames_in_segment', None)
                        
                        # With large queues, no warmup delay needed - always ready to publish
                        total_buffered_frames = processed_frame_queue.qsize()
                        
                        # ADAPTIVE STREAMING OPTIMIZED: Generous tolerance for natural frame variation
                        # Expected: ~72 frames per segment at 24 FPS * 3 seconds
                        expected_frames_24fps = 72
                        tolerance = 0.5  # 50% tolerance for natural frame count variation
                        min_frames = max(36, int(expected_frames_24fps * (1 - tolerance)))  # ~36 frames minimum (1.5 seconds)
                        max_frames = int(expected_frames_24fps * (1 + tolerance))  # ~108 frames maximum (4.5 seconds)
                        
                        segment_ready = False
                        if expected_frame_count is not None:
                            # CRITICAL: Use decoder's exact frame count - this maintains input stream correspondence
                            segment_ready = (current_count >= expected_frame_count)
                            if segment_ready:
                                logger.debug(f"Segment {input_segment_id} ready (EXACT INPUT MATCH): {current_count}/{expected_frame_count} frames")
                        else:
                            # NO TIMEOUT PUBLISHING - wait for input stream to determine natural segment boundaries
                            # This prevents PTS discontinuities and timing gaps
                            segment_ready = False
                            logger.debug(f"Segment {input_segment_id} waiting for input stream boundary: {current_count} frames collected")
                        
                        # NATURAL ACCUMULATION: Let segments reach their natural size
                        # Remove emergency publishing - trust the input stream boundaries
                        
                        # REMOVE TIMEOUT PUBLISHING - it breaks input stream correspondence and PTS tracking
                        # The input stream decoder should provide exact frame counts, timeout interferes with this
                        
                        # With large queues (800+ frames), no buffer maintenance delays needed
                        
                        if segment_ready and current_count > 0 and not active_segments[input_segment_id].get('published', False):
                            segment_frames = active_segments[input_segment_id]['frames']
                            await _segment_aware_publish(
                                segment_frames, stream_manifest.encoder,
                                output_control_queue, input_segment_id
                            )
                            
                            # Mark as published to prevent duplicates
                            active_segments[input_segment_id]['published'] = True
                            
                            logger.info(f"Published segment {input_segment_id}: {len(segment_frames)} frames "
                                      f"(buffer: {processed_frame_queue.qsize()} frames remaining)")
                            completed_segments.append(input_segment_id)
                        
                    except asyncio.TimeoutError:
                        # WAIT FOR INPUT STREAM BOUNDARIES: No timeout publishing that breaks correspondence
                        # Only publish segments that have completed naturally from input stream
                        
                        for segment_id in list(active_segments.keys()):
                            segment_info = active_segments[segment_id]
                            frame_count = len(segment_info['frames'])
                            
                            # PATIENT WAITING: Trust the input stream to provide boundaries
                            # No emergency publishing - let segments complete naturally
                            logger.debug(f"Waiting for input boundary: segment {segment_id} has {frame_count} frames")
                        
                        if stream_manifest.status != 'active':
                            break
                        continue
                    except Exception as e:
                        logger.error(f"Error in segment-aware controller: {e}")
                        continue
                
                logger.info(f"Segment-aware controller finished: {len(completed_segments)} unique segments with perfect correspondence")
                
            except Exception as e:
                logger.error(f"Segment-aware controller error: {e}")
            finally:
                await output_control_queue.put(None)
        
        # Task 5: Adaptive Buffered Publisher (builds segment buffer and adapts to ComfyUI throughput)
        async def buffered_publisher_task():
            """Adaptive buffered publishing with throughput monitoring and FPS adaptation"""
            try:
                logger.info("Starting adaptive buffered publisher with throughput monitoring")
                
                segment_buffer = {}  # segment_id -> segment_data (ready segments)
                published_segments = set()  # Track published segment IDs to prevent duplicates
                next_publish_id = None  # Next segment ID to publish (sequential)
                buffer_target = 1  # Target buffer size (3 seconds ahead)
                buffer_warmup = 0  # No warmup - immediate publishing
                published_count = 0
                playback_started = False
                
                # Adaptive throughput monitoring
                segment_times = []  # Track processing times for throughput calculation
                adaptive_fps = 24.0  # Start with target FPS, adapt based on actual throughput
                last_throughput_check = time.time()
                throughput_check_interval = 15.0  # Check every 15 seconds
                
                logger.info(f"Adaptive buffer strategy: warmup={buffer_warmup} segments, target={buffer_target} segments, initial_fps={adaptive_fps}")
                
                while True:
                    try:
                        # Get completed segment from output controller
                        segment_item = await asyncio.wait_for(
                            output_control_queue.get(),
                            timeout=2.0
                        )
                        
                        if segment_item is None:
                            # Publish all remaining buffered segments in order
                            if segment_buffer:
                                logger.info(f"Stream ending: publishing {len(segment_buffer)} remaining buffered segments")
                                for segment_id in sorted(segment_buffer.keys()):
                                    if segment_id not in published_segments:
                                        await frame_queue.put((segment_id, segment_buffer[segment_id]))
                                        published_segments.add(segment_id)
                                        published_count += 1
                            break
                        
                        # Handle tuple format (segment_id, segment_data)
                        if isinstance(segment_item, tuple) and len(segment_item) == 2:
                            segment_id, segment_data = segment_item
                            
                            # Skip duplicates
                            if segment_id in published_segments or segment_id in segment_buffer:
                                logger.debug(f"Skipping duplicate segment {segment_id}")
                                continue
                            
                            # Add to buffer with timing tracking
                            segment_buffer[segment_id] = segment_data
                            current_time = time.time()
                            segment_times.append(current_time)
                            
                            # Set the starting point for sequential publishing
                            if next_publish_id is None:
                                next_publish_id = segment_id
                                logger.info(f"Starting segment sequence at ID {next_publish_id}")
                            
                            buffer_size = len(segment_buffer)
                            logger.debug(f"Buffered segment {segment_id} (buffer: {buffer_size}/{buffer_target} segments)")
                            
                            # ADAPTIVE THROUGHPUT MONITORING: Check if ComfyUI throughput requires FPS adaptation
                            if current_time - last_throughput_check > throughput_check_interval and len(segment_times) >= 5:
                                # Calculate actual segment processing rate over last period
                                recent_segments = [t for t in segment_times if t > last_throughput_check]
                                if len(recent_segments) >= 3:
                                    time_span = recent_segments[-1] - recent_segments[0]
                                    segments_per_second = (len(recent_segments) - 1) / time_span if time_span > 0 else 0
                                    
                                    # Each segment should be 3 seconds at target FPS
                                    # If we're getting segments slower, ComfyUI can't keep up with target FPS
                                    expected_segments_per_second = 1.0 / 3.0  # 1 segment every 3 seconds at full speed
                                    throughput_ratio = segments_per_second / expected_segments_per_second if expected_segments_per_second > 0 else 1.0
                                    
                                    # Calculate what FPS ComfyUI can actually sustain
                                    sustainable_fps = adaptive_fps * throughput_ratio
                                    
                                    # Adapt if ComfyUI is significantly slower than target
                                    if throughput_ratio < 0.8 and sustainable_fps < adaptive_fps:
                                        old_fps = adaptive_fps
                                        adaptive_fps = max(12.0, sustainable_fps)  # Never go below 12 FPS
                                        
                                        # Adjust buffer strategy for lower throughput
                                        if adaptive_fps < 20:
                                            buffer_target = 2  # Slightly larger buffer for slower throughput
                                            buffer_warmup = 1
                                        
                                        logger.warning(f"🔄 THROUGHPUT ADAPTATION: ComfyUI throughput {throughput_ratio:.2f}x target")
                                        logger.warning(f"📉 FPS ADAPTATION: {old_fps:.1f} → {adaptive_fps:.1f} FPS (buffer: {buffer_target} segments)")
                                    
                                    elif throughput_ratio > 1.1 and sustainable_fps > adaptive_fps:
                                        # ComfyUI is faster than expected, can increase FPS
                                        old_fps = adaptive_fps
                                        adaptive_fps = min(24.0, sustainable_fps)  # Cap at original target
                                        
                                        logger.info(f"📈 FPS IMPROVEMENT: {old_fps:.1f} → {adaptive_fps:.1f} FPS")
                                    
                                    last_throughput_check = current_time
                                    
                                    # Clean old timing data
                                    cutoff_time = current_time - 60.0  # Keep last 60 seconds
                                    segment_times = [t for t in segment_times if t > cutoff_time]
                            
                            # BUFFER WARMUP: Start immediately or when minimum buffer reached
                            if not playback_started and (buffer_warmup == 0 or buffer_size >= buffer_warmup):
                                playback_started = True
                                if buffer_warmup == 0:
                                    logger.info(f"🚀 IMMEDIATE PLAYBACK: Starting with no warmup delay")
                                else:
                                    logger.info(f"🚀 PLAYBACK STARTING: Buffer warmed up with {buffer_size} segments")
                            
                            # BUFFERED PUBLISHING: Maintain target buffer size
                            while playback_started and next_publish_id in segment_buffer and len(segment_buffer) > buffer_target:
                                # Publish the next sequential segment
                                segment_data_to_publish = segment_buffer.pop(next_publish_id)
                                published_segments.add(next_publish_id)
                                
                                await frame_queue.put((next_publish_id, segment_data_to_publish))
                                published_count += 1
                                
                                remaining_buffer = len(segment_buffer)
                                logger.info(f"📺 PUBLISHED segment {next_publish_id} (buffer: {remaining_buffer}/{buffer_target} segments remaining)")
                                
                                # Move to next segment in sequence
                                next_publish_id += 1
                                
                                # Brief pause to prevent overwhelming the publisher
                                await asyncio.sleep(0.1)
                            
                            # CONTINUOUS PUBLISHING: Once warmed up, publish ready segments to maintain buffer
                            if playback_started and next_publish_id in segment_buffer:
                                # Always publish if we have the next sequential segment ready
                                segment_data_to_publish = segment_buffer.pop(next_publish_id)
                                published_segments.add(next_publish_id)
                                
                                await frame_queue.put((next_publish_id, segment_data_to_publish))
                                published_count += 1
                                
                                remaining_buffer = len(segment_buffer)
                                logger.debug(f"📺 CONTINUOUS: Published segment {next_publish_id} (buffer: {remaining_buffer} segments)")
                                next_publish_id += 1
                        
                        else:
                            # Fallback for old format (should not happen with our setup)
                            await frame_queue.put(segment_item)
                            published_count += 1
                            logger.debug(f"Published segment {published_count} (fallback format)")
                        
                        flow_metrics['published_segments'] = published_count
                        flow_metrics['segment_buffer_size'] = len(segment_buffer)
                        flow_metrics['output_queue_depth'] = output_control_queue.qsize()
                        
                        # Log buffer status periodically
                        if published_count % 5 == 0 and published_count > 0:
                            logger.info(f"📊 Buffer status: {published_count} published, {len(segment_buffer)} buffered, next_id={next_publish_id}")
                        
                    except asyncio.TimeoutError:
                        # During timeout, check if we can publish any ready segments
                        if playback_started and next_publish_id in segment_buffer:
                            segment_data_to_publish = segment_buffer.pop(next_publish_id)
                            published_segments.add(next_publish_id)
                            
                            await frame_queue.put((next_publish_id, segment_data_to_publish))
                            published_count += 1
                            
                            logger.debug(f"📺 TIMEOUT: Published segment {next_publish_id} (buffer: {len(segment_buffer)} segments)")
                            next_publish_id += 1
                        
                        if stream_manifest.status != 'active':
                            break
                        continue
                    except Exception as e:
                        logger.error(f"Error in buffered publisher: {e}")
                        continue
                
                logger.info(f"Buffered publisher finished: {published_count} segments published with buffer strategy")
                
            except Exception as e:
                logger.error(f"Buffered publisher error: {e}")
            finally:
                await frame_queue.put(None)
        
        # Start all async tasks with enhanced flow control
        logger.info("Starting all async tasks with flow control")
        
        tasks = await asyncio.gather(
            asyncio.create_task(trickle_subscriber_task()),
            asyncio.create_task(decoder_task()),
            asyncio.create_task(pipeline_processor_task()),
            asyncio.create_task(output_flow_controller()),
            asyncio.create_task(buffered_publisher_task()),
            return_exceptions=True
        )
        
        # Log task completion and final metrics
        task_names = ["subscriber", "decoder", "pipeline", "flow_controller", "publisher"]
        for i, result in enumerate(tasks):
            if isinstance(result, Exception):
                logger.error(f"Task {task_names[i]} failed: {result}")
            else:
                logger.info(f"Task {task_names[i]} completed successfully")
        
        logger.info(f"Final flow metrics: {flow_metrics}")
        logger.info("Async flow-controlled trickle pipeline finished")
        
    except Exception as e:
        logger.error(f"Flow-controlled pipeline error: {e}")


async def _process_frame_batch(batch, pipeline, output_queue, metrics):
    """Process a batch of frames through pipeline with ComfyUI blocking protection"""
    try:
        for frame_item in batch:
            frame = frame_item['frame']
            original_pts = frame_item['original_pts']
            original_time_base = frame_item['original_time_base']
            segment_id = frame_item['segment_id']
            expected_frames = frame_item.get('expected_frames_in_segment', None)
            
            try:
                # Process through pipeline with patient timeout to allow ComfyUI to work
                await asyncio.wait_for(pipeline.put_video_frame(frame), timeout=2.0)  # More time for queuing
                processed_frame = await asyncio.wait_for(pipeline.get_processed_video_frame(), timeout=10.0)  # Much more time for processing
                
                # Restore timing
                processed_frame.pts = original_pts
                processed_frame.time_base = original_time_base
                
                # Queue with flow info including expected frame count
                processed_with_timing = {
                    'frame': processed_frame,
                    'segment_id': segment_id,
                    'processing_batch': True,
                    'expected_frames_in_segment': expected_frames
                }
                
                await output_queue.put(processed_with_timing)
                metrics['processed_frames'] += 1
                metrics['processed_queue_depth'] = output_queue.qsize()
                
            except asyncio.TimeoutError:
                # ComfyUI is severely blocked (>10s per frame) - this indicates a real problem
                logger.error(f"ComfyUI severely blocked (>10s timeout), using original frame for segment {segment_id}")
                
                # Try to clear the pipeline state in case it's stuck
                try:
                    # Attempt to flush the pipeline (non-blocking)
                    while True:
                        try:
                            await asyncio.wait_for(pipeline.get_processed_video_frame(), timeout=0.1)
                        except:
                            break
                except:
                    pass
                
                # Use original frame as fallback only after attempting pipeline recovery
                frame.pts = original_pts
                frame.time_base = original_time_base
                
                processed_with_timing = {
                    'frame': frame,  # Use original frame
                    'segment_id': segment_id,
                    'processing_batch': True,
                    'bypass_comfyui': True,  # Mark as bypassed
                    'expected_frames_in_segment': expected_frames
                }
                
                await output_queue.put(processed_with_timing)
                metrics['processed_frames'] += 1
                metrics['bypassed_frames'] = metrics.get('bypassed_frames', 0) + 1
                metrics['processed_queue_depth'] = output_queue.qsize()
                
                # Add a brief pause to prevent cascade failures
                await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error processing frame batch: {e}")


async def _segment_aware_publish(frames, encoder, output_queue, segment_id):
    """Publish segment with segment ID preservation"""
    try:
        if not frames or not encoder:
            return
        
        # Encode with preserved timing
        segment_data = encoder.encode_frames_batch(frames, segment_id)
        
        if segment_data:
            # Pass segment data with ID for correspondence
            await output_queue.put((segment_id, segment_data))
            logger.debug(f"Segment-aware publish: segment {segment_id}, {len(frames)} frames, {len(segment_data)} bytes")
        else:
            logger.error(f"Failed to encode segment {segment_id}")
            
    except Exception as e:
        logger.error(f"Error in segment-aware publish {segment_id}: {e}")


# Old complex helper functions removed - using simplified approach

async def _read_complete_segment(segment_reader) -> bytes:
    """Read the complete trickle segment data"""
    try:
        complete_data = b""
        while True:
            chunk = await segment_reader.read(8192)
            if not chunk:
                break
            complete_data += chunk
        return complete_data
    except Exception as e:
        logger.error(f"Error reading segment: {e}")
        return b""


async def start_stream(request: web.Request) -> web.Response:
    """Start a new streaming session with real trickle stream processing"""
    try:
        data = await request.json()
        
        prompts = data.get('prompts', [])
        input_stream_url = data.get('stream_url', '')  # Input trickle stream URL
        width = data.get('width', 512)
        height = data.get('height', 512)
        manifest_id = data.get('gateway_request_id', str(uuid.uuid4()))
        
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
        # manifest_id =
        
        # Generate output URL if not provided - use simple "-out" suffix
        if not output_stream_url:
            if input_stream_url.endswith('/'):
                base_url = input_stream_url.rstrip('/')
            else:
                base_url = input_stream_url
            
            output_stream_url = f"{base_url}-out"
        
        # Create persistent encoder for timestamp continuity
        persistent_encoder = TrickleSegmentEncoder(
            width=width,
            height=height,
            fps=24,  # Match segmenter FPS
            format="mpegts",
            video_codec="libx264"
        )
        
        stream_manifest = StreamManifest(
            manifest_id=manifest_id,
            input_stream_url=input_stream_url,
            output_stream_url=output_stream_url,
            created_at=datetime.now(),
            status='starting',
            pipeline=pipeline,
            frame_queue=asyncio.Queue(),
            encoder=persistent_encoder,  # Store persistent encoder
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
        
        # Start the simplified trickle frame processing
        stream_manifest.frame_processor_task = asyncio.create_task(
            _process_trickle_stream_direct(stream_manifest)
        )
        
        # Start streaming publisher with timing control for smooth playback
        # Should match segmenter FPS
        if stream_manifest.frame_queue is not None:
            stream_manifest.publisher_task = asyncio.create_task(
                high_throughput_segment_publisher(
                    stream_manifest.output_stream_url, 
                    stream_manifest.frame_queue, 
                    max_fps=24.0,
                    skip_frame_on_backlog=True
                )
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

    tracks: Dict[str, Optional[Any]] = {"video": None, "audio": None}
    
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
    

async def health(_):
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
    # Simple text reversal endpoint for testing
    
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
        if log_level is not None:
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
