import argparse
import asyncio
import json
import logging
import os
import sys
import time
import secrets
import torch
import av
from typing import Dict, Optional, Union

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
# Import WHIP handler
from whip_handler import setup_whip_routes
# Import WHEP handler
from whep_handler import setup_whep_routes
from aiortc.codecs import h264
from aiortc.rtcrtpsender import RTCRtpSender
from comfystream.pipeline import Pipeline
from twilio.rest import Client
from comfystream.server.utils import patch_loop_datagram, add_prefix_to_app_routes, FPSMeter
from comfystream.server.metrics import MetricsManager, StreamStatsManager
import time

# used for testing default warmup prompts
from comfystream.prompts import DEFAULT_PROMPT, INVERTED_PROMPT, DEFAULT_SD_PROMPT
CURRENT_PROMPT = DEFAULT_PROMPT

logger = logging.getLogger(__name__)
logging.getLogger("aiortc.rtcrtpsender").setLevel(logging.WARNING)
logging.getLogger("aiortc.rtcreceiver").setLevel(logging.WARNING)


MAX_BITRATE = 8000000  # 8 Mbps for better quality
MIN_BITRATE = 4000000  # 4 Mbps minimum


class VideoStreamTrack(MediaStreamTrack):
    """video stream track that processes video frames using a pipeline.

    Attributes:
        kind (str): The kind of media, which is "video" for this class.
        track (MediaStreamTrack): The underlying media stream track.
        pipeline (Pipeline): The processing pipeline to apply to each video frame.
    """

    kind = "video"

    def __init__(self, track: MediaStreamTrack, pipeline: Pipeline, app: web.Application):
        """Initialize the VideoStreamTrack.

        Args:
            track: The underlying media stream track.
            pipeline: The processing pipeline to apply to each video frame.
            app: The aiohttp application instance.
        """
        super().__init__()
        self.track = track
        self.pipeline = pipeline
        self.app = app
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
                    # Check if frame is a VideoFrame before passing to pipeline
                    if isinstance(frame, av.VideoFrame):
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

        # Update WHEP stream manager with the processed frame
        try:
            if 'whep_handler' in self.app and self.app['whep_handler'].stream_manager:
                await self.app['whep_handler'].stream_manager.update_video_frame(processed_frame)
        except Exception as e:
            # Don't let WHEP errors affect the main pipeline
            print(f"Error updating WHEP stream manager: {e}")

        # Increment the frame count to calculate FPS.
        await self.fps_meter.increment_frame_count()

        return processed_frame


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track: MediaStreamTrack, pipeline, app: web.Application):
        super().__init__()
        self.track = track
        self.pipeline = pipeline
        self.app = app
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
                    # Check if frame is an AudioFrame before passing to pipeline
                    if isinstance(frame, av.AudioFrame):
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
        processed_frame = await self.pipeline.get_processed_audio_frame()
        
        # Update WHEP stream manager with the processed audio frame
        try:
            if 'whep_handler' in self.app and self.app['whep_handler'].stream_manager:
                await self.app['whep_handler'].stream_manager.update_audio_frame(processed_frame)
        except Exception as e:
            # Don't let WHEP errors affect the main pipeline
            print(f"Error updating WHEP audio stream manager: {e}")
        
        return processed_frame


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

    # Use a smaller, curated list of reliable STUN servers for faster ICE gathering
    default_stun_servers = [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
    ]
    
    for stun_url in default_stun_servers:
        stun_server = RTCIceServer(urls=[stun_url])
        ice_servers.append(stun_server)

    # Add Twilio TURN servers if available
    token = get_twilio_token()
    if token is not None and hasattr(token, 'ice_servers') and token.ice_servers:
        # Use Twilio TURN servers
        for server in token.ice_servers:
            if isinstance(server, dict) and server.get("url", "").startswith("turn:"):
                turn = RTCIceServer(
                    urls=[server.get("urls", "")],
                    credential=server.get("credential", ""),
                    username=server.get("username", ""),
                )
                ice_servers.append(turn)

    return ice_servers


def create_offer_handler(app):
    """Create the offer handler with app context."""
    
    async def offer(request):
        pipeline = request.app["pipeline"]
        pcs = request.app["pcs"]

        params = await request.json()
        prompts = params.get("prompts", CURRENT_PROMPT)
        prompts = params["prompts"]
        await pipeline.set_prompts(prompts)

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

        tracks: Dict[str, Union[VideoStreamTrack, AudioStreamTrack, None]] = {"video": None, "audio": None}
        
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
                            
                            # Warm the video pipeline with the new resolution only if requested
                            if request.app.get("warm_pipeline", False) and "m=video" in pc.remoteDescription.sdp:
                                await pipeline.warm_video()
                                logger.info(f"[Control] Pipeline warmed with new resolution {params['width']}x{params['height']}")
                            elif "m=video" in pc.remoteDescription.sdp:
                                logger.info(f"[Control] Pipeline warming skipped for resolution update (use --warm-pipeline to enable)")
                            
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
                videoTrack = VideoStreamTrack(track, pipeline, app)
                tracks["video"] = videoTrack
                sender = pc.addTrack(videoTrack)

                # Store video track in app for stats.
                stream_id = track.id
                request.app["video_tracks"][stream_id] = videoTrack

                codec = "video/H264"
                force_codec(pc, sender, codec)
            elif track.kind == "audio":
                audioTrack = AudioStreamTrack(track, pipeline, app)
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

        # Only warm audio here if requested, video warming happens after resolution update
        if request.app.get("warm_pipeline", False) and "m=audio" in pc.remoteDescription.sdp:
            await pipeline.warm_audio()
            logger.info("[WebRTC] Audio pipeline warmed")
        elif "m=audio" in pc.remoteDescription.sdp:
            logger.info("[WebRTC] Audio pipeline warming skipped (use --warm-pipeline to enable)")
        
        # We no longer warm video here - it will be warmed after receiving resolution

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )
    
    return offer


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
        width=app["video_width"],
        height=app["video_height"],
        cwd=app["workspace"], 
        disable_cuda_malloc=True, 
        gpu_only=True, 
        preview_method='none',
        comfyui_inference_log_level=app.get("comfui_inference_log_level", None),
    )
    
    # Set default prompts for the pipeline using the pattern from main.py
    try:
        import json
        await app["pipeline"].set_prompts(json.loads(CURRENT_PROMPT))
        logger.info("Default prompts set for pipeline")
    except Exception as e:
        logger.error(f"Error setting default prompts: {e}")
    
    # Only warm up pipeline if explicitly requested
    if app.get("warm_pipeline", False):
        try:
            logger.info("Warming up video pipeline on startup...")
            await app["pipeline"].warm_video()
            logger.info("Video pipeline warmed up successfully on startup")
        except Exception as e:
            logger.error(f"Error warming up pipeline on startup: {e}")
            # Don't raise the exception to allow the application to start
            # The pipeline will be warmed when needed
    else:
        logger.info("Pipeline warming skipped (use --warm-pipeline to enable)")
    
    app["pcs"] = set()
    app["video_tracks"] = {}


async def on_shutdown(app: web.Application):
    pcs = app["pcs"]
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    
    # Clean up WHIP resources
    if 'whip_handler' in app:
        await app['whip_handler'].cleanup_all_resources()
        
    # Clean up WHEP resources
    if 'whep_handler' in app:
        await app['whep_handler'].cleanup_all_resources()


def parse_media_ports(media_ports_str: Optional[str]) -> Optional[list]:
    """Parse media ports string into a list of port numbers.
    
    Supports:
    - Single port: "5678"
    - Comma-separated: "5678,5679,5680"
    - Port range: "5678-5900"
    - Mixed: "5678-5680,5900-5902"
    
    Args:
        media_ports_str: String containing port specification
        
    Returns:
        List of port numbers or None if no ports specified
    """
    if not media_ports_str:
        return None
    
    ports = []
    parts = media_ports_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Port range: "5678-5900"
            try:
                start, end = part.split('-')
                start_port = int(start.strip())
                end_port = int(end.strip())
                if start_port > end_port:
                    start_port, end_port = end_port, start_port
                ports.extend(range(start_port, end_port + 1))
            except ValueError:
                logger.warning(f"Invalid port range format: {part}")
                continue
        else:
            # Single port
            try:
                port = int(part)
                ports.append(port)
            except ValueError:
                logger.warning(f"Invalid port number: {part}")
                continue
    
    # Remove duplicates and sort
    ports = sorted(list(set(ports)))
    
    if not ports:
        logger.warning("No valid ports found in media_ports specification")
        return None
    
    logger.info(f"Parsed {len(ports)} media ports: {ports[:5]}{'...' if len(ports) > 5 else ''}")
    return ports


def start_server(
    port: int = 8889,
    host: str = "127.0.0.1",
    workspace: Optional[str] = None,
    media_ports: Optional[str] = None,
    log_level: str = "INFO",
    monitor: bool = False,
    stream_id_label: bool = False,
    comfyui_log_level: Optional[str] = None,
    comfyui_inference_log_level: Optional[str] = None,
    warm_pipeline: bool = False,
    max_bitrate: int = 8000000,
    min_bitrate: int = 4000000,
    video_width: int = 512,
    video_height: int = 512,
):
    """Start the ComfyStream server with the specified configuration.
    
    Args:
        port: The port to bind the server to (default: 8889)
        host: The host to bind the server to (default: "127.0.0.1")
        workspace: The ComfyUI workspace directory (required)
        media_ports: Media ports specification (default: "5678-5900")
        log_level: The logging level (default: "INFO")
        monitor: Whether to enable Prometheus metrics (default: False)
        stream_id_label: Whether to include stream ID in metrics (default: False)
        comfyui_log_level: The global logging level for ComfyUI
        comfyui_inference_log_level: The logging level for ComfyUI inference
        warm_pipeline: Whether to warm up the pipeline on startup (default: False)
        max_bitrate: Maximum video bitrate in bits per second (default: 8000000)
        min_bitrate: Minimum video bitrate in bits per second (default: 4000000)
        video_width: Default video width in pixels (default: 512)
        video_height: Default video height in pixels (default: 512)
        jpeg_quality: JPEG compression quality 0-100 (default: 90)
    """
    if workspace is None:
        raise ValueError("workspace parameter is required")
    
    # Set default media ports if not specified
    if media_ports is None:
        media_ports = "5678-5900"
    
    # Update global bitrate constants
    global MAX_BITRATE, MIN_BITRATE
    MAX_BITRATE = max_bitrate
    MIN_BITRATE = min_bitrate
    
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    app = web.Application()
    app["media_ports"] = parse_media_ports(media_ports)
    app["workspace"] = workspace
    app["warm_pipeline"] = warm_pipeline
    app["video_width"] = video_width
    app["video_height"] = video_height
    
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
    offer_handler = create_offer_handler(app)
    app.router.add_post("/offer", offer_handler)
    app.router.add_post("/prompt", set_prompt)
    
    # Setup HTTP streaming routes
    setup_routes(app, cors)
    
    # Setup WHIP routes
    setup_whip_routes(app, cors, get_ice_servers, VideoStreamTrack, AudioStreamTrack)
    
    # Setup WHEP routes
    setup_whep_routes(app, cors, get_ice_servers)
    
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
    
    # Add processing readiness status endpoint for WHEP clients
    async def processing_status_handler(request):
        """Endpoint for WHEP clients to check if processed streams are ready."""
        try:
            status = {
                "processing_ready": False,
                "whip_sessions": 0,
                "whep_sessions": 0,
                "active_pipelines": 0,
                "frames_available": False,
                "message": "No active processing"
            }
            
            # Check WHIP sessions (incoming streams)
            whip_sessions = {}
            if 'whip_handler' in app:
                whip_sessions = app['whip_handler'].get_active_sessions()
                status["whip_sessions"] = len(whip_sessions)
            
            # Check WHEP sessions (outgoing streams)
            whep_sessions = {}
            if 'whep_handler' in app:
                whep_sessions = app['whep_handler'].get_active_sessions()
                status["whep_sessions"] = len(whep_sessions)
            
            # Check if there are active processing pipelines
            active_pipelines = 0
            frames_available = False
            
            for session_id, session in whip_sessions.items():
                if session.get('connection_state') == 'connected' and session.get('has_video'):
                    active_pipelines += 1
                    
            # Check if frame buffer has frames available
            try:
                from frame_buffer import FrameBuffer
                frame_buffer = FrameBuffer.get_instance()
                # Check if current frame is not None instead of calling non-existent has_frames method
                if frame_buffer.get_current_frame() is not None:
                    frames_available = True
            except:
                # Frame buffer not available or no frames
                pass
            
            # Check WHEP stream manager for available frames
            if 'whep_handler' in app and app['whep_handler'].stream_manager:
                if (app['whep_handler'].stream_manager.latest_video_frame is not None):
                    frames_available = True
            
            status["active_pipelines"] = active_pipelines
            status["frames_available"] = frames_available
            
            # Determine overall readiness
            if active_pipelines > 0 and frames_available:
                status["processing_ready"] = True
                status["message"] = f"Processing ready - {active_pipelines} active pipeline(s) with frames available"
            elif active_pipelines > 0:
                status["processing_ready"] = False
                status["message"] = f"Processing warming up - {active_pipelines} pipeline(s) starting"
            elif status["whip_sessions"] > 0:
                status["processing_ready"] = False
                status["message"] = "WHIP sessions active but no connected pipelines yet"
            else:
                status["processing_ready"] = False
                status["message"] = "No active WHIP sessions - start publishing first"
            
            # Add detailed session info
            status["details"] = {
                "whip_sessions": whip_sessions,
                "whep_sessions": whep_sessions
            }
            
            return web.json_response(status)
            
        except Exception as e:
            logger.error(f"Error in processing status handler: {e}")
            return web.json_response({
                "processing_ready": False,
                "error": str(e),
                "message": "Error checking processing status"
            }, status=500)
    
    cors.add(app.router.add_get("/processing/status", processing_status_handler))

    # Add detailed session info endpoint
    async def session_info_handler(request):
        """Endpoint for getting detailed session information."""
        try:
            info = {
                "whip_sessions": {},
                "whep_sessions": {},
                "video_tracks": {},
                "pipeline_status": {
                    "initialized": False,
                    "warmed": False
                }
            }
            
            # Get WHIP sessions
            if 'whip_handler' in app:
                info["whip_sessions"] = app['whip_handler'].get_active_sessions()
            
            # Get WHEP sessions
            if 'whep_handler' in app:
                info["whep_sessions"] = app['whep_handler'].get_active_sessions()
            
            # Get video tracks
            if 'video_tracks' in app:
                for track_id, track in app['video_tracks'].items():
                    info["video_tracks"][track_id] = {
                        "running": getattr(track, 'running', False),
                        "collect_task_done": track.collect_task.done() if hasattr(track, 'collect_task') else True
                    }
            
            # Get pipeline status
            if 'pipeline' in app:
                pipeline = app['pipeline']
                info["pipeline_status"] = {
                    "initialized": True,
                    "width": getattr(pipeline, 'width', None),
                    "height": getattr(pipeline, 'height', None),
                    "warmed": getattr(pipeline, '_warmed', False)
                }
            
            return web.json_response(info)
            
        except Exception as e:
            logger.error(f"Error in session info handler: {e}")
            return web.json_response({
                "error": str(e),
                "message": "Error getting session info"
            }, status=500)
    
    cors.add(app.router.add_get("/sessions/info", session_info_handler))

    # Add Prometheus metrics endpoint.
    app["metrics_manager"] = MetricsManager(include_stream_id=stream_id_label)
    if monitor:
        app["metrics_manager"].enable()
        logger.info(
            f"Monitoring enabled - Prometheus metrics available at: "
            f"http://{host}:{port}/metrics"
        )
        app.router.add_get("/metrics", app["metrics_manager"].metrics_handler)

    # Add hosted platform route prefix.
    # NOTE: This ensures that the local and hosted experiences have consistent routes.
    add_prefix_to_app_routes(app, "/live")

    def force_print(*args, **kwargs):
        print(*args, **kwargs, flush=True)
        sys.stdout.flush()

    # Allow overriding of ComyfUI log levels.
    if comfyui_log_level:
        log_level_val = logging._nameToLevel.get(comfyui_log_level.upper())
        if log_level_val is not None:
            logging.getLogger("comfy").setLevel(log_level_val)
    if comfyui_inference_log_level:
        app["comfui_inference_log_level"] = comfyui_inference_log_level

    web.run_app(app, host=host, port=int(port), print=force_print)


def main():
    """Console script entry point for comfystream-server."""
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
        "--warm-pipeline",
        default=False,
        action="store_true",
        help="Warm up the pipeline on startup (similar to main.py)",
    )
    parser.add_argument(
        "--max-bitrate",
        default=8000000,
        type=int,
        help="Maximum video bitrate in bits per second (default: 8000000)",
    )
    parser.add_argument(
        "--min-bitrate",
        default=4000000,
        type=int,
        help="Minimum video bitrate in bits per second (default: 4000000)",
    )
    parser.add_argument(
        "--video-width",
        default=1280,
        type=int,
        help="Default video width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--video-height",
        default=720,
        type=int,
        help="Default video height in pixels (default: 720)",
    )

    args = parser.parse_args()

    start_server(
        port=args.port,
        host=args.host,
        workspace=args.workspace,
        media_ports=args.media_ports,
        log_level=args.log_level,
        monitor=args.monitor,
        stream_id_label=args.stream_id_label,
        comfyui_log_level=args.comfyui_log_level,
        comfyui_inference_log_level=args.comfyui_inference_log_level,
        warm_pipeline=args.warm_pipeline,
        max_bitrate=args.max_bitrate,
        min_bitrate=args.min_bitrate,
        video_width=args.video_width,
        video_height=args.video_height,
    )


if __name__ == "__main__":
    main()
