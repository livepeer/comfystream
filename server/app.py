import argparse
import asyncio
import json
import logging
import os
import sys
import torch
import av
# Initialize CUDA before any other imports to prevent core dump.
if torch.cuda.is_available():
    torch.cuda.init()

from typing import Dict, Union, Any, Optional
from pydantic import BaseModel, Field, field_validator

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
from comfystream.server.workflows import get_default_workflow, load_workflow
from trickle_api import setup_trickle_routes, cleanup_trickle_streams
from frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)
logging.getLogger("aiortc.rtcrtpsender").setLevel(logging.WARNING)
logging.getLogger("aiortc.rtcrtpreceiver").setLevel(logging.WARNING)

MAX_BITRATE = 2000000
MIN_BITRATE = 2000000


class HealthStateManager:
    """Manages the health state of the ComfyStream server."""
    
    def __init__(self):
        self.state = "LOADING"  # Initial state during startup
        self.error_message = None
        self.is_pipeline_warming = False
        self.pipeline_ready = False
        self.active_webrtc_streams = 0
        self.active_trickle_streams = 0
        self.startup_complete = False
        
    def set_loading(self, reason: Optional[str] = None):
        """Set state to LOADING (pipeline warming up)."""
        self.state = "LOADING"
        self.error_message = None
        logger.info(f"Health state: LOADING{f' - {reason}' if reason else ''}")
        
    def set_idle(self):
        """Set state to IDLE (no active streams)."""
        if self.state == "ERROR":
            return  # Don't change from ERROR state unless explicitly cleared
        self.state = "IDLE"
        self.error_message = None
        logger.debug("Health state: IDLE")
        
    def set_ok(self):
        """Set state to OK (streams active)."""
        if self.state == "ERROR":
            return  # Don't change from ERROR state unless explicitly cleared
        self.state = "OK"
        self.error_message = None
        logger.debug("Health state: OK")
        
    def set_error(self, message: str):
        """Set state to ERROR with error message."""
        self.state = "ERROR"
        self.error_message = message
        logger.error(f"Health state: ERROR - {message}")
        
    def clear_error(self):
        """Clear error state and recalculate appropriate state."""
        self.error_message = None
        # Reset state from ERROR before recalculating - this allows _update_state to work properly
        if self.state == "ERROR":
            self.state = "LOADING"  # Temporary state before recalculation
        self._update_state()
        
    def set_pipeline_warming(self, warming: bool):
        """Set pipeline warming state."""
        self.is_pipeline_warming = warming
        if warming:
            self.set_loading("Pipeline warming")
        else:
            self._update_state()
    
    def set_pipeline_ready(self, ready: bool):
        """Set pipeline ready state."""
        self.pipeline_ready = ready
        if ready and not self.is_pipeline_warming:
            self._update_state()
            
    def set_startup_complete(self):
        """Mark startup as complete."""
        self.startup_complete = True
        self._update_state()
        
    def update_webrtc_streams(self, count: int):
        """Update count of active WebRTC streams."""
        self.active_webrtc_streams = count
        self._update_state()
        
    def update_trickle_streams(self, count: int):
        """Update count of active trickle streams."""
        self.active_trickle_streams = count
        self._update_state()
        
    def _update_state(self):
        """Internal method to update state based on current conditions."""
        if self.state == "ERROR":
            return  # Don't change from ERROR state
            
        # Check if we should be in LOADING state
        if not self.startup_complete or self.is_pipeline_warming:
            self.set_loading("Startup in progress" if not self.startup_complete else "Pipeline warming")
            return
            
        # Check if we have active streams
        total_streams = self.active_webrtc_streams + self.active_trickle_streams
        if total_streams > 0:
            self.set_ok()
        else:
            self.set_idle()
            
    def get_status(self) -> dict:
        """Get current health status."""
        return {
            "status": self.state,
            "error_message": self.error_message,
            "pipeline_ready": self.pipeline_ready,
            "active_webrtc_streams": self.active_webrtc_streams,
            "active_trickle_streams": self.active_trickle_streams,
            "startup_complete": self.startup_complete
        }


# Simplified models - use centralized validation functions instead of custom Pydantic validators

class OfferRequest(BaseModel):
    """Pydantic model for WebRTC offer requests."""
    
    offer: Dict[str, Any] = Field(..., description="WebRTC offer parameters")
    prompts: Any = Field(..., description="Prompt data - can be a JSON string, single prompt dict, or list of prompt dicts")


class ControlMessage(BaseModel):
    """Pydantic model for WebRTC control channel messages."""
    
    type: str = Field(..., description="Message type")
    prompts: Any = Field(None, description="Prompt data - can be a JSON string, single prompt dict, or list of prompt dicts")
    width: Union[int, str, None] = Field(None, description="Video width")
    height: Union[int, str, None] = Field(None, description="Video height")
    
    @field_validator('width', 'height', mode='before')
    @classmethod
    def parse_dimensions(cls, v):
        """Parse width/height from string to int if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                raise ValueError(f"Invalid dimension value: {v}")
        return v





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
                    # Check if frame is a VideoFrame before passing to pipeline
                    if isinstance(frame, av.VideoFrame):
                        await self.pipeline.put_video_frame(frame)
                    else:
                        logger.error(f"Received non-video frame as VideoStreamTrack: {type(frame)}")
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
            frame_buffer = FrameBuffer.get_instance()
            frame_buffer.update_frame(processed_frame)
        except Exception as e:
            # Don't let frame buffer errors affect the main pipeline
            logger.error(f"Error updating frame buffer: {e}")

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
                    # Check if frame is an AudioFrame before passing to pipeline
                    if isinstance(frame, av.AudioFrame):
                        await self.pipeline.put_audio_frame(frame)
                    else:
                        logger.error(f"Received non-audio frame as AudioStreamTrack: {type(frame)}")
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

    # Add default STUN servers
    default_stun_servers = [
        "stun:stun.l.google.com:19302",
        "stun:stun.cloudflare.com:3478", 
        "stun:stun1.l.google.com:19302",
        "stun:stun2.l.google.com:19302",
        "stun:stun3.l.google.com:19302"
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


async def offer(request):
    pipeline = request.app["pipeline"]
    pcs = request.app["pcs"]

    params = await request.json()

    # Parse and validate request using Pydantic model
    try:
        offer_request = OfferRequest(**params)
        # Pipeline now handles prompt parsing internally
        await pipeline.set_prompts(offer_request.prompts)
    except ValueError as e:
        logger.error(f"[Offer] Invalid prompt format: {e}")
        return web.Response(status=400, text="Invalid prompt format.")
    except Exception as e:
        logger.error(f"[Offer] Error setting prompts: {e}")
        return web.Response(status=500, text="An internal server error occurred.")

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

    # Handle data channels from client
    @pc.on("datachannel")
    def on_datachannel(channel):
        if channel.label == "control":

            @channel.on("message")
            async def on_message(message):
                try:
                    params = json.loads(message)
                    
                    # Parse and validate message using Pydantic model
                    try:
                        control_msg = ControlMessage(**params)
                    except ValueError as e:
                        logger.error(f"[Control] Invalid message format: {e}")
                        return

                    if control_msg.type == "get_nodes":
                        nodes_info = await pipeline.get_nodes_info()
                        response = {"type": "nodes_info", "nodes": nodes_info}
                        channel.send(json.dumps(response))
                    elif control_msg.type == "update_prompts":
                        if control_msg.prompts is None:
                            logger.warning(
                                "[Control] Missing prompts in update_prompts message"
                            )
                            return
                        try:
                            # Pipeline now handles prompt parsing internally
                            await pipeline.update_prompts(control_msg.prompts)
                        except ValueError as e:
                            logger.error(f"[Control] Invalid prompt format: {e}")
                        except Exception as e:
                            logger.error(f"Error updating prompts: {str(e)}")
                            health_manager = request.app["health_manager"]
                            health_manager.set_error("Error updating prompts")
                        response = {"type": "prompts_updated", "success": True}
                        channel.send(json.dumps(response))
                    elif control_msg.type == "update_resolution":
                        if control_msg.width is None or control_msg.height is None:
                            logger.warning("[Control] Missing width or height in update_resolution message")
                            return
                        try:
                            # Width and height are already validated and converted by Pydantic
                            width = control_msg.width
                            height = control_msg.height
                            
                            # Update pipeline resolution for future frames
                            pipeline.width = width
                            pipeline.height = height
                            logger.info(f"[Control] Updated resolution to {width}x{height}")
                            
                            # Mark that we've received resolution
                            resolution_received["value"] = True
                            
                            # Warm the video pipeline with the new resolution if pipeline is not already warmed
                            if "m=video" in pc.remoteDescription.sdp: 
                                if not request.app.get("pipeline_warmed", {}).get("video", False):
                                    health_manager = request.app["health_manager"]
                                    health_manager.set_pipeline_warming(True)
                                    await pipeline.warm_video()
                                    request.app["pipeline_warmed"]["video"] = True
                                    health_manager.set_pipeline_warming(False)
                                    logger.info(f"[Control] Pipeline warmed with new resolution {width}x{height}")
                                else:
                                    logger.info(f"[Control] Video pipeline already warmed on startup")
                                
                            response = {
                                "type": "resolution_updated",
                                "success": True
                            }
                            channel.send(json.dumps(response))
                        except Exception as e:
                            logger.error(f"[Control] Error updating resolution: {e}")
                            health_manager = request.app["health_manager"]
                            health_manager.set_error("Error updating resolution")
                            response = {
                                "type": "resolution_updated",
                                "success": False,
                                "error": "Error updating resolution."
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
        
        elif channel.label == "text":
            # Text data channel for streaming text output
            logger.info("Text data channel established")
            
            # Create background task to stream text data
            async def stream_text_data():
                try:
                    while pc.connectionState in ["connecting", "connected"]:
                        try:
                            # Get text output from pipeline with timeout
                            text_output = await asyncio.wait_for(
                                pipeline.get_text_output(), 
                                timeout=0.1
                            )
                            
                            # Send text data through channel if still open
                            if channel.readyState == "open":
                                text_message = {
                                    "type": "text_output",
                                    "data": text_output,
                                    "timestamp": asyncio.get_event_loop().time()
                                }
                                channel.send(json.dumps(text_message))
                                logger.debug(f"Sent text output: {text_output[:100]}...")
                            else:
                                break
                                
                        except asyncio.TimeoutError:
                            # No text output available, continue
                            await asyncio.sleep(0.01)
                            continue
                        except Exception as e:
                            logger.error(f"Error streaming text data: {e}")
                            await asyncio.sleep(0.1)
                            continue
                            
                except asyncio.CancelledError:
                    logger.info("Text streaming task cancelled")
                except Exception as e:
                    logger.error(f"Text streaming task error: {e}")
                finally:
                    logger.info("Text streaming task ended")
            
            # Start background task for text streaming
            text_task = asyncio.create_task(stream_text_data())
            
            # Clean up task when channel closes
            @channel.on("close")
            def on_text_channel_close():
                logger.info("Text data channel closed")
                if not text_task.done():
                    text_task.cancel()

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track received: {track.kind}")
        health_manager = request.app["health_manager"]
        
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

        # Update stream count for health tracking
        health_manager.update_webrtc_streams(len(request.app["video_tracks"]))

        @track.on("ended")
        async def on_ended():
            logger.info(f"{track.kind} track ended")
            request.app["video_tracks"].pop(track.id, None)
            # Update stream count after track ends
            health_manager.update_webrtc_streams(len(request.app["video_tracks"]))

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

    # Audio pipeline warming is not supported
    if "m=audio" in pc.remoteDescription.sdp:
        logger.info("[WebRTC] Audio pipeline warming is not supported")
    
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
    health_manager = request.app["health_manager"]

    prompt_data = await request.json()
    
    # Pipeline now handles prompt parsing internally
    try:
        await pipeline.set_prompts(prompt_data)
    except ValueError as e:
        logger.error(f"[SetPrompt] Invalid prompt format: {e}")
        return web.Response(status=400, text="Invalid prompt format.")
    except Exception as e:
        logger.error(f"[SetPrompt] Error setting prompts: {e}")
        health_manager.set_error("Error setting prompts")
        return web.Response(status=500, text="An internal server error occurred.")

    return web.Response(content_type="application/json", text="OK")
    

async def health(request):
    health_manager = request.app["health_manager"]
    status = health_manager.get_status()
    return web.json_response({"status": status["status"]})


async def on_startup(app: web.Application) -> None:
    # Initialize health state manager
    health_manager = HealthStateManager()
    app["health_manager"] = health_manager
    
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
    
    # Set prompts for the pipeline (either warmup workflow or default)
    try:
        warmup_workflow = app.get("warmup_workflow")
        if warmup_workflow:
            logger.info(f"Using warmup workflow: {warmup_workflow}")
            warmup_prompt = load_workflow(warmup_workflow)
        else:
            logger.info("Using default workflow for warmup")
            warmup_prompt = get_default_workflow()
        
        await app["pipeline"].set_prompts([warmup_prompt])
        logger.info("Warmup prompts set for pipeline")
    except Exception as e:
        logger.error(f"Error setting prompts, warmup failed on startup: {e}")
        health_manager.set_error("Error setting prompts on startup")
        
    
    # Track warming status to avoid redundant warming
    app["pipeline_warmed"] = {"video": False}
    
    # Warm up pipeline by default unless explicitly skipped
    if app.get("warm_pipeline", True):
        try:
            logger.info("Warming up video pipeline on startup...")
            health_manager.set_pipeline_warming(True)
            await app["pipeline"].warm_video()
            app["pipeline_warmed"]["video"] = True
            health_manager.set_pipeline_warming(False)
            health_manager.set_pipeline_ready(True)
            logger.info("Video pipeline warmed up successfully on startup")
        except Exception as e:
            logger.error(f"Error warming up pipeline on startup: {e}")
            health_manager.set_error("Error warming up pipeline on startup")
            # Don't raise the exception to allow the application to start
            # The pipeline will be warmed when needed
    else:
        logger.info("Pipeline warming skipped on startup")
        health_manager.set_pipeline_ready(True)
    
    app["pcs"] = set()
    app["video_tracks"] = {}
    
    # Setup trickle routes now that health manager is initialized
    setup_trickle_routes(app, app["cors"])
    logger.info("Trickle API routes enabled")
    
    # Mark startup as complete
    health_manager.set_startup_complete()


async def on_shutdown(app: web.Application) -> None:
    pcs = app["pcs"]
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    
    # Cleanup trickle streams
    await cleanup_trickle_streams()
    logger.info("Trickle streams cleaned up")


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
        "--skip-warmup",
        default=False,
        action="store_true",
        help="Skip warming the pipeline on startup (reduces startup time but increases latency for first user)",
    )
    parser.add_argument(
        "--warmup-workflow",
        default=None,
        help="Specify a workflow file name to use for pipeline warmup (e.g., 'sd15-tensorrt-api.json'). If not specified, uses default workflow.",
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
    app["warm_pipeline"] = not args.skip_warmup
    app["warmup_workflow"] = args.warmup_workflow
    
    # Setup CORS
    cors = setup_cors(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["GET", "POST", "OPTIONS"]
        )
    })
    
    # Store cors in app for use in startup
    app["cors"] = cors

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/", health)
    app.router.add_get("/health", health)

    # WebRTC signalling and control routes.
    app.router.add_post("/offer", offer)
    app.router.add_post("/prompt", set_prompt)
    
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

    web.run_app(app, host=args.host, port=int(args.port), print=force_print)
