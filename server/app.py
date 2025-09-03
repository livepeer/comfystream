import argparse
import asyncio
import json
import logging
import os
import sys
from urllib.parse import urlparse

import torch
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
from aiortc.codecs import h264
from aiortc.rtcrtpsender import RTCRtpSender
from twilio.rest import Client


from pytrickle.stream_processor import StreamProcessor
from pytrickle.frames import VideoFrame, AudioFrame
from pytrickle.utils.register import RegisterCapability
from pytrickle.frame_skipper import FrameSkipConfig
from comfystream.pipeline import Pipeline
from comfystream.utils import load_prompt_from_file, convert_prompt, ComfyStreamParamsUpdateRequest
from comfystream import tensor_cache
from comfystream.server.utils import FPSMeter, patch_loop_datagram, add_prefix_to_app_routes
from comfystream.server.metrics import MetricsManager, StreamStatsManager
from frame_processor import ComfyStreamFrameProcessor
from http_streaming.routes import setup_routes
from frame_buffer import FrameBuffer

logger = logging.getLogger(__name__)
logging.getLogger("aiortc.rtcrtpsender").setLevel(logging.WARNING)
logging.getLogger("aiortc.rtcrtpreceiver").setLevel(logging.WARNING)

MAX_BITRATE = 2000000
MIN_BITRATE = 2000000


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
        frame_buffer = FrameBuffer.get_instance()
        frame_buffer.update_frame(processed_frame)

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


async def offer(request):
    pipeline = request.app["pipeline"]
    pcs = request.app["pcs"]

    params = await request.json()

    # Validate and set prompts
    if "prompts" in params:
        validated_prompts = validate_prompts(params["prompts"])
        if validated_prompts:
            await pipeline.set_prompts([validated_prompts])
            logger.info("WebRTC prompts validated and set successfully")

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
            
            # Set up text monitoring callback for this channel
            async def text_callback(text_data: str) -> bool:
                try:
                    response = {"type": "text_output", "text": text_data}
                    channel.send(json.dumps(response))
                    return True
                except Exception as e:
                    logger.error(f"Failed to send text data via control channel: {e}")
                    return False
            
            # Set up text monitoring
            pipeline.set_text_callback(text_callback)
            pipeline.start_text_monitoring()

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
                        validated_prompts = validate_prompts(params["prompts"])
                        if validated_prompts:
                            await pipeline.update_prompts([validated_prompts])
                            logger.debug("Control channel prompts updated successfully")
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
                        
                        # Run warmup after resolution is set
                        try:
                            modalities = pipeline.get_prompt_modalities()
                            logger.info(f"Running warmup with modalities: {modalities}")
                            
                            if modalities.get("video", {}).get("input") or modalities.get("video", {}).get("output"):
                                logger.info("Running video warmup...")
                                await pipeline.warm_video()
                                logger.info("Video warmup completed")
                            
                            if modalities.get("audio", {}).get("input") or modalities.get("audio", {}).get("output"):
                                logger.info("Running audio warmup...")
                                await pipeline.warm_audio()
                                logger.info("Audio warmup completed")
                                
                        except Exception as e:
                            logger.error(f"Warmup failed: {e}")
                            
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
            
            logger.info(f"Added video track to WebRTC, sender: {sender}")
            logger.info(f"Video track state - readyState: {track.readyState}, kind: {track.kind}")

            # Store video track in app for stats.
            stream_id = track.id
            request.app["video_tracks"][stream_id] = videoTrack

            codec = "video/H264"
            force_codec(pc, sender, codec)
            logger.info(f"Set video codec to {codec}")
            
        elif track.kind == "audio":
            audioTrack = AudioStreamTrack(track, pipeline)
            tracks["audio"] = audioTrack
            sender = pc.addTrack(audioTrack)
            logger.info(f"Added audio track to WebRTC, sender: {sender}")

        @track.on("ended")
        async def on_ended():
            logger.info(f"{track.kind} track ended")
            request.app["video_tracks"].pop(track.id, None)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state is: {pc.connectionState}")
        if pc.connectionState == "failed":
            logger.error("WebRTC connection failed!")
            pipeline.stop_text_monitoring()
            await pc.close()
            pcs.discard(pc)
        elif pc.connectionState == "closed":
            logger.info("WebRTC connection closed")
            pipeline.stop_text_monitoring()
            await pc.close()
            pcs.discard(pc)
        elif pc.connectionState == "connected":
            logger.info("WebRTC connection fully established")

    await pc.setRemoteDescription(offer)
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
    
    validated_prompts = validate_prompts(prompt)
    if validated_prompts:
        await pipeline.set_prompts([validated_prompts])
        return web.Response(content_type="application/json", text="OK")
    else:
        return web.Response(content_type="application/json", text="Invalid prompts", status=400)

def health(_):
    return web.Response(content_type="application/json", text="OK")

def validate_prompts(prompts_data):
    """Validate and normalize prompts data."""
    try:
        validated_request = ComfyStreamParamsUpdateRequest.model_validate({"prompts": prompts_data})
        validated_params = validated_request.model_dump()
        return validated_params.get("prompts")
    except Exception as e:
        logger.error(f"Prompt validation failed: {e}")
        return None




async def on_startup(app: web.Application):
    if app["media_ports"]:
        patch_loop_datagram(app["media_ports"])
    app["pcs"] = set()
    app["video_tracks"] = {}

async def warmup_on_startup(app: web.Application):
    # In WebRTC mode, only set prompts during startup
    # Actual warmup happens when resolution is received via control message
    warmup_path = app.get("warmup_workflow")
    if warmup_path and "pipeline" in app:
        try:
            prompt = load_prompt_from_file(warmup_path)
            await app["pipeline"].set_prompts([prompt])
            logger.info("Warmup workflow loaded, warmup will run when resolution is received")
        except Exception as e:
            logger.error(f"Failed to load warmup workflow: {e}")


async def on_shutdown(app: web.Application):
    pcs = app["pcs"]
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comfystream server. Mode is automatically detected: "
                   "pytrickle mode if CAPABILITY_URL env var is set, WebRTC mode otherwise."
    )
    parser.add_argument("--port", default=8889, help="Set the signaling port")
    parser.add_argument(
        "--media-ports", default=None, help="Set the UDP ports for WebRTC media"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Set the host")
    parser.add_argument(
        "--workspace", default=None, required=True, help="Set Comfy workspace"
    )
    parser.add_argument(
        "--warmup-workflow",
        default=os.getenv("WARMUP_WORKFLOW"),
        help="Path to a JSON workflow file to warm up at startup (RTC only). Can also be set via WARMUP_WORKFLOW env var",
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
        "--orch-url",
        default=None,
        help="Orchestrator URL for capability registration",
    )
    parser.add_argument(
        "--orch-secret",
        default=None,
        help="Orchestrator secret for capability registration",
    )
    parser.add_argument(
        "--capability-name",
        default=None,
        help="Name for this capability (default: comfystream-processor)",
    )
    
    # Frame skipping arguments
    parser.add_argument(
        "--frame-skip-enabled",
        default=True,
        action="store_true",
        help="Enable adaptive frame skipping based on queue sizes",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Allow overriding of ComfyUI log levels.
    if args.comfyui_log_level:
        log_level = logging._nameToLevel.get(args.comfyui_log_level.upper())
        logging.getLogger("comfy").setLevel(log_level)

    def force_print(*args, **kwargs):
        print(*args, **kwargs, flush=True)
        sys.stdout.flush()

    async def register_orchestrator(app_instance=None):
        """Register capability with orchestrator if configured."""
        try:
            orch_url = args.orch_url or os.getenv("ORCH_URL")
            orch_secret = args.orch_secret or os.getenv("ORCH_SECRET")
            
            if orch_url and orch_secret:
                os.environ.update({
                    "CAPABILITY_NAME": args.capability_name or os.getenv("CAPABILITY_NAME") or "comfystream-processor",
                    "CAPABILITY_DESCRIPTION": "ComfyUI streaming processor",
                    "CAPABILITY_URL": f"http://{args.host}:{args.port}",
                    "CAPABILITY_CAPACITY": "1",
                    "ORCH_URL": orch_url,
                    "ORCH_SECRET": orch_secret
                })
                
                result = await RegisterCapability.register(logger=logger)
                if result:
                    logger.info(f"Registered capability: {result.geturl()}")
        except Exception as e:
            logger.error(f"Orchestrator registration failed: {e}")



    # Choose between pytrickle and WebRTC based on CAPABILITY_URL environment variable
    capability_url = os.getenv("CAPABILITY_URL")
    enable_trickle = capability_url is not None
    
    if enable_trickle:
        try:
            parsed_capability_url = urlparse(capability_url)
            
            if not parsed_capability_url.port:
                raise ValueError(f"Port not specified in URL: {capability_url}")
            
            if not parsed_capability_url.hostname:
                raise ValueError(f"Hostname not specified in URL: {capability_url}")
            
        except Exception as e:
            logger.error(f"Failed to parse or validate CAPABILITY_URL '{capability_url}': {e}")
            logger.info("Falling back to WebRTC mode due to invalid CAPABILITY_URL")
            enable_trickle = False
    
    if enable_trickle and StreamProcessor is not None and 'parsed_capability_url' in locals():
        logger.info(f"CAPABILITY_URL detected ({capability_url}), starting pytrickle StreamProcessor mode...")
        frame_processor = ComfyStreamFrameProcessor(
            width=512,
            height=512,
            workspace=args.workspace,
            disable_cuda_malloc=True,
            gpu_only=True,
            preview_method='none',
            comfyui_inference_log_level=args.comfyui_inference_log_level,
            warmup_workflow=args.warmup_workflow
        )
        
        # Create frame skip configuration only if enabled
        frame_skip_config = None
        if args.frame_skip_enabled:
            frame_skip_config = FrameSkipConfig(
                target_fps=args.frame_skip_target_fps,
                max_queue_size=args.frame_skip_max_queue_size,
                max_cleanup_frames=args.frame_skip_max_cleanup_frames,
                adaptation_cooldown=args.frame_skip_adaptation_cooldown
            )
            logger.info(f"Frame skipping enabled: target_fps={args.frame_skip_target_fps}, "
                       f"max_queue_size={args.frame_skip_max_queue_size}, "
                       f"max_cleanup_frames={args.frame_skip_max_cleanup_frames}, "
                       f"adaptation_cooldown={args.frame_skip_adaptation_cooldown}")
        else:
            logger.info("Frame skipping disabled")
        
        processor = StreamProcessor(
            video_processor=frame_processor.process_video_async,
            audio_processor=frame_processor.process_audio_async,
            model_loader=frame_processor.load_model,
            param_updater=frame_processor.update_params,
            on_stream_stop=frame_processor.on_stream_stop,
            name="comfystream-processor",
            port=int(args.port),
            host=args.host,
            frame_skip_config=frame_skip_config
        )

        frame_processor.set_stream_processor(processor)
        
        # Create async startup function to load model
        async def load_model_on_startup(app):
            await processor._frame_processor.load_model()
        
        # Add model loading and registration to startup hooks
        processor.server.app.on_startup.append(load_model_on_startup)
        processor.server.app.on_startup.append(register_orchestrator)
        processor.run()
        
    else:
        # Use WebRTC server (default mode)
        if enable_trickle and StreamProcessor is None:
            logger.warning("CAPABILITY_URL detected but pytrickle not available, falling back to WebRTC mode")
        elif enable_trickle:
            logger.info("CAPABILITY_URL detected but pytrickle not imported, using WebRTC mode")
        else:
            logger.info("No CAPABILITY_URL detected, starting WebRTC server mode...")
        app = web.Application()
        app["media_ports"] = args.media_ports.split(",") if args.media_ports else None
        app["workspace"] = args.workspace
        app["warmup_workflow"] = args.warmup_workflow
        
        cors = setup_cors(app, defaults={
            "*": ResourceOptions(allow_credentials=True, expose_headers="*", 
                               allow_headers="*", allow_methods=["GET", "POST", "OPTIONS"])
        })

        # Create pipeline for WebRTC mode
        app["pipeline"] = Pipeline(
            width=512, height=512, cwd=args.workspace,
            disable_cuda_malloc=True, gpu_only=True, preview_method='none',
            comfyui_inference_log_level=args.comfyui_inference_log_level
        )
        
        app.on_startup.extend([on_startup, warmup_on_startup])
        app.on_shutdown.append(on_shutdown)

        app.router.add_get("/", health)
        app.router.add_get("/health", health)
        app.router.add_post("/offer", offer)
        app.router.add_post("/prompt", set_prompt)
        
        setup_routes(app, cors)
        app.router.add_static("/", path=os.path.join(os.path.dirname(__file__), "public"), name="static")
        
        # Add metrics if enabled
        app["metrics_manager"] = MetricsManager(include_stream_id=args.stream_id_label)
        if args.monitor:
            app["metrics_manager"].enable()
            app.router.add_get("/metrics", app["metrics_manager"].metrics_handler)
            
        # Add stream stats
        stream_stats_manager = StreamStatsManager(app)
        app.router.add_get("/streams/stats", stream_stats_manager.collect_all_stream_metrics)
        app.router.add_get("/stream/{stream_id}/stats", stream_stats_manager.collect_stream_metrics_by_id)

        add_prefix_to_app_routes(app, "/live")
        web.run_app(app, host=args.host, port=int(args.port), print=force_print)
