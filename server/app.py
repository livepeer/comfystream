import argparse
import asyncio
import json
import logging
import os
import sys
import time
import secrets
import torch
import numpy as np

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
from comfystream.utils import load_prompt_from_file
from twilio.rest import Client
from comfystream.server.utils import FPSMeter, patch_loop_datagram, add_prefix_to_app_routes
from comfystream.server.metrics import MetricsManager, StreamStatsManager
import time

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
                    #     if "width" not in params or "height" not in params:
                    #         logger.warning("[Control] Missing width or height in update_resolution message")
                    #         return
                    #     # Update pipeline resolution for future frames
                    #     pipeline.width = params["width"]
                    #     pipeline.height = params["height"]
                    #     logger.info(f"[Control] Updated resolution to {params['width']}x{params['height']}")
                        
                    #     # Mark that we've received resolution
                        resolution_received["value"] = True
                        
                        # Warm the video pipeline with the new resolution if workflow has video
                        # modalities = pipeline.get_prompt_modalities()
                        # if "m=video" in pc.remoteDescription.sdp:
                        #     if (modalities.get("video", {}).get("input")):
                        #         await pipeline.warm_video()                            
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
            
            # # Check transceiver state
            # transceivers = pc.getTransceivers()
            # for i, t in enumerate(transceivers):
            #     if t.sender == sender:
            #         logger.info(f"Video transceiver {i}: direction={t.direction}, currentDirection={t.currentDirection}")
            #         break
            
            # # Test if recv works by calling it manually after a delay
            # async def test_recv():
            #     await asyncio.sleep(2)
            #     try:
            #         logger.info("Testing manual recv() call...")
            #         test_frame = await videoTrack.recv()
            #         logger.info(f"Manual recv() successful: {test_frame.width}x{test_frame.height}")
            #     except Exception as e:
            #         logger.error(f"Manual recv() failed: {e}")
            
            # asyncio.create_task(test_recv())
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
            await pc.close()
            pcs.discard(pc)
        elif pc.connectionState == "closed":
            logger.info("WebRTC connection closed")
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

    return web.Response(content_type="application/json", text="OK")

async def warm_pipeline(app: web.Application, prompt: dict):
    await app["pipeline"].set_prompts([prompt])
    modalities = app["pipeline"].get_prompt_modalities()
    logger.info(f"Startup warmup - detected modalities: {modalities}")
    if modalities.get("video", {}).get("input") or modalities.get("video", {}).get("output"):
        logger.info("Running startup video warmup")
        await app["pipeline"].warm_video()
    if modalities.get("audio", {}).get("input") or modalities.get("audio", {}).get("output"):
        logger.info("Running startup audio warmup")
        await app["pipeline"].warm_audio()

async def warmup_on_startup(app: web.Application):
    warmup_path = app.get("warmup_workflow")
    if not warmup_path:
        return
    try:
        raw_prompt = load_prompt_from_file(warmup_path)
        await warm_pipeline(app, raw_prompt)
    except Exception as e:
        # Log full traceback and brief diagnostics about the loaded prompt
        try:
            if 'raw_prompt' in locals() and isinstance(raw_prompt, dict):
                total_nodes = len(raw_prompt)
                sample_keys = list(raw_prompt.keys())[:5]
                missing = [
                    node_id for node_id, node in raw_prompt.items()
                    if not isinstance(node, dict) or 'class_type' not in node or 'inputs' not in node
                ][:5]
                logger.debug(
                    f"Warmup prompt diagnostics: total_nodes={total_nodes}, sample_keys={sample_keys}, missing_fields_nodes={missing}"
                )
        except Exception:
            pass
        logger.exception("Warmup workflow failed")


async def on_shutdown(app: web.Application):
    pcs = app["pcs"]
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


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
        "--warmup-workflow",
        default=None,
        help="Path to a JSON workflow file to warm up at startup (RTC only)",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    app = web.Application()
    app["media_ports"] = args.media_ports.split(",") if args.media_ports else None
    app["workspace"] = args.workspace
    
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
    app.on_startup.append(warmup_on_startup)
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
        logging.getLogger("comfy").setLevel(log_level)
    if args.comfyui_inference_log_level:
        app["comfui_inference_log_level"] = args.comfyui_inference_log_level

    # Store warmup workflow path for startup hook
    app["warmup_workflow"] = args.warmup_workflow

    web.run_app(app, host=args.host, port=int(args.port), print=force_print)
