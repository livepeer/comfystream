import argparse
import asyncio
import json
import logging
import os
import sys
import torch
import uuid
import base64
from datetime import datetime

# Initialize CUDA before any other imports to prevent core dump.
if torch.cuda.is_available():
    torch.cuda.init()

from aiohttp import web
from aiohttp_cors import setup as setup_cors, ResourceOptions
from webrtc_utils import offer

# Import HTTP streaming modules
from http_streaming.routes import setup_routes
from comfystream.pipeline import Pipeline
from comfystream.server.utils import patch_loop_datagram, add_prefix_to_app_routes
from comfystream.server.metrics import MetricsManager, StreamStatsManager
from trickle_utils import _process_video_capability, _process_image_capability, _process_trickle_stream_direct
from trickle_pipeline import StreamManifest
# Import trickle streaming components
from comfystream.server.trickle import (
    high_throughput_segment_publisher,
    TrickleSegmentEncoder
)

logger = logging.getLogger(__name__)

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
        if stream_manifest.frame_queue is not None:
            stream_manifest.publisher_task = asyncio.create_task(
                high_throughput_segment_publisher(
                    stream_manifest.output_stream_url, 
                    stream_manifest.frame_queue, 
                    max_fps=24.0,  # Match segmenter FPS
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
