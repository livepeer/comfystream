import argparse
import asyncio
import json
import logging
import os
import sys
import time
import secrets
import torch
from typing import Optional, Dict, Any, List

# Initialize CUDA before any other imports to prevent core dump.
if torch.cuda.is_available():
    torch.cuda.init()

from aiohttp import web
from aiohttp_cors import setup as setup_cors, ResourceOptions
from pytrickle import StreamServer, RegisterCapability
from comfystream.server.utils import add_prefix_to_app_routes
from comfystream.server.metrics import MetricsManager
from frame_processor import ComfyStreamFrameProcessor

logger = logging.getLogger(__name__)


async def get_pipeline_info(request):
    """HTTP endpoint to get pipeline information."""
    frame_processor = request.app.get("frame_processor")
    if not frame_processor or not frame_processor.pipeline:
        return web.json_response(
            {"error": "Pipeline not available"}, 
            status=500
        )
    
    try:
        pipeline = frame_processor.pipeline
        info = {
            "initialized": frame_processor._initialized,
            "modalities": list(pipeline.get_workflow_modalities()),
            "capabilities": {
                "accepts_video_input": pipeline.accepts_video_input(),
                "accepts_audio_input": pipeline.accepts_audio_input(),
                "produces_video_output": pipeline.produces_video_output(),
                "produces_audio_output": pipeline.produces_audio_output(),
                "produces_text_output": pipeline.produces_text_output(),
            },
            "resolution": {
                "width": pipeline.width,
                "height": pipeline.height
            }
        }
        return web.json_response(info)
        
    except Exception as e:
        logger.error(f"[BYOC] Error getting pipeline info: {e}")
        return web.json_response(
            {"error": f"Failed to get pipeline info: {str(e)}"}, 
            status=500
        )


def health(_):
    """Health check endpoint."""
    return web.Response(content_type="application/json", text="OK")


async def main(args):
    """Start StreamServer with ComfyStream frame processor and custom routes."""
    logger.info("[BYOC] Starting BYOC StreamServer")

    # Initialize frame processor and pipeline
    frame_processor = ComfyStreamFrameProcessor(
        workspace=args.workspace,
        comfyui_inference_log_level=None,
    )
    await frame_processor.load_model()

    # Configure CORS similar to aiohttp-cors setup
    cors_config = {
        "*": {
            "allow_credentials": True,
            "expose_headers": "*",
            "allow_headers": "*",
            "allow_methods": ["GET", "POST", "OPTIONS"],
        }
    }

    # Create StreamServer with custom options
    server = StreamServer(
        frame_processor=frame_processor,
        host=args.host,
        port=int(args.port),
        route_prefix="/api",
        enable_default_routes=True,
        cors_config=cors_config,
    )

    # Expose frame_processor for custom handlers
    server.set_context("frame_processor", frame_processor)

    # Optional metrics
    metrics_manager = MetricsManager(include_stream_id=False)
    if args.monitor:
        metrics_manager.enable()
    server.set_context("metrics_manager", metrics_manager)

    # Register custom routes on the same app
    server.add_route("GET", "/", health)
    server.add_route("GET", "/health", health)
    server.add_route("GET", "/info", get_pipeline_info)
    if args.monitor:
        server.add_route("GET", "/metrics", metrics_manager.metrics_handler)

    # Add hosted platform route prefix (e.g., /comfystream)
    add_prefix_to_app_routes(server.get_app(), "/comfystream")

    # Start server
    await server.start_server()
    logger.info(f"[BYOC] StreamServer started at http://{args.host}:{args.port}")

    # Attempt orchestrator capability registration (non-blocking, quiet on failure)
    async def _try_register_capability():
        try:
            quiet_logger = logging.getLogger("comfystream.byoc.register")
            quiet_logger.setLevel(logging.CRITICAL)  # suppress warnings/errors inside registrar
            quiet_logger.propagate = False
            if not quiet_logger.handlers:
                quiet_logger.addHandler(logging.NullHandler())

            registered = await RegisterCapability.register(
                logger=quiet_logger,
                max_retries=1,
                delay=0.5,
                timeout=1.0,
            )
            if registered:
                logger.info(f"[BYOC] Capability registered with orchestrator: {registered}")
            else:
                logger.debug("[BYOC] Capability registration skipped or failed; continuing without orchestrator")
        except Exception:
            # Swallow any registration exceptions to avoid noisy logs
            logger.debug("[BYOC] Capability registration encountered an exception; continuing normally")

    asyncio.create_task(_try_register_capability())

    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ComfyStream BYOC server")
    parser.add_argument("--port", default=8890, help="Set the HTTP server port")
    parser.add_argument("--host", default="127.0.0.1", help="Set the host")
    parser.add_argument(
        "--workspace", default=None, required=True, help="Set ComfyUI workspace"
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

    # Allow overriding of ComfyUI log levels
    if args.comfyui_log_level:
        log_level = logging._nameToLevel.get(args.comfyui_log_level.upper())
        logging.getLogger("comfy").setLevel(log_level)

    if args.comfyui_inference_log_level:
        os.environ["COMFYUI_INFERENCE_LOG_LEVEL"] = args.comfyui_inference_log_level

    # Run the async StreamServer with our processor and custom routes
    asyncio.run(main(args))
