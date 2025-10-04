import argparse
import asyncio
import logging
import os
import sys

import torch
# Initialize CUDA before any other imports to prevent core dump.
if torch.cuda.is_available():
    torch.cuda.init()

from aiohttp import web
from pytrickle.stream_processor import StreamProcessor
from pytrickle.utils.register import RegisterCapability
from pytrickle.frame_skipper import FrameSkipConfig
from frame_processor import ComfyStreamFrameProcessor
from comfystream.exceptions import ComfyStreamTimeoutFilter

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run comfystream server in BYOC (Bring Your Own Compute) mode using pytrickle."
    )
    parser.add_argument("--port", default=8000, help="Set the server port")
    parser.add_argument("--host", default="0.0.0.0", help="Set the host")
    parser.add_argument(
        "--workspace", default="/workspace/ComfyUI", required=False, help="Set Comfy workspace"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
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
        default=os.getenv("ORCH_URL"),
        help="Orchestrator URL for capability registration",
    )
    # Orchestrator secret is always read from environment variable for security
    # No command line argument to avoid exposing secrets in process lists
    parser.add_argument(
        "--capability-name",
        default=os.getenv("CAPABILITY_NAME", "comfystream"),
        help="Name for this capability (default: comfystream)",
    )
    parser.add_argument(
        "--capability-url",
        default=os.getenv("CAPABILITY_URL", "http://172.17.0.1:8000"),
        help="URL for this capability (default: http://172.17.0.1:8000)",
    )
    parser.add_argument(
        "--capability-description",
        default=os.getenv("CAPABILITY_DESCRIPTION", "ComfyUI streaming processor for BYOC mode"),
        help="Description for this capability",
    )
    parser.add_argument(
        "--capability-price-per-unit",
        default=os.getenv("CAPABILITY_PRICE_PER_UNIT", "0"),
        help="Price per unit for this capability (default: 0)",
    )
    parser.add_argument(
        "--capability-price-scaling",
        default=os.getenv("CAPABILITY_PRICE_SCALING", "1"),
        help="Price scaling factor for this capability (default: 1)",
    )
    parser.add_argument(
        "--capability-capacity",
        default=os.getenv("CAPABILITY_CAPACITY", "1"),
        help="Capacity for this capability (default: 1)",
    )
    parser.add_argument(
        "--disable-frame-skip",
        default=False,
        action="store_true",
        help="Disable adaptive frame skipping based on queue sizes (enabled by default)",
    )
    parser.add_argument(
        "--width",
        default=512,
        type=int,
        help="Default video width for processing",
    )
    parser.add_argument(
        "--height",
        default=512,
        type=int,
        help="Default video height for processing",
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
    
    # Add ComfyStream timeout filter to suppress verbose execution logging
    logging.getLogger("comfy.cmd.execution").addFilter(ComfyStreamTimeoutFilter())

    def force_print(*args, **kwargs):
        print(*args, **kwargs, flush=True)
        sys.stdout.flush()

    logger.info("Starting ComfyStream BYOC server with pytrickle StreamProcessor...")
    
    # Create frame processor with configuration
    frame_processor = ComfyStreamFrameProcessor(
        width=args.width,
        height=args.height,
        workspace=args.workspace,
        disable_cuda_malloc=True,
        gpu_only=True,
        preview_method='none',
        blacklist_nodes=["ComfyUI-Manager"],
        comfyui_inference_log_level=args.comfyui_inference_log_level
    )
    
    # Create frame skip configuration only if enabled
    frame_skip_config = None
    if args.disable_frame_skip:
        logger.info("Frame skipping disabled")
    else:
        frame_skip_config = FrameSkipConfig()
        logger.info("Frame skipping enabled: adaptive skipping based on queue sizes")
        
    # Create StreamProcessor with frame processor
    processor = StreamProcessor(
        video_processor=frame_processor.process_video_async,
        audio_processor=frame_processor.process_audio_async,
        model_loader=frame_processor.load_model,
        param_updater=frame_processor.update_params,
        on_stream_stop=frame_processor.on_stream_stop,
        # Align processor name with capability for consistent logs
        name=args.capability_name,
        port=int(args.port),
        host=args.host,
        frame_skip_config=frame_skip_config,
        # Ensure server metadata reflects the desired capability name
        capability_name=args.capability_name,
        #server_kwargs...
        route_prefix="/",
    )

    # Set the stream processor reference for text data publishing
    frame_processor.set_stream_processor(processor)
    
    # Create async startup function to load model
    async def load_model_on_startup(app):
        await processor._frame_processor.load_model()
        
    # Create async startup function for orchestrator registration
    async def register_orchestrator_startup(app):
        try:
            # Use command line arguments as primary source for URL, but always read secret from environment
            orch_url = args.orch_url
            orch_secret = os.getenv("ORCH_SECRET")

            if orch_url and orch_secret:
                # Set environment variables from command line arguments for RegisterCapability
                os.environ.update({
                    "CAPABILITY_NAME": args.capability_name,
                    "CAPABILITY_DESCRIPTION": args.capability_description,
                    "CAPABILITY_URL": args.capability_url,
                    "CAPABILITY_PRICE_PER_UNIT": args.capability_price_per_unit,
                    "CAPABILITY_PRICE_SCALING": args.capability_price_scaling,
                    "CAPABILITY_CAPACITY": args.capability_capacity,
                    "ORCH_URL": orch_url,
                    "ORCH_SECRET": orch_secret
                })

                result = await RegisterCapability.register(
                    logger=logger,
                    capability_name=args.capability_name
                )
                if result:
                    logger.info(f"Registered capability: {result.geturl()}")
        except Exception as e:
            logger.error(f"Orchestrator registration failed: {e}")

    # Add model loading and registration to startup hooks
    processor.server.app.on_startup.append(load_model_on_startup)
    processor.server.app.on_startup.append(register_orchestrator_startup)

    # Add warmup endpoint: accepts same body as prompts update
    async def warmup_handler(request):
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"Invalid JSON in warmup request: {e}")
            return web.json_response({"error": "Invalid JSON"}, status=400)
        try:
            # Inject sentinel to trigger warmup inside update_params on the model thread
            if isinstance(body, dict):
                body["warmup"] = True
            else:
                body = {"warmup": True}
            # Fire-and-forget: do not await warmup; update_params will schedule it
            asyncio.get_running_loop().create_task(frame_processor.update_params(body))
            return web.json_response({"status": "accepted"})
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    # Mount at same API namespace as StreamProcessor defaults
    processor.server.add_route("POST", "/api/stream/warmup", warmup_handler)
    
    # Run the processor
    processor.run()


if __name__ == "__main__":
    main()
