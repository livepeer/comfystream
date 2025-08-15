#!/usr/bin/env python3
"""
ComfyStream server using pytrickle StreamServer.

This module provides a streamlined ComfyStream server that uses pytrickle's
StreamServer infrastructure instead of the complex WebRTC/HTTP setup.
"""

import argparse
import asyncio
import logging
import os
import sys

# Ensure package imports work when running this file directly (python src/comfystream/trickle_app.py)
# Add the project "src" directory to sys.path so "comfystream" resolves as a package
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Initialize CUDA early to prevent potential core dumps
try:
    import torch  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.init()
except Exception:
    pass

from pytrickle import StreamServer, RegisterCapability
from comfystream.buffered_processor import BufferedComfyStreamProcessor
from comfystream.server.workflows import get_default_workflow, load_workflow

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the ComfyStream trickle server."""
    parser = argparse.ArgumentParser(description="Run ComfyStream with pytrickle StreamServer")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--workspace", required=True, help="ComfyUI workspace directory")
    parser.add_argument("--width", type=int, default=512, help="Default video width")
    parser.add_argument("--height", type=int, default=512, help="Default video height")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    parser.add_argument(
        "--warmup-workflow",
        default=None,
        help="Specify a workflow file name to use for pipeline warmup",
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
        "--capability-name",
        default="comfystream",
        help="Name of the capability for orchestrator registration",
    )
    parser.add_argument(
        "--disable-cuda-malloc",
        action="store_true",
        default=True,
        help="Disable CUDA malloc",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true", 
        default=True,
        help="Use GPU only",
    )
    
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure ComfyUI logging levels
    if args.comfyui_log_level:
        log_level = logging._nameToLevel.get(args.comfyui_log_level.upper())
        if log_level is not None:
            logging.getLogger("comfy").setLevel(log_level)

    server = None
    processor = None
    try:
        # Load workflow for pipeline initialization
        workflow = None
        if args.warmup_workflow:
            logger.info(f"Loading warmup workflow: {args.warmup_workflow}")
            logger.info(f"Current working directory: {os.getcwd()}")
            
            # Load workflow (loader handles package data and dev fallback)
            workflow = load_workflow(args.warmup_workflow)
        else:
            logger.info("Using default workflow")
            workflow = get_default_workflow()
        
        # Create the BufferedComfyStream processor
        logger.info("Creating BufferedComfyStream processor...")
        processor = BufferedComfyStreamProcessor(
            width=args.width,
            height=args.height,
            workspace=args.workspace,
            disable_cuda_malloc=args.disable_cuda_malloc,
            gpu_only=args.gpu_only,
            preview_method='none',
            comfyui_inference_log_level=args.comfyui_inference_log_level,
            default_workflow=workflow,
        )

        if not processor.ready:
            logger.error("Failed to initialize ComfyStream processor")

        # Apply the startup workflow and fully warm up ComfyUI models
        if workflow:
            logger.info("Applying startup workflow to processor")
            await processor.set_prompts(workflow)
            
            logger.info("Starting ComfyUI model warmup...")
            try:
                # Force model loading by running full startup warmup
                await processor.warm_models_for_startup(workflow)
                logger.info("ComfyUI models fully loaded and warmed up")
            except Exception as e:
                logger.error(f"Failed to warm up ComfyUI models: {e}")
                logger.warning("Models will load on first stream (may cause initial delay)")
                # Don't fail startup - continue and let models load on first stream
                pass
        else:
            logger.info("No startup workflow provided; processor will wait for first stream or warmup endpoint")
            logger.warning("No workflow provided - models will load on first stream")
        
        # Ensure processor has coordinated control methods for proper lifecycle management
        logger.info("BufferedComfyStream processor initialized with coordinated control support")

        # Register with orchestrator if URL provided
        try:
            result = await RegisterCapability.register(
                logger,
                capability_name=args.capability_name,
                capability_desc="ComfyStream AI video processing service"
            )
            if result and result != False:
                if hasattr(result, 'port') and result.port:
                    args.port = result.port
                    logger.info(f"Registered with orchestrator, using port {args.port}")
                else:
                    logger.info("Registered with orchestrator")
            else:
                logger.warning("Registration with orchestrator failed")
        except Exception as e:
            logger.warning(f"Registration with orchestrator failed: {e}")

        # Create and configure the StreamServer
        logger.info(f"🚀 Starting ComfyStream server on {args.host}:{args.port}")
        logger.info("📡 API endpoints:")
        logger.info(f"  - POST http://{args.host}:{args.port}/api/stream/start")
        logger.info(f"  - POST http://{args.host}:{args.port}/api/stream/stop")
        logger.info(f"  - POST http://{args.host}:{args.port}/api/stream/params")
        logger.info(f"  - GET  http://{args.host}:{args.port}/api/stream/status")
        logger.info(f"  - GET  http://{args.host}:{args.port}/health")

        # Create StreamServer with coordinated TrickleClient control
        # The StreamServer will now properly coordinate frame processor lifecycle
        # with TrickleProtocol start/stop events to prevent A/V sync issues
        server = StreamServer(
            frame_processor=processor,
            port=args.port,
            host=args.host,
            capability_name=args.capability_name,
            version="0.1.4",
            # Enable CORS for web clients
            cors_config={
                "*": {
                    "allow_credentials": True,
                    "expose_headers": "*",
                    "allow_headers": "*",
                    "allow_methods": ["GET", "POST", "OPTIONS"]
                }
            }
        )

        # Mark service startup complete and pipeline ready for health endpoint (unified state)
        try:
            server.state.set_state(server.state.state.WARMING_PIPELINE)
            server.state.set_startup_complete()
            server.state.set_state(server.state.state.READY)
        except Exception as e:
            logger.warning(f"Failed to update health manager readiness: {e}")

        # Run the server
        await server.run_forever()

    except KeyboardInterrupt:
        logger.info("🛑 ComfyStream server stopped by user")
    except Exception as e:
        logger.error(f"ComfyStream server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup with coordinated shutdown
        logger.info("Shutting down ComfyStream server...")
        if 'server' in locals() and server is not None:
            try:
                # Server cleanup will now properly coordinate client/protocol shutdown
                await server.stop()
                logger.info("StreamServer stopped cleanly")
            except Exception as e:
                logger.warning(f"Error during server shutdown: {e}")
        
        if 'processor' in locals() and processor is not None:
            try:
                await processor.cleanup(full_shutdown=True)
                logger.info("Frame processor cleanup completed")
            except Exception as e:
                logger.warning(f"Error during processor cleanup: {e}")
        
        logger.info("Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
