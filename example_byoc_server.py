#!/usr/bin/env python3
"""
Example BYOC (Bring Your Own Container) Server for ComfyStream

This example demonstrates how to start and use the ComfyStream BYOC server
which provides a reverse server interface compatible with Livepeer orchestrators.

Usage:
    python example_byoc_server.py --workspace /path/to/comfyui/workspace

Testing the server:
    # Health check
    curl http://localhost:5000/health

    # Text reversal example (from BYOC docs)
    curl -X POST http://localhost:5000/process/request/text-reversal \
      -H "Content-Type: application/json" \
      -H "Livepeer: eyJyZXF1ZXN0IjogIntcInJ1blwiOiBcImVjaG9cIn0iLCAiY2FwYWJpbGl0eSI6ICJ0ZXh0LXJldmVyc2FsIiwgInRpbWVvdXRfc2Vjb25kcyI6IDMwfQ==" \
      -d '{"text":"Hello, ComfyStream BYOC!"}'

    # Start a ComfyStream video processing stream
    curl -X POST http://localhost:5000/stream/start \
      -H "Content-Type: application/json" \
      -d '{
        "prompts": [{"prompt": "a beautiful landscape"}],
        "stream_url": "http://your-trickle-endpoint/stream",
        "width": 512,
        "height": 512
      }'

    # List active streams
    curl http://localhost:5000/streams

    # Stop a stream (replace manifest_id with actual ID)
    curl -X DELETE http://localhost:5000/stream/{manifest_id}
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from src.comfystream.server import start_byoc_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """Main function to start the BYOC server"""
    parser = argparse.ArgumentParser(description="ComfyStream BYOC Server Example")
    parser.add_argument("--workspace", required=True, help="Path to ComfyUI workspace")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate workspace path
    workspace_path = Path(args.workspace)
    if not workspace_path.exists():
        logger.error(f"Workspace path does not exist: {workspace_path}")
        return 1
        
    if not workspace_path.is_dir():
        logger.error(f"Workspace path is not a directory: {workspace_path}")
        return 1
    
    logger.info(f"Starting ComfyStream BYOC Server")
    logger.info(f"Workspace: {workspace_path}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    
    # Global variable to hold server instance for signal handling
    server = None
    
    def signal_handler(sig, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {sig}, shutting down...")
        if server:
            asyncio.create_task(server.stop())
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the BYOC server
        server = await start_byoc_server(
            workspace=str(workspace_path),
            host=args.host,
            port=args.port
        )
        
        logger.info("BYOC Server started successfully!")
        logger.info("")
        logger.info("Available endpoints:")
        logger.info(f"  Health check: http://{args.host}:{args.port}/health")
        logger.info(f"  BYOC capability: http://{args.host}:{args.port}/process/request/{{capability}}")
        logger.info(f"  Start stream: http://{args.host}:{args.port}/stream/start")
        logger.info(f"  List streams: http://{args.host}:{args.port}/streams")
        logger.info(f"  Stop stream: http://{args.host}:{args.port}/stream/{{manifest_id}}")
        logger.info("")
        logger.info("Example BYOC request (text reversal):")
        logger.info('curl -X POST http://localhost:5000/process/request/text-reversal \\')
        logger.info('  -H "Content-Type: application/json" \\')
        logger.info('  -H "Livepeer: eyJyZXF1ZXN0IjogIntcInJ1blwiOiBcImVjaG9cIn0iLCAiY2FwYWJpbGl0eSI6ICJ0ZXh0LXJldmVyc2FsIiwgInRpbWVvdXRfc2Vjb25kcyI6IDMwfQ==" \\')
        logger.info('  -d \'{"text":"Hello, ComfyStream BYOC!"}\' ')
        logger.info("")
        logger.info("Press Ctrl+C to stop the server")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error running BYOC server: {e}")
        return 1
    finally:
        if server:
            await server.stop()
            logger.info("BYOC Server stopped")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 