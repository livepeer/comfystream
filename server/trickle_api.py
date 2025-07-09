"""
Trickle API routes for ComfyStream.

This module implements REST API endpoints for streaming with trickle protocol integration.
Handles ingress/egress to ComfyStream pipeline using the trickle-app package.
"""

import asyncio
import json
import logging
from typing import Dict, Optional
from aiohttp import web
from comfystream.pipeline import Pipeline

# Import the actual trickle integration, not the fallback
try:
    from trickle_integration import TrickleStreamManager
    logger = logging.getLogger(__name__)
    logger.info("Using real trickle integration")
except ImportError:
    from trickle_integration_fallback import TrickleStreamManager
    logger = logging.getLogger(__name__)
    logger.warning("Trickle integration not available (No module named 'trickle_app'), using mock implementation")

# Global stream manager instance - will be initialized in setup_trickle_routes
stream_manager: Optional[TrickleStreamManager] = None

async def start_stream(request):
    """Start a new trickle stream.
    
    Expected request format:
    {
        "subscribe_url": "http://192.168.10.61:3389/sample",
        "publish_url": "http://192.168.10.61:3389/sample-output", 
        "gateway_request_id": "sample2",
        "params": {
            "width": 512,
            "height": 512,
            "prompt": "{\"1\":{\"inputs\":{\"images\":[\"2\",0]},\"class_type\":\"SaveTensor\"},...}"
        }
    }
    """
    try:
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
            
        data = await request.json()
        
        # Validate required fields
        required_fields = ['subscribe_url', 'publish_url', 'gateway_request_id']
        for field in required_fields:
            if field not in data:
                return web.json_response(
                    {'error': f'Missing required field: {field}'}, 
                    status=400
                )
        
        # Extract configuration
        request_id = data['gateway_request_id']
        params = data.get('params', {})
        
        width = params.get('width', 512)
        height = params.get('height', 512)
        
        # Create pipeline for this stream
        pipeline = Pipeline(
            width=width,
            height=height,
            cwd=request.app.get('workspace'),
            disable_cuda_malloc=True,
            gpu_only=True,
            preview_method='none'
        )
        
        # Set prompts if provided
        if 'prompt' in params:
            try:
                prompt_data = json.loads(params['prompt'])
                await pipeline.set_prompts(prompt_data)
                logger.info(f"Set prompts for stream {request_id}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid prompt JSON for stream {request_id}: {e}")
                return web.json_response(
                    {'error': f'Invalid prompt JSON: {str(e)}'}, 
                    status=400
                )
        else:
            # Set a default simple inversion workflow for testing
            default_workflow = {
                "1": {
                    "inputs": {
                        "images": ["2", 0]
                    },
                    "class_type": "SaveTensor"
                },
                "2": {
                    "inputs": {},
                    "class_type": "LoadTensor"
                },
                "3": {
                    "inputs": {
                        "images": ["2", 0]
                    },
                    "class_type": "ImageInvert"
                }
            }
            await pipeline.set_prompts(default_workflow)
            logger.info(f"Set default inversion workflow for stream {request_id}")
        
        # Start the stream
        success = await stream_manager.create_stream(
            request_id=request_id,
            subscribe_url=data['subscribe_url'],
            publish_url=data['publish_url'],
            pipeline=pipeline,
            width=width,
            height=height
        )
        
        if success:
            return web.json_response({
                'status': 'success',
                'message': f'Stream {request_id} started successfully',
                'request_id': request_id,
                'config': {
                    'subscribe_url': data['subscribe_url'],
                    'publish_url': data['publish_url'],
                    'width': width,
                    'height': height
                }
            })
        else:
            # Cleanup pipeline if stream creation failed
            await pipeline.cleanup()
            return web.json_response({
                'status': 'error',
                'message': f'Failed to start stream {request_id}'
            }, status=500)
            
    except json.JSONDecodeError:
        return web.json_response({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        return web.json_response({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }, status=500)

async def stop_stream(request):
    """Stop a trickle stream."""
    try:
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
            
        request_id = request.match_info.get('request_id')
        if not request_id:
            return web.json_response({'error': 'Missing request_id'}, status=400)
        
        success = await stream_manager.stop_stream(request_id)
        
        if success:
            return web.json_response({
                'status': 'success',
                'message': f'Stream {request_id} stopped successfully'
            })
        else:
            return web.json_response({
                'status': 'error',
                'message': f'Stream {request_id} not found or failed to stop'
            }, status=404)
            
    except Exception as e:
        logger.error(f"Error stopping stream: {e}")
        return web.json_response({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }, status=500)

async def get_stream_status(request):
    """Get status of a trickle stream."""
    try:
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
            
        request_id = request.match_info.get('request_id')
        if not request_id:
            return web.json_response({'error': 'Missing request_id'}, status=400)
        
        status = await stream_manager.get_stream_status(request_id)
        
        if status:
            return web.json_response(status)
        else:
            return web.json_response({
                'error': f'Stream {request_id} not found'
            }, status=404)
            
    except Exception as e:
        logger.error(f"Error getting stream status: {e}")
        return web.json_response({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }, status=500)

async def list_streams(request):
    """List all active trickle streams."""
    try:
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
            
        streams = await stream_manager.list_streams()
        return web.json_response({
            'streams': streams,
            'count': len(streams)
        })
        
    except Exception as e:
        logger.error(f"Error listing streams: {e}")
        return web.json_response({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }, status=500)

def setup_trickle_routes(app, cors):
    """Setup trickle API routes.
    
    Args:
        app: The aiohttp web application
        cors: The CORS setup object
    """
    global stream_manager
    stream_manager = TrickleStreamManager(app_context={'warm_pipeline': app.get('warm_pipeline', False)})

    # Trickle streaming routes
    cors.add(app.router.add_post("/stream/start", start_stream))
    cors.add(app.router.add_post("/stream/{request_id}/stop", stop_stream))
    cors.add(app.router.add_get("/stream/{request_id}/status", get_stream_status))
    cors.add(app.router.add_get("/streams", list_streams))
    
    # Alias for live-video-to-video endpoint (same as stream/start)
    cors.add(app.router.add_post("/live-video-to-video", start_stream))
    
    logger.info("Trickle API routes registered")

# Cleanup function for app shutdown
async def cleanup_trickle_streams():
    """Cleanup all trickle streams on app shutdown."""
    if stream_manager:
        await stream_manager.cleanup_all()