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

# Import the trickle integration - trickle-app should always be installed
from trickle_integration import TrickleStreamManager

logger = logging.getLogger(__name__)
logger.info("Using trickle integration")

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
        
        # Use the shared pipeline from app.py instead of creating a new one
        pipeline = request.app.get('pipeline')
        if not pipeline:
            return web.json_response({
                'error': 'Pipeline not initialized in app'
            }, status=500)
        
        # Update pipeline resolution if different from current
        if pipeline.width != width or pipeline.height != height:
            pipeline.width = width
            pipeline.height = height
            logger.info(f"Updated pipeline resolution to {width}x{height} for stream {request_id}")
        
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
        
        # Start the stream using the shared pipeline
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

async def stop_current_stream(request):
    """Stop the current stream (webrtc-worker compatible endpoint)."""
    try:
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
        
        # Get all streams and stop them (simple approach for single stream scenarios)
        streams = await stream_manager.list_streams()
        
        if not streams:
            return web.json_response({
                'status': 'error',
                'message': 'No active streams found'
            }, status=404)
        
        # Stop all streams (typically there should only be one in process capability mode)
        stopped_count = 0
        for stream_id in streams.keys():
            success = await stream_manager.stop_stream(stream_id)
            if success:
                stopped_count += 1
        
        if stopped_count > 0:
            return web.json_response({
                'status': 'stopped',
                'message': f'Stopped {stopped_count} stream(s)'
            })
        else:
            return web.json_response({
                'status': 'error',
                'message': 'Failed to stop streams'
            }, status=500)
            
    except Exception as e:
        logger.error(f"Error stopping current stream: {e}")
        return web.json_response({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }, status=500)

async def get_current_stream_status(request):
    """Get current stream status (webrtc-worker compatible endpoint)."""
    try:
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
        
        streams = await stream_manager.list_streams()
        
        if not streams:
            return web.json_response({
                'processing_active': False,
                'stream_count': 0,
                'message': 'No active streams'
            })
        
        # Return status compatible with webrtc-worker format
        stream_id = next(iter(streams.keys()))  # Get first stream
        stream_status = streams[stream_id]
        
        return web.json_response({
            'processing_active': stream_status.get('running', False),
            'stream_count': len(streams),
            'current_stream': stream_status,
            'all_streams': streams
        })
        
    except Exception as e:
        logger.error(f"Error getting current stream status: {e}")
        return web.json_response({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }, status=500)

async def health_check(request):
    """Health check endpoint (webrtc-worker compatible)."""
    try:
        # Check if stream manager is initialized
        manager_healthy = stream_manager is not None
        
        return web.json_response({
            'status': 'healthy',
            'service': 'trickle-stream-processor',
            'version': '1.0.0',
            'stream_manager_ready': manager_healthy
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return web.json_response({
            'status': 'unhealthy',
            'service': 'trickle-stream-processor',
            'error': str(e)
        }, status=500)

async def root_info(request):
    """Root endpoint with service info (webrtc-worker compatible)."""
    try:
        return web.json_response({
            'service': 'Trickle Stream Processor',
            'version': '1.0.0',
            'description': 'ComfyStream trickle streaming processor for real-time video processing',
            'capabilities': ['video-processing', 'trickle-streaming'],
            'endpoints': {
                'start': 'POST /stream/start - Start stream processing',
                'stop': 'POST /stream/stop - Stop current stream processing',
                'stop_by_id': 'POST /stream/{request_id}/stop - Stop specific stream',
                'status': 'GET /stream/status - Get current stream status',
                'status_by_id': 'GET /stream/{request_id}/status - Get specific stream status',
                'list': 'GET /streams - List all active streams',
                'health': 'GET /health - Health check',
                'live_video': 'POST /live-video-to-video - Start live video processing'
            }
        })
    except Exception as e:
        logger.error(f"Error in root info: {e}")
        return web.json_response({
            'service': 'Trickle Stream Processor',
            'status': 'error',
            'error': str(e)
        }, status=500)

def setup_trickle_routes(app, cors):
    """Setup trickle API routes.
    
    Args:
        app: The aiohttp web application
        cors: The CORS setup object
    """
    global stream_manager
    stream_manager = TrickleStreamManager(app_context={
        'warm_pipeline': app.get('warm_pipeline', False),
        'workspace': app.get('workspace'),
        'pipeline': app.get('pipeline')  # This will be set later during startup
    })

    # Core trickle streaming routes
    cors.add(app.router.add_post("/stream/start", start_stream))
    cors.add(app.router.add_post("/stream/{request_id}/stop", stop_stream))
    cors.add(app.router.add_get("/stream/{request_id}/status", get_stream_status))
    cors.add(app.router.add_get("/streams", list_streams))
    
    # Process capability compatible routes (for byoc worker compatibility)
    cors.add(app.router.add_post("/stream/stop", stop_current_stream))
    cors.add(app.router.add_get("/stream/status", get_current_stream_status))
    
    # Service info routes
    cors.add(app.router.add_get("/health", health_check))
    cors.add(app.router.add_get("/", root_info))
    
    # Alias for live-video-to-video endpoint (same as stream/start)
    cors.add(app.router.add_post("/live-video-to-video", start_stream))
    
    logger.info("Trickle API routes registered")

# Cleanup function for app shutdown
async def cleanup_trickle_streams():
    """Cleanup all trickle streams on app shutdown."""
    if stream_manager:
        await stream_manager.cleanup_all()