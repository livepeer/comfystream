"""
Trickle API routes for ComfyStream.

This module implements REST API endpoints for streaming with trickle protocol integration.
Handles ingress/egress to ComfyStream pipeline using the trickle-app package.
"""

import json
import logging
from typing import Optional
from aiohttp import web
from pydantic import ValidationError

from trickle_integration import TrickleStreamManager
from api_spec import (
    StreamStartRequest, 
    StreamParamsUpdateRequest, 
    StreamResponse, 
    StreamStatusResponse, 
    HealthCheckResponse, 
    ServiceInfoResponse
)

logger = logging.getLogger(__name__)
logger.info("Using trickle integration")

# Global stream manager instance - will be initialized in setup_trickle_routes
stream_manager: Optional[TrickleStreamManager] = None

async def start_stream(request):
    """Start a new trickle stream."""
    try:
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
        data = await request.json()
        
        # Debug logging to see what data is being received
        logger.info(f"[DEBUG] Received request data: {data}")
        logger.info(f"[DEBUG] Request data type: {type(data)}")
        logger.info(f"[DEBUG] Request data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        try:
            stream_request = StreamStartRequest(**data)
        except ValidationError as e:
            logger.error(f"[DEBUG] Pydantic validation failed: {e}")
            logger.error(f"[DEBUG] Validation errors: {e.errors()}")
            return web.json_response({
                'error': 'Validation error',
                'details': e.errors()
            }, status=400)
        except Exception as e:
            logger.error(f"[DEBUG] Unexpected error during validation: {e}")
            return web.json_response({
                'error': 'Unexpected validation error',
                'details': str(e)
            }, status=400)
            
        request_id = stream_request.gateway_request_id
        params = stream_request.get_comfy_params()
        width = params.width
        height = params.height
        prompts = params.prompts
        pipeline = request.app.get('pipeline')
        if not pipeline:
            return web.json_response({'error': 'Pipeline not initialized in app'}, status=500)
        if pipeline.width != width or pipeline.height != height:
            pipeline.width = width
            pipeline.height = height
            logger.info(f"Updated pipeline resolution to {width}x{height} for stream {request_id}")
        # Set prompts - prompts is already a single workflow object, wrap in array for set_prompts
        try:
            await pipeline.set_prompts([prompts])
            logger.info(f"Set prompts for stream {request_id}")
        except Exception as e:
            logger.error(f"Invalid prompt for stream {request_id}: {e}")
            return web.json_response({'error': f'Invalid prompt: {str(e)}'}, status=400)
        
        # Log if control_url is not provided
        if not stream_request.control_url:
            logger.info(f"Control URL not provided for stream {request_id} - parameter updates will not be read from orchestrator")
        
        success = await stream_manager.create_stream(
            request_id=request_id,
            subscribe_url=stream_request.subscribe_url,
            publish_url=stream_request.publish_url,
            control_url=stream_request.control_url or "",
            events_url=stream_request.events_url or "",
            pipeline=pipeline,
            width=width,
            height=height
        )
        if success:
            response_data = StreamResponse(
                status='success',
                message=f'Stream {request_id} started successfully',
                request_id=request_id,
                config={
                    'subscribe_url': stream_request.subscribe_url,
                    'publish_url': stream_request.publish_url,
                    'width': width,
                    'height': height
                }
            )
            return web.json_response(response_data.model_dump())
        else:
            response_data = StreamResponse(
                status='error',
                message=f'Failed to start stream {request_id}'
            )
            return web.json_response(response_data.model_dump(), status=500)
    except json.JSONDecodeError as e:
        logger.error(f"[DEBUG] JSON decode error: {e}")
        return web.json_response({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        response_data = StreamResponse(
            status='error',
            message=f'Internal server error: {str(e)}'
        )
        return web.json_response(response_data.model_dump(), status=500)

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
            # Add current prompts info to status for debugging
            handler = stream_manager.handlers.get(request_id)
            if handler and handler.pipeline:
                try:
                    current_prompts = handler.pipeline.client.current_prompts
                    status['current_prompts'] = current_prompts
                    status['prompts_count'] = len(current_prompts)
                except Exception as e:
                    logger.debug(f"Could not get current prompts: {e}")
                    status['prompts_error'] = str(e)
            
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
            response_data = StreamStatusResponse(
                processing_active=False,
                stream_count=0,
                message='No active streams'
            )
            return web.json_response(response_data.model_dump())
        
        # Return status compatible with webrtc-worker format
        stream_id = next(iter(streams.keys()))  # Get first stream
        stream_status = streams[stream_id]
        
        response_data = StreamStatusResponse(
            processing_active=stream_status.get('running', False),
            stream_count=len(streams),
            current_stream=stream_status,
            all_streams=streams
        )
        return web.json_response(response_data.model_dump())
        
    except Exception as e:
        logger.error(f"Error getting current stream status: {e}")
        return web.json_response({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }, status=500)

async def update_stream_params(request):
    """Update parameters for a specific stream."""
    try:
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
        request_id = request.match_info.get('request_id')
        if not request_id:
            return web.json_response({'error': 'Missing request_id'}, status=400)
        stream_status = await stream_manager.get_stream_status(request_id)
        if not stream_status:
            return web.json_response({'error': f'Stream {request_id} not found'}, status=404)
        data = await request.json()
        try:
            params_request = StreamParamsUpdateRequest(**data)
        except ValidationError as e:
            logger.error(f"[DEBUG] Pydantic validation failed in update_stream_params: {e}")
            logger.error(f"[DEBUG] Validation errors: {e.errors()}")
            return web.json_response({
                'error': 'Validation error',
                'details': e.errors()
            }, status=400)
        except Exception as e:
            logger.error(f"[DEBUG] Unexpected error during validation in update_stream_params: {e}")
            return web.json_response({
                'error': 'Unexpected validation error',
                'details': str(e)
            }, status=400)
        handler = stream_manager.handlers.get(request_id)
        if not handler:
            return web.json_response({'error': f'Stream handler {request_id} not found'}, status=404)
        # Use the validated prompts directly from the request
        # HTTP API uses 'prompts' (plural) - single workflow object
        await handler._handle_control_message({'prompts': params_request.prompts, 'width': params_request.width, 'height': params_request.height})
        response_data = StreamResponse(
            status='success',
            message=f'Parameters updated for stream {request_id}',
            request_id=request_id
        )
        return web.json_response(response_data.model_dump())
    except json.JSONDecodeError:
        return web.json_response({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Error updating stream parameters: {e}")
        response_data = StreamResponse(
            status='error',
            message=f'Internal server error: {str(e)}'
        )
        return web.json_response(response_data.model_dump(), status=500)

async def health_check(request):
    """Health check endpoint (webrtc-worker compatible)."""
    try:
        # Check if stream manager is initialized
        manager_healthy = stream_manager is not None
        
        response_data = HealthCheckResponse(
            status='healthy',
            service='trickle-stream-processor',
            version='1.0.0',
            stream_manager_ready=manager_healthy
        )
        return web.json_response(response_data.model_dump())
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        response_data = HealthCheckResponse(
            status='unhealthy',
            service='trickle-stream-processor',
            version='1.0.0',
            error=str(e)
        )
        return web.json_response(response_data.model_dump(), status=500)

async def get_stream_prompts(request):
    """Get current prompts for a specific stream (debugging)."""
    try:
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
        
        request_id = request.match_info.get('request_id')
        if not request_id:
            return web.json_response({'error': 'Missing request_id'}, status=400)
        
        handler = stream_manager.handlers.get(request_id)
        if not handler:
            return web.json_response({'error': f'Stream {request_id} not found'}, status=404)
        
        try:
            current_prompts = handler.pipeline.client.current_prompts
            running_prompts = list(handler.pipeline.client.running_prompts.keys())
            
            return web.json_response({
                'request_id': request_id,
                'current_prompts': current_prompts,
                'running_prompts': running_prompts,
                'prompts_count': len(current_prompts)
            })
        except Exception as e:
            return web.json_response({
                'error': f'Error getting prompts: {str(e)}',
                'request_id': request_id
            }, status=500)
            
    except Exception as e:
        logger.error(f"Error getting stream prompts: {e}")
        return web.json_response({
            'error': f'Internal server error: {str(e)}'
        }, status=500)

async def set_stream_prompts(request):
    """Set prompts for a specific stream directly (debugging)."""
    try:
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
        
        request_id = request.match_info.get('request_id')
        if not request_id:
            return web.json_response({'error': 'Missing request_id'}, status=400)
        
        handler = stream_manager.handlers.get(request_id)
        if not handler:
            return web.json_response({'error': f'Stream {request_id} not found'}, status=404)
        
        data = await request.json()
        prompts = data.get('prompts')
        if not prompts:
            return web.json_response({'error': 'Missing prompts field'}, status=400)
        
        try:
            # Set prompts directly on the pipeline
            await handler.pipeline.set_prompts([prompts])
            logger.info(f"[Debug] Manually set prompts for stream {request_id}")
            
            return web.json_response({
                'status': 'success',
                'message': f'Prompts set for stream {request_id}',
                'request_id': request_id
            })
        except Exception as e:
            logger.error(f"Error setting prompts for stream {request_id}: {e}")
            return web.json_response({
                'error': f'Error setting prompts: {str(e)}',
                'request_id': request_id
            }, status=500)
            
    except Exception as e:
        logger.error(f"Error in set_stream_prompts: {e}")
        return web.json_response({
            'error': f'Internal server error: {str(e)}'
        }, status=500)

async def root_info(request):
    """Root endpoint with service info (webrtc-worker compatible)."""
    try:
        response_data = ServiceInfoResponse(
            service='Trickle Stream Processor',
            version='1.0.0',
            description='ComfyStream trickle streaming processor for real-time video processing',
            capabilities=['video-processing', 'trickle-streaming'],
            endpoints={
                'start': 'POST /stream/start - Start stream processing',
                'stop': 'POST /stream/stop - Stop current stream processing',
                'stop_by_id': 'POST /stream/{request_id}/stop - Stop specific stream',
                'status': 'GET /stream/status - Get current stream status',
                'status_by_id': 'GET /stream/{request_id}/status - Get specific stream status',
                'params': 'POST /stream/{request_id}/params - Update stream prompts',
                'prompts': 'GET /stream/{request_id}/prompts - Get current prompts (debug)',
                'set_prompts': 'POST /stream/{request_id}/prompts - Set prompts directly (debug)',
                'list': 'GET /streams - List all active streams',
                'health': 'GET /health - Health check',
                'live_video': 'POST /live-video-to-video - Start live video processing'
            }
        )
        return web.json_response(response_data.model_dump())
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
        'pipeline': app.get('pipeline'),  # This will be set later during startup
        'health_manager': app.get('health_manager')  # Pass health manager for stream tracking
    })

    # Core trickle streaming routes
    cors.add(app.router.add_post("/stream/start", start_stream))
    cors.add(app.router.add_post("/stream/{request_id}/stop", stop_stream))
    cors.add(app.router.add_get("/stream/{request_id}/status", get_stream_status))
    cors.add(app.router.add_get("/streams", list_streams))
    cors.add(app.router.add_post("/stream/{request_id}/params", update_stream_params))
    cors.add(app.router.add_get("/stream/{request_id}/prompts", get_stream_prompts))
    cors.add(app.router.add_post("/stream/{request_id}/prompts", set_stream_prompts))
    
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