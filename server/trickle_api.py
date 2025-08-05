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

from trickle_stream_manager import TrickleStreamManager
from api_spec import StreamStartRequest, StreamParamsUpdateRequest
from pytrickle.api_spec import (
    StreamResponse, 
    StreamStatusResponse, 
    HealthCheckResponse, 
    ServiceInfoResponse
)

logger = logging.getLogger(__name__)
logger.info("Using trickle integration")

async def start_stream(request):
    """Start a new trickle stream."""
    try:
        stream_manager = request.app.get('stream_manager')
        if not stream_manager:
            return web.json_response({'error': 'Stream manager not initialized'}, status=500)
        data = await request.json()
        logger.info(f"Received start stream request with data: {json.dumps(data, indent=2)}")
        try:
            stream_request = StreamStartRequest(**data)
            logger.info(f"Validated stream start request: {stream_request.model_dump_json(indent=2)}")
        except ValidationError as e:
            logger.error(f"Pydantic validation failed: {e}")
            return web.json_response({
                'error': 'Validation error'
            }, status=400)
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return web.json_response({
                'error': 'Unexpected validation error'
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
            await pipeline.set_prompts(prompts)
            logger.info(f"Set prompts for stream {request_id}")
        except Exception as e:
            logger.error(f"Invalid prompt for stream {request_id}: {e}")
            return web.json_response({'error': 'Invalid prompt'}, status=400)
        
        # Log if control_url is not provided
        if not stream_request.control_url:
            logger.info(f"Control URL not provided for stream {request_id} - parameter updates will not be read from orchestrator")
        
        success = await stream_manager.create_stream(
            request_id=request_id,
            subscribe_url=stream_request.subscribe_url,
            publish_url=stream_request.publish_url,
            control_url=stream_request.control_url or "",
            events_url=stream_request.events_url or "",
            data_url=stream_request.data_url,
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
                    'data_url': stream_request.data_url,
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
        logger.error(f"JSON decode error: {e}")
        return web.json_response({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        response_data = StreamResponse(
            status='error',
            message='Internal server error'
        )
        return web.json_response(response_data.model_dump(), status=500)

async def stop_stream(request):
    """Stop a trickle stream."""
    try:
        stream_manager = request.app.get('stream_manager')
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
            'message': 'Internal server error'
        }, status=500)

async def get_stream_status(request):
    """Get status of a trickle stream."""
    try:
        stream_manager = request.app.get('stream_manager')
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
            'message': 'Internal server error'
        }, status=500)

async def list_streams(request):
    """List all active trickle streams."""
    try:
        stream_manager = request.app.get('stream_manager')
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
            'message': 'Internal server error'
        }, status=500)

async def stop_current_stream(request):
    """Stop the current stream (webrtc-worker compatible endpoint)."""
    try:
        stream_manager = request.app.get('stream_manager')
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
            'message': 'Internal server error'
        }, status=500)

async def get_current_stream_status(request):
    """Get current stream status (webrtc-worker compatible endpoint)."""
    try:
        stream_manager = request.app.get('stream_manager')
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
            'message': 'Internal server error'
        }, status=500)

async def update_stream_params(request):
    """Update parameters for a specific stream."""
    try:
        stream_manager = request.app.get('stream_manager')
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
            logger.error(f"Pydantic validation failed in update_stream_params: {e}")
            return web.json_response({
                'error': 'Validation error'
            }, status=400)
        except Exception as e:
            logger.error(f"Unexpected error during validation in update_stream_params: {e}")
            return web.json_response({
                'error': 'Unexpected validation error'
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
            message='Internal server error'
        )
        return web.json_response(response_data.model_dump(), status=500)

async def health_check(request):
    """Health check endpoint (webrtc-worker compatible)."""
    try:
        # Check if stream manager is initialized
        stream_manager = request.app.get('stream_manager')
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
            error='Internal server error'
        )
        return web.json_response(response_data.model_dump(), status=500)

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
            'error': 'Internal server error'
        }, status=500)

def setup_trickle_routes(app, cors):
    """Setup trickle API routes.
    
    Args:
        app: The aiohttp web application
        cors: The CORS setup object
    """
    stream_manager = TrickleStreamManager(app_context={
        'warm_pipeline': app.get('warm_pipeline', False),
        'workspace': app.get('workspace'),
        'pipeline': app.get('pipeline'),  # This will be set later during startup
        'health_manager': app.get('health_manager')  # Pass health manager for stream tracking
    })
    
    # CRITICAL: Add the stream manager reference to its own app_context so streams can remove themselves during error cleanup
    stream_manager.app_context['stream_manager'] = stream_manager
    
    # Store stream manager in app context for dependency injection
    app['stream_manager'] = stream_manager

    # Core trickle streaming routes
    cors.add(app.router.add_post("/stream/start", start_stream))
    cors.add(app.router.add_post("/stream/{request_id}/stop", stop_stream))
    cors.add(app.router.add_get("/stream/{request_id}/status", get_stream_status))
    cors.add(app.router.add_get("/streams", list_streams))
    cors.add(app.router.add_post("/stream/{request_id}/params", update_stream_params))
    
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
async def cleanup_trickle_streams(app):
    """Cleanup all trickle streams on app shutdown."""
    stream_manager = app.get('stream_manager')
    if stream_manager:
        await stream_manager.cleanup_all()