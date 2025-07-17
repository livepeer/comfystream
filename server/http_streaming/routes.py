"""
HTTP streaming routes for ComfyStream.

This module contains the routes for HTTP streaming.
"""
import asyncio
import logging
import io
import json
import av
from aiohttp import web
from frame_buffer import FrameBuffer
from .tokens import cleanup_expired_sessions, validate_token, create_stream_token

logger = logging.getLogger(__name__)

async def process_segment(request):
    """Process a video segment using PyAV and the ComfyUI pipeline
    
    Extracts frames from the uploaded video segment, processes them through
    the pipeline, and returns a new video segment with the processed frames.
    """
    import time
    start_time = time.time()
    
    pipeline = request.app['pipeline']
    
    try:
        # Get the multipart data
        reader = await request.multipart()
        
        segment_data = None
        segment_index = None
        timestamp = None
        prompts = None
        resolution = None
        
        # Process multipart fields
        async for field in reader:
            if field.name == 'segment':
                segment_data = await field.read()
            elif field.name == 'segmentIndex':
                segment_index = int(await field.text())
            elif field.name == 'timestamp':
                timestamp = int(await field.text())
            elif field.name == 'prompts':
                prompts_text = await field.text()
                try:
                    prompts = json.loads(prompts_text)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse prompts JSON")
            elif field.name == 'resolution':
                resolution_text = await field.text()
                try:
                    resolution = json.loads(resolution_text)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse resolution JSON")
        
        if not segment_data:
            return web.Response(status=400, text="No segment data provided")
        
        logger.info(f"Processing segment {segment_index} ({len(segment_data)} bytes)")
        
        # Update pipeline with prompts if provided
        if prompts:
            await pipeline.update_prompts(prompts)
        
        # Create input container from segment data
        input_container = av.open(io.BytesIO(segment_data))
        
        # Find the first video and audio streams
        video_stream = None
        audio_stream = None
        
        for stream in input_container.streams:
            if stream.type == 'video' and video_stream is None:
                video_stream = stream
            elif stream.type == 'audio' and audio_stream is None:
                audio_stream = stream
        
        if not video_stream:
            return web.Response(status=400, text="No video stream found in segment")
        
        logger.info(f"Input video: {video_stream.width}x{video_stream.height}, "
                   f"codec: {video_stream.codec_context.name}, "
                   f"fps: {video_stream.average_rate}")
        
        # Determine output format based on input codec
        input_video_codec = video_stream.codec_context.name
        input_format = input_container.format.name
        
        # Map input codec to appropriate output codec
        video_codec_map = {
            'h264': 'libx264',
            'h265': 'libx265', 
            'hevc': 'libx265',
            'vp8': 'libvpx',
            'vp9': 'libvpx-vp9',
            'av1': 'libaom-av1'
        }
        
        # Use input codec or fallback to vp9 for webm
        output_video_codec = video_codec_map.get(input_video_codec, input_video_codec)
        
        # Verify codec is available, fallback to safe defaults
        try:
            # Test if codec is available by creating a test stream
            test_container = av.open(io.BytesIO(), mode='w', format='null')
            test_stream = test_container.add_stream(output_video_codec)
            test_container.close()
        except Exception as e:
            logger.warning(f"Codec {output_video_codec} not available, falling back to libx264: {e}")
            output_video_codec = 'libx264'
        
        # Determine output format - prefer input format if supported, otherwise webm
        supported_formats = ['webm', 'mp4', 'mkv', 'avi']
        output_format = input_format if input_format in supported_formats else 'webm'
        
        # If using mp4, ensure codec compatibility
        if output_format == 'mp4' and output_video_codec in ['libvpx', 'libvpx-vp9']:
            output_video_codec = 'libx264'  # VP8/VP9 not widely supported in MP4
        
        logger.info(f"Using output format: {output_format}, video codec: {output_video_codec}")
        
        # Create output container in memory
        output_buffer = io.BytesIO()
        output_container = av.open(output_buffer, mode='w', format=output_format)
        
        # Create output video stream matching input properties
        output_video_stream = output_container.add_stream(output_video_codec, rate=video_stream.average_rate)
        output_video_stream.width = video_stream.width
        output_video_stream.height = video_stream.height
        
        # Use input pixel format if available, otherwise default
        if hasattr(video_stream.codec_context, 'pix_fmt') and video_stream.codec_context.pix_fmt:
            output_video_stream.pix_fmt = video_stream.codec_context.pix_fmt
        else:
            output_video_stream.pix_fmt = 'yuv420p'
            
        # Copy bitrate and other encoding parameters
        if hasattr(video_stream.codec_context, 'bit_rate') and video_stream.codec_context.bit_rate:
            output_video_stream.bit_rate = video_stream.codec_context.bit_rate
        else:
            output_video_stream.bit_rate = 2500000  # fallback bitrate
        
        # Create output audio stream if input has audio
        output_audio_stream = None
        if audio_stream:
            input_audio_codec = audio_stream.codec_context.name
            
            # Map input audio codec to appropriate output codec
            audio_codec_map = {
                'aac': 'aac',
                'mp3': 'libmp3lame',
                'opus': 'libopus',
                'vorbis': 'libvorbis',
                'flac': 'flac',
                'pcm_s16le': 'pcm_s16le'
            }
            
            # Use input codec or fallback based on container format
            output_audio_codec = audio_codec_map.get(input_audio_codec, input_audio_codec)
            
            # Container-specific codec adjustments
            if output_format == 'webm' and output_audio_codec not in ['libopus', 'libvorbis']:
                output_audio_codec = 'libopus'  # WebM prefers Opus
            elif output_format == 'mp4' and output_audio_codec not in ['aac', 'libmp3lame']:
                output_audio_codec = 'aac'  # MP4 prefers AAC
            
            # Verify audio codec is available
            try:
                test_container = av.open(io.BytesIO(), mode='w', format='null')
                test_stream = test_container.add_stream(output_audio_codec)
                test_container.close()
            except Exception as e:
                logger.warning(f"Audio codec {output_audio_codec} not available, falling back to aac: {e}")
                output_audio_codec = 'aac'
            
            logger.info(f"Input audio: {audio_stream.rate}Hz, "
                       f"channels: {audio_stream.channels}, "
                       f"codec: {input_audio_codec} -> {output_audio_codec}")
            
            output_audio_stream = output_container.add_stream(output_audio_codec, rate=audio_stream.rate)
            output_audio_stream.channels = audio_stream.channels
            output_audio_stream.layout = audio_stream.layout
            
            # Copy audio encoding parameters
            if hasattr(audio_stream.codec_context, 'bit_rate') and audio_stream.codec_context.bit_rate:
                output_audio_stream.bit_rate = audio_stream.codec_context.bit_rate
            else:
                output_audio_stream.bit_rate = 128000  # fallback audio bitrate
        
        processed_frames = []
        audio_frames = []
        frame_count = 0
        
        # Process video frames through pipeline and count them
        logger.info("Extracting and processing video frames...")
        for packet in input_container.demux(video_stream):
            for frame in packet.decode():
                frame_count += 1
                # Put frame in pipeline for processing
                await pipeline.put_video_frame(frame)
        
        # Process audio frames if present
        if audio_stream:
            logger.info("Extracting audio frames...")
            for packet in input_container.demux(audio_stream):
                for frame in packet.decode():
                    audio_frames.append(frame)
                    # Optionally process audio through pipeline
                    # await pipeline.put_audio_frame(frame)
        
        input_container.close()
        
        # Collect processed video frames from pipeline
        logger.info(f"Collecting {frame_count} processed frames from pipeline...")
        
        # Add timeout protection for pipeline processing
        timeout_seconds = 30  # 30 second timeout
        
        for i in range(frame_count):
            try:
                # Use asyncio.wait_for to add timeout protection
                processed_frame = await asyncio.wait_for(
                    pipeline.get_processed_video_frame(),
                    timeout=timeout_seconds
                )
                processed_frames.append(processed_frame)
                
                if (i + 1) % 10 == 0:  # Log every 10 frames
                    logger.info(f"Processed {i + 1}/{frame_count} frames")
                    
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for processed frame {i}")
                break
            except Exception as e:
                logger.error(f"Error getting processed frame {i}: {e}")
                break
        
        logger.info(f"Collected {len(processed_frames)} processed frames")
        
        # If we didn't get any processed frames, return an error
        if not processed_frames:
            return web.Response(
                status=500, 
                text="No frames were successfully processed"
            )
        
        # Encode processed frames to output container
        logger.info("Encoding processed frames to output...")
        for i, frame in enumerate(processed_frames):
            try:
                for packet in output_video_stream.encode(frame):
                    output_container.mux(packet)
            except Exception as e:
                logger.error(f"Error encoding frame {i}: {e}")
                # Continue with remaining frames
        
        # Flush video encoder
        for packet in output_video_stream.encode():
            output_container.mux(packet)
        
        # Encode audio frames if present
        if output_audio_stream and audio_frames:
            for frame in audio_frames:
                for packet in output_audio_stream.encode(frame):
                    output_container.mux(packet)
            
            # Flush audio encoder
            for packet in output_audio_stream.encode():
                output_container.mux(packet)
        
        # Finalize output
        output_container.close()
        
        # Get the processed segment data
        output_data = output_buffer.getvalue()
        output_buffer.close()
        
        processing_time = time.time() - start_time
        fps = len(processed_frames) / processing_time if processing_time > 0 else 0
        
        logger.info(f"Processed segment {segment_index}: "
                   f"input {len(segment_data)} bytes -> output {len(output_data)} bytes, "
                   f"processed {len(processed_frames)}/{frame_count} frames, "
                   f"processing time: {processing_time:.2f}s ({fps:.1f} fps)")
        
        # Determine Content-Type based on output format
        content_type_map = {
            'webm': 'video/webm',
            'mp4': 'video/mp4', 
            'mkv': 'video/x-matroska',
            'avi': 'video/x-msvideo'
        }
        content_type = content_type_map.get(output_format, 'video/webm')
        
        # Get codec names for headers (handle potential undefined variables)
        video_codec_name = output_video_codec if 'output_video_codec' in locals() else 'unknown'
        audio_codec_name = output_audio_codec if 'output_audio_codec' in locals() and output_audio_stream else 'none'
        
        # Return the processed segment
        return web.Response(
            body=output_data,
            headers={
                'Content-Type': content_type,
                'Content-Length': str(len(output_data)),
                'X-Segment-Index': str(segment_index),
                'X-Timestamp': str(timestamp),
                'X-Processed-Frames': str(len(processed_frames)),
                'X-Total-Frames': str(frame_count),
                'X-Processing-Time': f"{processing_time:.2f}",
                'X-Processing-FPS': f"{fps:.1f}",
                'X-Output-Format': output_format,
                'X-Video-Codec': video_codec_name,
                'X-Audio-Codec': audio_codec_name
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing segment: {e}", exc_info=True)
        return web.Response(status=500, text=f"Error processing segment: {str(e)}")

async def segment(request):
    """Serve a single video segment (legacy endpoint)"""
    return web.Response(status=501, text="Use /api/segments for segment processing")

async def stream_mjpeg(request):
    """Serve an MJPEG stream with token validation"""
    # Clean up expired sessions
    cleanup_expired_sessions()
    
    stream_id = request.query.get("token")
    
    # Validate the stream token
    is_valid, error_message = validate_token(stream_id)
    if not is_valid:
        return web.Response(status=403, text=error_message)
    
    frame_buffer = FrameBuffer.get_instance()
    
    # Use a fixed frame delay for 30 FPS
    frame_delay = 1.0 / 30
    
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache',
            'Connection': 'close',
        }
    )
    await response.prepare(request)
    
    try:
        while True:
            jpeg_frame = frame_buffer.get_current_frame()
            if jpeg_frame is not None:
                await response.write(
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n'
                )
            await asyncio.sleep(frame_delay)
    except (ConnectionResetError, asyncio.CancelledError):
        logger.info("MJPEG stream connection closed")
    except Exception as e:
        logger.error(f"Error in MJPEG stream: {e}")
    finally:
        return response

def setup_routes(app, cors):
    """Setup HTTP streaming routes
    
    Args:
        app: The aiohttp web application
        cors: The CORS setup object
    """
    # Stream token endpoints
    cors.add(app.router.add_post("/api/stream-token", create_stream_token))
    
    # Stream endpoint with token validation
    cors.add(app.router.add_get("/api/stream", stream_mjpeg))
    
    # Segment processing endpoint
    cors.add(app.router.add_post("/api/segments", process_segment))
