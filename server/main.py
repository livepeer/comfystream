#!/usr/bin/env python3
import asyncio
import json
import logging
import sys
import os
# import av
# import numpy as np
import torch
# Initialize CUDA before any other imports to prevent core dump.
if torch.cuda.is_available():
    torch.cuda.init()


from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer
from comfystream.utils import DEFAULT_PROMPT, DEFAULT_SD_PROMPT
from comfystream.pipeline import Pipeline
from comfystream.trickle import TrickleServer

DEFAULT_CURRENT_PROMPT = DEFAULT_SD_PROMPT

ORCH_URL = os.getenv("ORCHESTRATOR_URL", "orchestrator:9995")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ComfyStreamVideoTrack(MediaStreamTrack):
    """Video track that processes video using ComfyStream pipeline"""
    
    def __init__(self, source_track, pipeline):
        super().__init__()
        self.source_track = source_track
        self.pipeline = pipeline
        self.kind = "video"
        self._started = False
        
    async def recv(self):
        if not self._started:
            self._started = True
            logger.info("ComfyStreamVideoTrack started receiving frames")
            
        try:
            frame = await self.source_track.recv()
            
            # Put frame into pipeline for processing
            await self.pipeline.put_video_frame(frame)
            
            # Get processed frame from pipeline
            processed_frame = await self.pipeline.get_processed_video_frame()
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing video frame: {e}")
            raise

class ComfyStreamAudioTrack(MediaStreamTrack):
    """Audio track that processes audio using ComfyStream pipeline"""
    
    def __init__(self, source_track, pipeline):
        super().__init__()
        self.source_track = source_track
        self.pipeline = pipeline
        self.kind = "audio"
        self._started = False
    
    async def recv(self):
        if not self._started:
            self._started = True
            logger.info("ComfyStreamAudioTrack started receiving frames")
            
        try:
            frame = await self.source_track.recv()
            
            # Put frame into pipeline for processing
#            await self.pipeline.put_audio_frame(frame)
            
            # Get processed frame from pipeline
 #           processed_frame = await self.pipeline.get_processed_audio_frame()
            
            return frame
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            # Return original frame on error
            return await self.source_track.recv()

class WebRTCServer:
    def __init__(self):
        self.whep_pc = None  # Peer connection for receiving (WHEP)
        self.whip_pc = None  # Peer connection for sending (WHIP)
        self.caller_ip = None
        self.stream_id = ""
        self.video_track = None
        self.audio_track = None
        self.transformed_video_track = None
        self.transformed_audio_track = None
        self.processing_active = False
        self.track_event = asyncio.Event()  # Event to signal when tracks are ready
        self.pipeline = None
    
    async def init_pipeline(self):
        """Initialize the ComfyStream pipeline"""
        
        try:
            logger.info("Initializing ComfyStream pipeline...")
            self.pipeline = Pipeline(
                width=512,
                height=512,
                comfyui_inference_log_level="DEBUG",
                cwd="/workspace/ComfyUI",
                disable_cuda_malloc=True, 
                gpu_only=True,
                preview_method="none"
            )
            
            logging.info(f"Pipeline initialized with width: {self.pipeline.width}, height: {self.pipeline.height}")
            
            # Set default prompts for the pipeline
            await self.pipeline.set_prompts(json.loads(DEFAULT_CURRENT_PROMPT))
            
            logger.info("ComfyStream pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    async def wait_for_ice(self, pc, timeout=10):
        async def ice_complete():
            while pc.iceGatheringState != "complete":
                await asyncio.sleep(0.1)
        try:
            await asyncio.wait_for(ice_complete(), timeout)
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out")
            
    async def start_processing(self, caller_ip: str, stream_id: str):
        """Start the WebRTC processing pipeline"""
        try:
            self.caller_ip = caller_ip
            self.stream_id = stream_id
            logger.info(f"Start processing for {self.caller_ip} stream {self.stream_id}")
            
            # Initialize ComfyStream pipeline if not already initialized
            if not self.pipeline:
                await self.init_pipeline()
                default_prompt_parsed = json.loads(DEFAULT_CURRENT_PROMPT)
                await self.pipeline.set_prompts(default_prompt_parsed)
                
                # Warm up both video and audio pipelines
                await self.pipeline.warm_video()
                # await self.pipeline.warm_audio()
            
            # Initialize WHEP connection to get source frames
            await self.init_whep_connection(stream_id)
            
            # Wait for tracks to be received
            await self.wait_for_tracks()
            
            # Create transformed tracks using ComfyStream
            await self.create_transformed_tracks()
            
            # Initialize WHIP connection to send transformed frames
            await self.init_whip_connection(stream_id)
            
            self.processing_active = True
            logger.info("WebRTC processing pipeline established successfully")
            
            # Start frame forwarding task
            asyncio.create_task(self.forward_frames())
            
            return {"status": "started", "caller_ip": self.caller_ip, "protocol": "webrtc"}
        
        except Exception as e:
            logger.error(f"Error in start processing: {e}")
            await self.cleanup()
            raise HTTPException(status_code=500, detail=str(e))
    
    async def forward_frames(self):
        """Continuously forward frames from source to transformed tracks"""
        logger.info("Starting frame forwarding task")
        try:
            while self.processing_active:
                # The transformed tracks will automatically receive frames when they're available
                # because they're connected to the source tracks via their recv() methods
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
                # Check if connections are still active
                if (self.whep_pc and self.whep_pc.connectionState != "connected") or \
                   (self.whip_pc and self.whip_pc.connectionState != "connected"):
                    logger.warning("One or more connections lost, stopping forwarding")
                    break
                    
        except Exception as e:
            logger.error(f"Error in frame forwarding: {e}")
        finally:
            logger.info("Frame forwarding task ended")
    
    async def init_whep_connection(self, stream_id: str):
        """Initialize WHEP connection to receive source frames"""
        logger.info("Initializing WHEP connection...")
        self.whep_pc = RTCPeerConnection()
        
        # Add receive-only transceivers for audio and video
        audio_transceiver = self.whep_pc.addTransceiver(
            "audio", 
            direction="recvonly"
        )
        logger.info(f"Added audio transceiver: {audio_transceiver.mid}, direction: {audio_transceiver.direction}")
        
        video_transceiver = self.whep_pc.addTransceiver(
            "video", 
            direction="recvonly"
        )
        logger.info(f"Added video transceiver: {video_transceiver.mid}, direction: {video_transceiver.direction}")
        
        # Track handler for incoming media
        @self.whep_pc.on("track")
        def on_track(track):
            logger.info(f"Received track: {track.kind} - ID: {track.id}")
            if track.kind == "video":
                self.video_track = track
                logger.info("Video track received and stored")
            elif track.kind == "audio":
                self.audio_track = track
                logger.info("Audio track received and stored")
            
            # Signal that we have tracks if we have both
            if self.video_track and self.audio_track:
                self.track_event.set()
        
        # Connection state handlers
        @self.whep_pc.on("connectionstatechange")
        def on_whep_connectionstatechange():
            state = self.whep_pc.connectionState
            logger.info(f"WHEP connection state: {state}")
            if state == "failed" or state == "disconnected" or state == "closed":
                self.processing_active = False
        
        @self.whep_pc.on("iceconnectionstatechange")
        def on_whep_iceconnectionstatechange():
            state = self.whep_pc.iceConnectionState
            logger.info(f"WHEP ICE connection state: {state}")
            if state == "failed" or state == "disconnected" or state == "closed":
                self.processing_active = False
        
        # Create offer for WHEP
        offer = await self.whep_pc.createOffer()
        await self.whep_pc.setLocalDescription(offer)
        logger.info("WHEP offer created and set as local description")
        await self.wait_for_ice(self.whep_pc)
        
        # Send offer to WHEP endpoint
        whep_url = f"https://{ORCH_URL}/process/worker/whep"
        logger.info(f"Sending WHEP request to: {whep_url}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.post(
                    whep_url,
                    content=self.whep_pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp", "Accept": "application/sdp", "X-Stream-Id": stream_id}
                )
                
                if response.status_code == 201:
                    answer_sdp = response.text
                    await self.whep_pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer_sdp, type="answer")
                    )
                    logger.info("WHEP connection established successfully")
                else:
                    logger.error(f"WHEP request failed: {response.status_code} - {response.text}")
                    raise Exception(f"WHEP connection failed with status {response.status_code}")
                    
        except httpx.RequestError as e:
            logger.error(f"WHEP request error: {e}")
            raise Exception(f"WHEP connection failed: {e}")
    
    async def wait_for_tracks(self):
        """Wait for both audio and video tracks to be received"""
        logger.info("Waiting for media tracks...")
        
        # Wait for either tracks to be received or timeout
        try:
            await asyncio.wait_for(self.track_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            raise Exception(f"Timeout waiting for media tracks. Video: {self.video_track is not None}, Audio: {self.audio_track is not None}")
        
        if not self.video_track or not self.audio_track:
            raise Exception(f"Didn't receive both tracks. Video: {self.video_track is not None}, Audio: {self.audio_track is not None}")
        
        logger.info("All media tracks received successfully")
    
    async def create_transformed_tracks(self):
        """Create transformed versions of the received tracks using ComfyStream"""
        logger.info("Creating ComfyStream transformed tracks...")
        
        if self.video_track and self.pipeline:
            self.transformed_video_track = ComfyStreamVideoTrack(self.video_track, self.pipeline)
            logger.info("ComfyStream video track created")
        
        if self.audio_track and self.pipeline:
            self.transformed_audio_track = ComfyStreamAudioTrack(self.audio_track, self.pipeline)
            logger.info("ComfyStream audio track created")
    
    async def init_whip_connection(self, stream_id: str):
        """Initialize WHIP connection to send transformed frames"""
        logger.info("Initializing WHIP connection...")
        self.whip_pc = RTCPeerConnection()
        
        # Connection state handlers
        @self.whip_pc.on("connectionstatechange")
        def on_whip_connectionstatechange():
            state = self.whip_pc.connectionState
            logger.info(f"WHIP connection state: {state}")
            if state == "failed" or state == "disconnected" or state == "closed":
                self.processing_active = False
            
        @self.whip_pc.on("iceconnectionstatechange")
        def on_whip_iceconnectionstatechange():
            state = self.whip_pc.iceConnectionState
            logger.info(f"WHIP ICE connection state: {state}")
            if state == "failed" or state == "disconnected" or state == "closed":
                self.processing_active = False
        
        # Add transformed tracks
        if self.transformed_video_track:
            self.whip_pc.addTrack(self.transformed_video_track)
            logger.info("Transformed video track added to WHIP connection")
            
        if self.transformed_audio_track:
            self.whip_pc.addTrack(self.transformed_audio_track)
            logger.info("Transformed audio track added to WHIP connection")
        
        # Create offer for WHIP
        offer = await self.whip_pc.createOffer()
        await self.whip_pc.setLocalDescription(offer)
        logger.info("WHIP offer created and set as local description")
        await self.wait_for_ice(self.whip_pc)
        
        # Send offer to WHIP endpoint
        whip_url = f"https://{ORCH_URL}/process/worker/whip"
        logger.info(f"Sending WHIP request to: {whip_url}")
        
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.post(
                    whip_url,
                    content=self.whip_pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp", "Accept": "application/sdp", "X-Stream-Id": stream_id}
                )
                
                if response.status_code in [200, 201]:
                    answer_sdp = response.text
                    await self.whip_pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer_sdp, type="answer")
                    )
                    logger.info("WHIP connection established successfully")
                else:
                    logger.error(f"WHIP request failed: {response.status_code} - {response.text}")
                    raise Exception(f"WHIP connection failed with status {response.status_code}")
                    
        except httpx.RequestError as e:
            logger.error(f"WHIP request error: {e}")
            raise Exception(f"WHIP connection failed: {e}")
    
    async def cleanup(self):
        """Clean up all connections and resources"""
        logger.info("Cleaning up WebRTC connections...")
        self.processing_active = False
        self.track_event.clear()
        
        try:
            if self.whep_pc:
                await self.whep_pc.close()
                logger.info("WHEP connection closed")
        except Exception as e:
            logger.error(f"Error closing WHEP connection: {e}")
        
        try:
            if self.whip_pc:
                await self.whip_pc.close()
                logger.info("WHIP connection closed")
        except Exception as e:
            logger.error(f"Error closing WHIP connection: {e}")
        
        try:
            if self.pipeline:
                await self.pipeline.cleanup()
                logger.info("ComfyStream pipeline cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up pipeline: {e}")
        
        # Reset state
        self.whep_pc = None
        self.whip_pc = None
        self.video_track = None
        self.audio_track = None
        self.transformed_video_track = None
        self.transformed_audio_track = None
        self.caller_ip = None
        self.pipeline = None
    
    def get_status(self):
        """Get current processing status"""
        return {
            "processing_active": self.processing_active,
            "caller_ip": self.caller_ip,
            "protocol": "webrtc",
            "whep_connected": self.whep_pc is not None and self.whep_pc.connectionState == "connected",
            "whip_connected": self.whip_pc is not None and self.whip_pc.connectionState == "connected",
            "video_track_available": self.video_track is not None,
            "audio_track_available": self.audio_track is not None,
            "transformed_tracks_created": self.transformed_video_track is not None and self.transformed_audio_track is not None,
            "pipeline_initialized": self.pipeline is not None
        }

class StreamingServer:
    """Unified streaming server supporting both WebRTC and Trickle protocols"""
    
    def __init__(self):
        self.webrtc_server = WebRTCServer()
        self.trickle_server = None
        self.pipeline = None
        self.current_protocol = None
        self.active_server = None
        
    async def init_pipeline(self):
        """Initialize shared pipeline"""
        if not self.pipeline:
            logger.info("Initializing shared ComfyStream pipeline...")
            self.pipeline = Pipeline(
                width=512,
                height=512,
                comfyui_inference_log_level="DEBUG",
                cwd="/workspace/ComfyUI",
                disable_cuda_malloc=True, 
                gpu_only=True,
                preview_method="none"
            )
            
            # Set default prompts
            await self.pipeline.set_prompts(json.loads(DEFAULT_CURRENT_PROMPT))
            logger.info("Shared ComfyStream pipeline initialized")
        
        return self.pipeline
    
    async def start_processing(self, caller_ip: str, stream_id: str, protocol: str = "webrtc", input_url: str = None):
        """Start processing with specified protocol"""
        try:
            # Initialize pipeline if not already done
            pipeline = await self.init_pipeline()
            
            if protocol.lower() == "webrtc":
                self.webrtc_server.pipeline = pipeline
                self.current_protocol = "webrtc"
                self.active_server = self.webrtc_server
                result = await self.webrtc_server.start_processing(caller_ip, stream_id)
                
            elif protocol.lower() == "trickle":
                self.trickle_server = TrickleServer(pipeline)
                self.current_protocol = "trickle"
                self.active_server = self.trickle_server
                result = await self.trickle_server.start_processing(caller_ip, stream_id, input_url)
                
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
            
            logger.info(f"Started processing with protocol: {protocol}")
            return result
            
        except Exception as e:
            logger.error(f"Error starting processing: {e}")
            await self.cleanup()
            raise
    
    async def cleanup(self):
        """Clean up all resources"""
        if self.active_server:
            await self.active_server.cleanup()
        
        if self.pipeline:
            await self.pipeline.cleanup()
        
        self.webrtc_server = WebRTCServer()
        self.trickle_server = None
        self.pipeline = None
        self.current_protocol = None
        self.active_server = None
    
    def get_status(self):
        """Get current server status"""
        if self.active_server:
            status = self.active_server.get_status()
            status["current_protocol"] = self.current_protocol
            return status
        
        return {
            "processing_active": False,
            "current_protocol": None,
            "pipeline_initialized": self.pipeline is not None
        }

# Global server instance
streaming_server = StreamingServer()

# Initialize pipeline at startup
async def startup_event():
    """Initialize and warm up the ComfyStream pipeline on startup"""
    try:
        logger.info("Initializing ComfyStream pipeline on startup...")
        await streaming_server.init_pipeline()
        
        # Warm up video pipeline
        logger.info("Warming up video pipeline...")
        await streaming_server.pipeline.warm_video()
        
        logger.info("ComfyStream pipeline warmed up successfully on startup")
        
    except Exception as e:
        logger.error(f"Error warming up pipeline on startup: {e}")
        # Don't raise the exception to allow the application to start
        # The pipeline will be re-initialized when processing starts

# Cleanup on shutdown
async def shutdown_event():
    """Cleanup resources on shutdown"""
    await streaming_server.cleanup()
    logger.info("Application shutdown complete")

# Initialize FastAPI app with lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield
    await shutdown_event()

app = FastAPI(
    title="ComfyStream Media Processor", 
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/stream/start")
async def start_endpoint(request: Request):
    """Handle /start endpoint to initiate streaming processing"""
    try:
        # Get caller IP from request
        caller_ip = request.client.host
        if request.headers.get("x-forwarded-for"):
            caller_ip = request.headers.get("x-forwarded-for").split(",")[0].strip()
        
        # Check for protocol preference in headers
        protocol = request.headers.get("x-protocol", "webrtc").lower()
        
        logger.info(f"Start request from {caller_ip} using protocol: {protocol}")
        
        # Parse request body based on protocol
        if protocol == "trickle":
            # For trickle, expect JSON with stream_id and input_url
            body = await request.json()
            stream_id = body.get("stream_id")
            input_url = body.get("input_url")
            
            if not stream_id:
                raise HTTPException(status_code=400, detail="Missing stream_id in request body")
            if not input_url:
                raise HTTPException(status_code=400, detail="Missing input_url in request body for trickle protocol")
                
            # Start processing with input URL
            result = await streaming_server.start_processing(caller_ip, stream_id, protocol, input_url=input_url)
        else:
            # For WebRTC, expect plain text stream_id (backward compatibility)
            body_bytes = await request.body()
            stream_id = body_bytes.decode()
            result = await streaming_server.start_processing(caller_ip, stream_id, protocol)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in start endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream/stop")
async def stop_endpoint():
    """Handle /stop endpoint to stop streaming processing"""
    try:
        await streaming_server.cleanup()
        return {"status": "stopped"}
    
    except Exception as e:
        logger.error(f"Error in stop endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream/status")
async def status_endpoint():
    """Get current processing status"""
    return streaming_server.get_status()

# Trickle protocol endpoints
@app.get("/trickle/subscribe")
async def trickle_subscribe(request: Request):
    """Trickle subscribe endpoint"""
    try:
        stream_id = request.headers.get("x-stream-id")
        if not stream_id:
            raise HTTPException(status_code=400, detail="Missing X-Stream-Id header")
        
        if not streaming_server.trickle_server:
            raise HTTPException(status_code=400, detail="Trickle server not initialized")
        
        frame_generator = await streaming_server.trickle_server.handle_subscribe(stream_id)
        
        return StreamingResponse(
            frame_generator,
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        logger.error(f"Trickle subscribe error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Direct streaming endpoints for ffplay compatibility
@app.get("/ai/trickle/{stream_name}")
async def direct_stream_input(stream_name: str, request: Request):
    """Direct streaming endpoint for input streams (ffplay compatible)"""
    try:
        if not streaming_server.trickle_server:
            raise HTTPException(status_code=400, detail="Trickle server not initialized")
        
        # Check if this is the correct input stream
        if streaming_server.trickle_server.stream_id != stream_name:
            raise HTTPException(status_code=404, detail=f"Stream {stream_name} not found")
        
        # For input streams, we could proxy from the input source or return an error
        # since the input should be accessed directly from the source
        raise HTTPException(status_code=404, detail="Input stream should be accessed from source directly")
        
    except Exception as e:
        logger.error(f"Direct input stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/trickle/{stream_name}-out")
async def direct_stream_output(stream_name: str, request: Request):
    """Direct streaming endpoint for processed output streams (ffplay compatible)"""
    try:
        if not streaming_server.trickle_server:
            raise HTTPException(status_code=400, detail="Trickle server not initialized")
        
        # Check if this is the correct output stream
        if streaming_server.trickle_server.stream_id != stream_name:
            raise HTTPException(status_code=404, detail=f"Stream {stream_name} not found. Active stream: {streaming_server.trickle_server.stream_id}")
        
        # Create a raw media stream generator
        async def raw_media_generator():
            try:
                # Subscribe to the trickle server's output
                frame_generator = await streaming_server.trickle_server.handle_subscribe(stream_name)
                
                async for frame_data in frame_generator:
                    try:
                        # Parse the NDJSON frame data
                        import json
                        frame_json = json.loads(frame_data.decode())
                        
                        if frame_json.get("type") == "frame":
                            # Extract raw frame data and yield it directly
                            import base64
                            raw_data = base64.b64decode(frame_json["data"])
                            yield raw_data
                            
                    except Exception as e:
                        logger.error(f"Error processing frame for direct stream: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error in raw media generator: {e}")
        
        # Return streaming response with appropriate media type
        return StreamingResponse(
            raw_media_generator(),
            media_type="video/mp4",  # or "application/octet-stream"
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        logger.error(f"Direct output stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trickle/publish")
async def trickle_publish(request: Request):
    """Trickle publish endpoint"""
    try:
        stream_id = request.headers.get("x-stream-id")
        if not stream_id:
            raise HTTPException(status_code=400, detail="Missing X-Stream-Id header")
        
        if not streaming_server.trickle_server:
            raise HTTPException(status_code=400, detail="Trickle server not initialized")
        
        frame_data = await request.json()
        result = await streaming_server.trickle_server.handle_publish(stream_id, frame_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Trickle publish error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trickle/control")
async def trickle_control(request: Request):
    """Trickle control endpoint"""
    try:
        stream_id = request.headers.get("x-stream-id")
        if not stream_id:
            raise HTTPException(status_code=400, detail="Missing X-Stream-Id header")
        
        if not streaming_server.trickle_server:
            raise HTTPException(status_code=400, detail="Trickle server not initialized")
        
        control_data = await request.json()
        result = await streaming_server.trickle_server.handle_control(stream_id, control_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Trickle control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "comfystream-media-processor"}

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "ComfyStream Media Processor",
        "version": "1.0.0",
        "supported_protocols": ["webrtc", "trickle"],
        "endpoints": {
            "start": "POST /stream/start - Start media processing (X-Protocol: trickle requires JSON with stream_id and optional input_url)",
            "stop": "POST /stream/stop - Stop media processing", 
            "status": "GET /stream/status - Get processing status",
            "trickle_subscribe": "GET /trickle/subscribe - Subscribe to trickle stream",
            "trickle_publish": "POST /trickle/publish - Publish to trickle stream",
            "trickle_control": "POST /trickle/control - Send trickle control messages",
            "direct_stream_input": "GET /ai/trickle/{stream_name} - Direct input stream (ffplay compatible)",
            "direct_stream_output": "GET /ai/trickle/{stream_name}-out - Direct processed output stream (ffplay compatible)",
            "health": "GET /health - Health check"
        },
        "examples": {
            "start_trickle": "curl -X POST http://localhost:8889/stream/start -H 'X-Protocol: trickle' -d '{\"stream_id\":\"test\",\"input_url\":\"http://source:3389/test\"}'",
            "watch_output": "ffplay http://localhost:8889/ai/trickle/test-out",
            "watch_input": "ffplay http://source:3389/test"
        }
    }

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description='ComfyStream Media Processor')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=9876, help='Port to listen on')
    parser.add_argument('--protocol', type=str, default='webrtc', choices=['webrtc', 'trickle'], 
                       help='Default protocol to use')
    args = parser.parse_args()

    logger.info(f"Starting ComfyStream Media Processor with default protocol: {args.protocol}")
    uvicorn.run(app, host=args.host, port=args.port)