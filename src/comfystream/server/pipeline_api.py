import av
import torch
import numpy as np
import asyncio
import logging
import time
import random
from collections import OrderedDict
import collections
import os
import socket

from typing import Any, Dict, Union, List, Optional, Deque
from comfystream.client_api import ComfyStreamClient
from utils import temporary_log_level # Not sure exactly what this does
from config import ComfyConfig

WARMUP_RUNS = 5
logger = logging.getLogger(__name__)


class MultiServerPipeline:
    def __init__(
            self, 
            width: int = 512, 
            height: int = 512,
            workers: int = 2, 
            comfyui_inference_log_level: int = None, 
            config_path: Optional[str] = None, 
            max_frame_wait_ms: int = 500, 
            client_mode: str = "toml", 
            workspace: str = None
        ):
        """Initialize the pipeline with the given configuration.
        Args:
            width: The width of the video frames.
            height: The height of the video frames.
            workers: The number of ComfyUI clients to spin up (if client_mode is "spawn").
            comfyui_inference_log_level: The logging level for ComfyUI inference.
                Defaults to None, using the global ComfyUI log level.
            config_path: The path to the ComfyUI config toml file (if client_mode is "toml").
            max_frame_wait_ms: The maximum number of milliseconds to wait for a frame before dropping it.
            client_mode: The mode to use for the ComfyUI clients.
                "toml": Use a config file to describe clients.
                "spawn": Spawn ComfyUI clients as external processes.
        """

        # There are two methods for starting the clients:
        # 1. client_mode == "toml" -> Use a config file to describe clients.
        # 2. client_mode == "spawn" -> Spawn ComfyUI clients as external processes.

        self.clients = []
        self.workspace = workspace
        self.client_mode = client_mode

        if (client_mode == "toml"):
            # Load server configurations
            self.config = ComfyConfig(config_path)
            self.servers = self.config.get_servers()
        elif (client_mode == "spawn"):
            # Set the number of workers to spawn
            self.workers = workers
        
        # Started in /offer
        # self.start_clients()
        
        self.width = width
        self.height = height
        
        self.video_incoming_frames = asyncio.Queue()
        self.audio_incoming_frames = asyncio.Queue()
        
        # Queue for processed frames from all clients
        self.processed_video_frames = asyncio.Queue()
        
        # Track which client gets each frame (round-robin)
        self.current_client_index = 0
        self.client_frame_mapping = {}  # Maps frame_id -> client_index
        
        # Frame ordering and timing
        self.max_frame_wait_ms = max_frame_wait_ms  # Max time to wait for a frame before dropping
        self.next_expected_frame_id = None  # Track expected frame ID
        self.ordered_frames = OrderedDict()  # Buffer for ordering frames (frame_id -> (timestamp, tensor))
        
        # Audio processing
        self.processed_audio_buffer = np.array([], dtype=np.int16)
        self.last_frame_time = 0

        # ComfyUI inference log level
        self._comfyui_inference_log_level = comfyui_inference_log_level

        # Frame rate limiting
        self.min_frame_interval = 1/30  # Limit to 30 FPS
        
        # Create background task for collecting processed frames
        self.running = True
        self.collector_task = asyncio.create_task(self._collect_processed_frames())

        self.output_interval = 1/30  # Start with 30 FPS
        self.last_output_time = None
        self.frame_interval_history = collections.deque(maxlen=30)
        self.output_pacer_task = asyncio.create_task(self._dynamic_output_pacer())
    
    async def _collect_processed_frames(self):
        """Background task to collect processed frames from all clients"""
        try:
            while self.running:
                for i, client in enumerate(self.clients):
                    try:
                        # Non-blocking check if client has output ready
                        if hasattr(client, '_prompt_id') and client._prompt_id is not None:
                            # Get frame without waiting
                            try:
                                # Use wait_for with small timeout to avoid blocking
                                result = await asyncio.wait_for(
                                    client.get_video_output(), 
                                    timeout=0.01
                                )
                                
                                # Check if result is already a tuple with frame_id
                                if isinstance(result, tuple) and len(result) == 2:
                                    frame_id, out_tensor = result
                                    logger.debug(f"Got result with embedded frame_id: {frame_id}")
                                else:
                                    out_tensor = result
                                    # Find which original frame this corresponds to using our mapping
                                    frame_ids = [frame_id for frame_id, client_idx in 
                                              self.client_frame_mapping.items() if client_idx == i]
                                    
                                    if frame_ids:
                                        # Use the oldest frame ID for this client
                                        frame_id = min(frame_ids)
                                    else:
                                        # If no mapping found, log warning and continue
                                        logger.warning(f"No frame_id mapping found for tensor from client {i}")
                                        continue
                                
                                # Store frame with timestamp for ordering
                                timestamp = time.time()
                                await self._add_frame_to_ordered_buffer(frame_id, timestamp, out_tensor)
                                
                                # Remove the mapping
                                self.client_frame_mapping.pop(frame_id, None)
                                logger.info(f"Collected processed frame from client {i}, frame_id: {frame_id}")
                            except asyncio.TimeoutError:
                                # No frame ready yet, continue
                                pass
                    except Exception as e:
                        logger.error(f"Error collecting frame from client {i}: {e}")
                
                # Check for frames that have waited too long
                await self._check_frame_timeouts()
                
                # Small sleep to avoid CPU spinning
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("Frame collector task cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in frame collector: {e}")

    async def _add_frame_to_ordered_buffer(self, frame_id, timestamp, tensor):
        """Add a processed frame to the ordered buffer"""
        self.ordered_frames[frame_id] = (timestamp, tensor)
        
        # If this is the first frame, set the next expected frame ID
        if self.next_expected_frame_id is None:
            self.next_expected_frame_id = frame_id
            
        # Check if we can release any frames now
        await self._release_ordered_frames()

    async def _release_ordered_frames(self):
        if self.next_expected_frame_id is None:
            return
        if self.ordered_frames and self.next_expected_frame_id in self.ordered_frames:
            timestamp, tensor = self.ordered_frames.pop(self.next_expected_frame_id)
            await self.processed_video_frames.put((self.next_expected_frame_id, tensor))
            logger.info(f"Released frame {self.next_expected_frame_id} to output queue")
            if self.ordered_frames:
                self.next_expected_frame_id = min(self.ordered_frames.keys())
            else:
                self.next_expected_frame_id += 1

    async def _check_frame_timeouts(self):
        """Check for frames that have waited too long and handle them"""
        if not self.ordered_frames or self.next_expected_frame_id is None:
            return
            
        current_time = time.time()
        
        # If the next expected frame has timed out, skip it and move on
        if self.next_expected_frame_id in self.ordered_frames:
            timestamp, _ = self.ordered_frames[self.next_expected_frame_id]
            wait_time_ms = (current_time - timestamp) * 1000
            
            if wait_time_ms > self.max_frame_wait_ms:
                logger.warning(f"Frame {self.next_expected_frame_id} exceeded max wait time, releasing anyway")
                await self._release_ordered_frames()
                
        # Check if we're missing the next expected frame and it's been too long
        elif self.ordered_frames:
            # The next frame we're expecting isn't in the buffer
            # Check how long we've been waiting since the oldest frame in the buffer
            oldest_frame_id = min(self.ordered_frames.keys())
            oldest_timestamp, _ = self.ordered_frames[oldest_frame_id]
            wait_time_ms = (current_time - oldest_timestamp) * 1000
            
            # If we've waited too long, skip the missing frame(s)
            if wait_time_ms > self.max_frame_wait_ms:
                logger.debug(f"Missing frame {self.next_expected_frame_id}, skipping to {oldest_frame_id}")
                self.next_expected_frame_id = oldest_frame_id
                await self._release_ordered_frames()

    async def warm_video(self):
        # Create dummy frame with the CURRENT resolution settings (which might have been updated via control channel)
        
        # Create a properly formatted dummy frame
        '''
        tensor = torch.rand(1, 3, 512, 512)  # Random values in [0,1]
        dummy_frame = av.VideoFrame(width=512, height=512, format="rgb24")
        dummy_frame.side_data.input = tensor
        '''
        dummy_frame = av.VideoFrame()
        dummy_frame.side_data.input = torch.randn(1, self.height, self.width, 3)

        logger.info(f"Warming video pipeline with resolution {self.width}x{self.height}")

        # Warm up each client
        warmup_tasks = []
        for i, client in enumerate(self.clients):
            warmup_tasks.append(self._warm_client_video(client, i, dummy_frame))
            
        # Wait for all warmup tasks to complete
        await asyncio.gather(*warmup_tasks)
        logger.info("Video pipeline warmup complete")
    
    async def _warm_client_video(self, client, client_index, dummy_frame):
        """Warm up a single client"""
        logger.info(f"Warming up client {client_index}")
        for i in range(WARMUP_RUNS):
            logger.info(f"Client {client_index} warmup iteration {i+1}/{WARMUP_RUNS}")
            client.put_video_input(dummy_frame)
            try:
                await asyncio.wait_for(client.get_video_output(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for warmup frame from client {client_index}")
            except Exception as e:
                logger.error(f"Error warming client {client_index}: {e}")

    async def warm_audio(self):
        # For now, only use the first client for audio
        if not self.clients:
            logger.warning("No clients available for audio warmup")
            return
            
        dummy_frame = av.AudioFrame()
        dummy_frame.side_data.input = np.random.randint(-32768, 32767, int(48000 * 0.5), dtype=np.int16)
        dummy_frame.sample_rate = 48000

        for _ in range(WARMUP_RUNS):
            self.clients[0].put_audio_input(dummy_frame)
            await self.clients[0].get_audio_output()

    async def set_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        """Set the same prompts for all clients"""
        if isinstance(prompts, dict):
            prompts = [prompts]
            
        # Set prompts for each client
        tasks = []
        for client in self.clients:
            tasks.append(client.set_prompts(prompts))
            
        await asyncio.gather(*tasks)
        logger.info(f"Set prompts for {len(self.clients)} clients")

    async def update_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        """Update prompts for all clients"""
        if isinstance(prompts, dict):
            prompts = [prompts]
            
        # Update prompts for each client
        tasks = []
        for client in self.clients:
            tasks.append(client.update_prompts(prompts))
            
        await asyncio.gather(*tasks)
        logger.info(f"Updated prompts for {len(self.clients)} clients")

    async def put_video_frame(self, frame: av.VideoFrame):
        """Distribute video frames among clients using round-robin"""
        current_time = time.time()
        if current_time - self.last_frame_time < self.min_frame_interval:
            return  # Skip frame if too soon
            
        self.last_frame_time = current_time
        
        # Generate a unique frame ID - use sequential IDs for better ordering
        if not hasattr(self, 'next_frame_id'):
            self.next_frame_id = 1
        
        frame_id = self.next_frame_id
        self.next_frame_id += 1

        frame.side_data.frame_id = frame_id
        
        # Preprocess the frame
        frame.side_data.input = self.video_preprocess(frame)
        frame.side_data.skipped = False
        
        # Select the next client in round-robin fashion
        client_index = self.current_client_index
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        
        # Store mapping of which client is processing this frame
        self.client_frame_mapping[frame_id] = client_index
        
        # Send frame to the selected client
        self.clients[client_index].put_video_input(frame)
        
        # Also add to the incoming queue for reference
        await self.video_incoming_frames.put((frame_id, frame))
        
        logger.debug(f"Sent frame {frame_id} to client {client_index}")

    async def put_audio_frame(self, frame: av.AudioFrame):
        # For now, only use the first client for audio
        if not self.clients:
            return
            
        frame.side_data.input = self.audio_preprocess(frame)
        frame.side_data.skipped = False
        self.clients[0].put_audio_input(frame)
        await self.audio_incoming_frames.put(frame)

    def audio_preprocess(self, frame: av.AudioFrame) -> Union[torch.Tensor, np.ndarray]:
        return frame.to_ndarray().ravel().reshape(-1, 2).mean(axis=1).astype(np.int16)
    
    def video_preprocess(self, frame: av.VideoFrame) -> Union[torch.Tensor, np.ndarray]:
        # Convert directly to tensor, avoiding intermediate numpy array when possible
        if hasattr(frame, 'to_tensor'):
            tensor = frame.to_tensor()
        else:
            # If direct tensor conversion not available, use numpy
            frame_np = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(frame_np)
        
        # Normalize to [0,1] range and add batch dimension
        return tensor.float().div(255.0).unsqueeze(0)

    def video_postprocess(self, output: Union[torch.Tensor, np.ndarray]) -> av.VideoFrame:
        return av.VideoFrame.from_ndarray(
            (output.squeeze(0).permute(1, 2, 0) * 255.0)
            .clamp(0, 255)
            .to(dtype=torch.uint8)
            .cpu()
            .numpy(),
            format='rgb24'
        )

    def audio_postprocess(self, output: Union[torch.Tensor, np.ndarray]) -> av.AudioFrame:
        return av.AudioFrame.from_ndarray(np.repeat(output, 2).reshape(1, -1))

    async def get_processed_video_frame(self):
        try:
            # Get the original frame from the incoming queue first to maintain timing
            frame_id, frame = await self.video_incoming_frames.get()
            
            # Skip frames if we're falling behind
            '''
            while not self.video_incoming_frames.empty():
                # Get newer frame and mark old one as skipped
                frame.side_data.skipped = True
                frame_id, frame = await self.video_incoming_frames.get()
                logger.info(f"Skipped older frame {frame_id} to catch up")
            '''
            # Get the processed frame from our output queue
            processed_frame_id, out_tensor = await self.processed_video_frames.get()
            
            if processed_frame_id != frame_id:
                logger.debug(f"Frame ID mismatch: expected {frame_id}, got {processed_frame_id}")
                pass
            
            # Process the frame
            processed_frame = self.video_postprocess(out_tensor)
            processed_frame.pts = frame.pts
            processed_frame.time_base = frame.time_base
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error in get_processed_video_frame: {str(e)}")
            # Create a black frame as fallback
            black_frame = av.VideoFrame(width=self.width, height=self.height, format='rgb24')
            return black_frame

    async def get_processed_audio_frame(self):
        # Only use the first client for audio
        if not self.clients:
            logger.warning("No clients available for audio processing")
            return av.AudioFrame(format='s16', layout='mono', samples=1024)
            
        frame = await self.audio_incoming_frames.get()
        if frame.samples > len(self.processed_audio_buffer):
            out_tensor = await self.clients[0].get_audio_output()
            self.processed_audio_buffer = np.concatenate([self.processed_audio_buffer, out_tensor])
        out_data = self.processed_audio_buffer[:frame.samples]
        self.processed_audio_buffer = self.processed_audio_buffer[frame.samples:]

        processed_frame = self.audio_postprocess(out_data)
        processed_frame.pts = frame.pts
        processed_frame.time_base = frame.time_base
        processed_frame.sample_rate = frame.sample_rate
        
        return processed_frame

    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata."""
        # Note that we pull the node info from the first client (as they should all be the same)
        # TODO: This is just retrofitting the functionality of the comfy embedded client, there could be major improvements here
        nodes_info = await self.clients[0].get_available_nodes()
        return nodes_info

    async def cleanup(self):
        """Clean up all clients and background tasks"""
        self.running = False
        
        # Cancel collector task
        if hasattr(self, 'collector_task') and not self.collector_task.done():
            self.collector_task.cancel()
            try:
                await self.collector_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all clients
        cleanup_tasks = []
        for client in self.clients:
            cleanup_tasks.append(client.cleanup())
            
        await asyncio.gather(*cleanup_tasks)
        logger.info("All clients cleaned up")

    async def _dynamic_output_pacer(self):
        while self.running:
            # Only release if the next expected frame is available
            if self.next_expected_frame_id is not None and self.next_expected_frame_id in self.ordered_frames:
                timestamp, tensor = self.ordered_frames.pop(self.next_expected_frame_id)
                now = time.time()

                # Calculate dynamic interval based on output history
                if self.last_output_time is not None:
                    actual_interval = now - self.last_output_time
                    self.frame_interval_history.append(actual_interval)
                    avg_interval = sum(self.frame_interval_history) / len(self.frame_interval_history)
                    self.output_interval = avg_interval
                self.last_output_time = now

                await self.processed_video_frames.put((self.next_expected_frame_id, tensor))
                logger.info(f"Released frame {self.next_expected_frame_id} to output queue")

                # Update next expected frame ID
                if self.ordered_frames:
                    self.next_expected_frame_id = min(self.ordered_frames.keys())
                else:
                    self.next_expected_frame_id += 1

                # Sleep for the dynamic interval, but don't sleep negative time
                await asyncio.sleep(max(self.output_interval, 0.001))
            else:
                # No frame ready, wait a bit and check again
                await asyncio.sleep(0.005)

    async def start_clients(self):
        """Start the clients based on the client_mode (TOML or spawn)"""
        logger.info(f"Starting clients with mode: {self.client_mode}")
        
        self.clients = []
        
        if hasattr(self, 'client_mode') and self.client_mode == "toml":
            # Use config file to create clients
            for server_config in self.servers:
                self.clients.append(ComfyStreamClient(
                    host=server_config["host"],
                    port=server_config["port"],
                    spawn=False,
                ))
                
        elif hasattr(self, 'client_mode') and self.client_mode == "spawn":
            # Spin up clients as external processes
            ports = [8195 + i for i in range(self.workers)]
            
            for i in range(self.workers):
                client = ComfyStreamClient(
                    host="127.0.0.1",
                    port=ports[i],
                    spawn=True,
                    comfyui_path=os.path.join(self.workspace, "main.py"),
                    workspace=self.workspace,
                    comfyui_args=[
                        "--disable-cuda-malloc", 
                        "--gpu-only", 
                        "--preview-method", "none", 
                        "--listen", 
                        "--cuda-device", "0", 
                        "--fast", 
                        "--enable-cors-header", "*", 
                        "--port", str(ports[i]),
                        "--disable-xformers", 
                    ],
                )
                self.clients.append(client)
                
        else:
            raise ValueError(f"Unknown client_mode: {getattr(self, 'client_mode', 'None')}")
            
        # Start all ComfyUI servers in parallel if in spawn mode
        if hasattr(self, 'client_mode') and self.client_mode == "spawn":
            # First, launch all server processes in parallel
            for client in self.clients:
                if client.spawn:
                    client._launch_comfyui_server()
            
            # Now create async functions to check server readiness
            async def check_server_ready(client, timeout=60, check_interval=0.5):
                """Async version of waiting for server to be ready"""
                logger.info(f"Waiting for ComfyUI server on port {client.port} to be ready...")
                
                start_time = time.time()
                while time.time() - start_time < timeout:
                    # Check if process is still running
                    if client._comfyui_proc and client._comfyui_proc.poll() is not None:
                        return_code = client._comfyui_proc.poll()
                        logger.error(f"ComfyUI process exited with code {return_code} before it was ready")
                        raise RuntimeError(f"ComfyUI process exited with code {return_code}")
                        
                    # Try to connect to the server
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(2)
                        result = sock.connect_ex((client.host, client.port))
                        sock.close()
                        
                        if result == 0:
                            logger.info(f"ComfyUI server on port {client.port} is now accepting connections")
                            return
                    except Exception:
                        pass
                        
                    # Sleep and try again
                    await asyncio.sleep(check_interval)
                    
                # If we get here, the server didn't start in time
                logger.error(f"Timed out waiting for ComfyUI server on port {client.port}")
                if client._comfyui_proc:
                    client._comfyui_proc.terminate()
                    client._comfyui_proc = None
                raise RuntimeError(f"Timed out waiting for ComfyUI server on port {client.port}")
            
            # Wait for all servers to be ready in parallel
            wait_tasks = []
            for client in self.clients:
                if client.spawn:
                    wait_tasks.append(check_server_ready(client))
            
            if wait_tasks:
                logger.info(f"Waiting for {len(wait_tasks)} ComfyUI servers to become ready...")
                await asyncio.gather(*wait_tasks)
                logger.info(f"All {len(wait_tasks)} ComfyUI servers are ready")
                
        logger.info(f"Initialized {len(self.clients)} clients")
        return self.clients
        
# For backwards compatibility, maintain the original Pipeline name
Pipeline = MultiServerPipeline