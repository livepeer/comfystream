import av
import torch
import numpy as np
import asyncio
import logging
import time
from collections import OrderedDict
import collections
import os
import fractions

from typing import Any, Dict, Union, List, Optional, Deque
from comfystream.client_api import ComfyStreamClient
from comfystream.server.utils import temporary_log_level # Not sure exactly what this does
from comfystream.server.utils.config import ComfyConfig
from comfystream.frame_logging import log_frame_timing

WARMUP_RUNS = 5
logger = logging.getLogger(__name__)


class MultiServerPipeline:
    def __init__(
            self, 
            width: int = 512, 
            height: int = 512,
            config_path: Optional[str] = None, 
            max_frame_wait_ms: int = 500, 
            client_mode: str = "toml", 
            workspace: str = None,
            workers: int = 2, 
            cuda_devices: str = '0',
            workers_start_port: int = 8195,
            comfyui_log_level: str = None,
        ):
        """Initialize the pipeline with the given configuration.
        Args:
            width: The width of the video frames.
            height: The height of the video frames.
            workers: The number of ComfyUI clients to spin up (if client_mode is "spawn").
            config_path: The path to the ComfyUI config toml file (if client_mode is "toml").
            max_frame_wait_ms: The maximum number of milliseconds to wait for a frame before dropping it.
            client_mode: The mode to use for the ComfyUI clients.
                "toml": Use a config file to describe clients.
                "spawn": Spawn ComfyUI clients as external processes.
            workers_start_port: The starting port number for worker processes (default: 8195).
            cuda_devices: The list of CUDA devices to use for the ComfyUI clients.
            comfyui_log_level: The logging level for ComfyUI
        """

        # There are two methods for starting the clients:
        # 1. client_mode == "toml" -> Use a config file to describe clients.
        # 2. client_mode == "spawn" -> Spawn ComfyUI clients as external processes.

        self.clients = []
        self.workspace = workspace
        self.client_mode = client_mode

        if (client_mode == "toml"):
            # TOML Mode: Use a config file to describe existing ComfyUI Instances

            # Load server configurations
            self.config = ComfyConfig(config_path)
            self.servers = self.config.get_servers()
        elif (client_mode == "spawn"):
            # SPAWN Mode: Spawn new ComfyUI Instances automatically

            self.workers = workers
            self.workers_start_port = workers_start_port
            self.cuda_devices = cuda_devices
        
        # Clients started in /offer (this is due to when the page refreshes, the clients automatically close)
        # TODO: Perhaps a better way would be to keep the the clients alive while the server is alive?
        # self.start_clients()
        
        self.width = width
        self.height = height
        
        self.video_incoming_frames = asyncio.Queue()
        self.audio_incoming_frames = asyncio.Queue()
        
        # Queue for processed frames from all clients
        self.processed_video_frames = asyncio.Queue()
        
        # Track which client gets each frame (round-robin)
        self.last_frame_time = 0
        self.current_client_index = 0
        self.client_frame_mapping = {}  # Maps frame_id -> client_index
        
        # Frame ordering and timing
        self.max_frame_wait_ms = max_frame_wait_ms  # Max time to wait for a frame before dropping
        self.next_expected_frame_id = None  # Track expected frame ID
        self.ordered_frames = OrderedDict()  # Buffer for ordering frames (frame_id -> (timestamp, tensor))
        
        # Audio processing
        self.processed_audio_buffer = np.array([], dtype=np.int16)

        # Frame rate limiting
        self.min_frame_interval = 1/30  # Limit to 30 FPS
        
        # Create background task for collecting processed frames
        self.running = True
        self.collector_task = asyncio.create_task(self._collect_processed_frames())

        self.output_interval = 1/30  # Start with 30 FPS
        self.last_output_time = None
        self.frame_interval_history = collections.deque(maxlen=30)
        self.output_pacer_task = asyncio.create_task(self._dynamic_output_pacer())

        self.comfyui_log_level = comfyui_log_level
    
    async def _collect_processed_frames(self):
        """Background task to collect processed frames from all clients"""
        try:
            while self.running:
                for i, client in enumerate(self.clients):
                    try:
                        # Non-blocking check if client has output ready
                        if hasattr(client, '_prompt_id') and client._prompt_id is not None:
                            try:
                                # Use wait_for with small timeout to avoid blocking
                                frame_id, out_tensor = await asyncio.wait_for(
                                    client.get_video_output(), 
                                    timeout=0.001
                                )
                                
                                # Store frame with timestamp for ordering
                                current_time = time.time()
                                await self._add_frame_to_ordered_buffer(frame_id, current_time, out_tensor)
                                
                                # Remove the mapping
                                self.client_frame_mapping.pop(frame_id, None)

                                # logger.debug(f"Collected processed frame from client {i}, frame_id: {frame_id}")
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
        
        # Only release frames in strict sequential order
        while self.ordered_frames and self.next_expected_frame_id in self.ordered_frames:
            timestamp, tensor = self.ordered_frames.pop(self.next_expected_frame_id)
            await self.processed_video_frames.put((self.next_expected_frame_id, tensor))
            logger.debug(f"Released frame {self.next_expected_frame_id} to output queue")
            # Always increment to next sequential frame ID
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
                # logger.warning(f"Frame {self.next_expected_frame_id} exceeded max wait time, releasing anyway")
                # await self._release_ordered_frames()
                
                # Remove frame
                self.ordered_frames.pop(self.next_expected_frame_id)
                
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
 
        tensor = torch.rand(1, 3, 512, 512)  # Random values in [0,1]
        dummy_frame = av.VideoFrame(width=512, height=512, format="rgb24")
        dummy_frame.side_data.input = tensor
        dummy_frame.side_data.frame_received_time = time.time()

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

        # Set frame input as dummyframe with side_data.input set to a random tensor
        dummy_frame.side_data.input = torch.randn(1, self.height, self.width, 3)
        dummy_frame.side_data.frame_id = -1

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
            logger.info(f"Setting prompts for client {client.port}")
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
        ''' Put a video frame into the pipeline round-robin to all clients '''
        current_time = time.time()

        '''
        if current_time - self.last_frame_time < self.min_frame_interval:
            print(f"Skipping frame due to rate limiting: {current_time - self.last_frame_time} seconds since last frame")
            return  # Skip frame if too soon
        '''

        self.last_frame_time = current_time
        
        # Generate a unique frame ID - use sequential IDs for better ordering
        if not hasattr(self, 'next_frame_id'):
            self.next_frame_id = 1
        
        frame_id = self.next_frame_id
        self.next_frame_id += 1

        # Select the next client in round-robin fashion
        client_index = self.current_client_index
        self.current_client_index = (self.current_client_index + 1) % len(self.clients)
        
        # Store mapping of which client is processing this frame
        self.client_frame_mapping[frame_id] = client_index

        # Set side data for the frame
        frame.side_data.input = self.video_preprocess(frame)
        frame.side_data.frame_id = frame_id
        frame.side_data.skipped = False
        frame.side_data.frame_received_time = time.time()
        frame.side_data.client_index = client_index
        
        # Send frame to the selected client
        self.clients[client_index].put_video_input(frame)
        await self.video_incoming_frames.put(frame)
        
    async def put_audio_frame(self, frame: av.AudioFrame):
        ''' Not implemented yet '''
        return

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
        """Preprocess a video frame before processing.
        
        Args:
            frame: The video frame to preprocess
            
        Returns:
            The preprocessed frame as a tensor or numpy array
        """
        frame_np = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
        return torch.from_numpy(frame_np).unsqueeze(0)

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
            frame = await self.video_incoming_frames.get()
            
            # Get the processed frame from our output queue
            processed_frame_id, out_tensor = await self.processed_video_frames.get()
            
            # Process the frame
            processed_frame = self.video_postprocess(out_tensor)
            processed_frame.pts = frame.pts
            processed_frame.time_base = frame.time_base

            # Log frame timing asynchronously
            log_frame_timing(
                frame_id=processed_frame_id,
                frame_received_time=frame.side_data.frame_received_time,
                frame_processed_time=time.time(),
                client_index=frame.side_data.client_index,
            )
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error in get_processed_video_frame: {str(e)}")
            # Create a black frame as fallback
            black_frame = av.VideoFrame(width=self.width, height=self.height, format='rgb24')
            
            # Set timestamps to avoid TypeError during encoding
            # Use default values that work with the aiortc encoding pipeline
            black_frame.pts = 0
            black_frame.time_base = fractions.Fraction(1, 90000)  # Standard video timebase
            
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
        """Clean up resources used by the pipeline."""
        logger.info("Performing complete pipeline cleanup")
        
        # Cancel the dynamic output pacer task if it exists
        if hasattr(self, "_pacer_task") and self._pacer_task is not None:
            self._pacer_task.cancel()
            try:
                await self._pacer_task
            except asyncio.CancelledError:
                pass
            self._pacer_task = None
        
        # Cancel any frame timeout tasks
        if hasattr(self, "_timeout_task") and self._timeout_task is not None:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
            self._timeout_task = None
        
        # Reset frame tracking state
        self.next_expected_frame_id = None
        self.ordered_frames.clear()
        self.next_frame_id = 1  # Reset frame ID counter for new connection
        self.client_frame_mapping.clear()
        
        # Clear any queued frames
        while not self.video_incoming_frames.empty():
            try:
                self.video_incoming_frames.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Reset client state and connections
        for i, client in enumerate(self.clients):
            if client:
                # Clean up client resources
                try:
                    await client.cleanup()
                except Exception as e:
                    logger.error(f"Error during client {i} cleanup: {e}")
                
                # Reset client connection status
                if hasattr(client, 'ws_connected'):
                    client.ws_connected = False
                
                # Clear any client-specific execution state
                if hasattr(client, 'prompt_executing'):
                    client.prompt_executing = False
        
        # Mark clients as needing reinitialization
        self.clients_initialized = False
        
        # Clear any cached prompt mappings
        if hasattr(self, "_prompt_ids"):
            self._prompt_ids = {}
        
        # Reset warmup state
        if hasattr(self, "_warmup_complete"):
            self._warmup_complete = False
        
        # Reset any frame buffers
        if hasattr(self, "_frame_buffer"):
            self._frame_buffer.clear()
        
        # Ensure dynamic state like frame rate trackers are reset
        if hasattr(self, "_last_frame_time"):
            self._last_frame_time = None
        
        # Reset output counters
        self.output_counter = 0
        
        logger.info("Pipeline cleanup completed, clients will be reinitialized on next connection")

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
                logger.debug(f"Released frame {self.next_expected_frame_id} to output queue")

                # Always increment to next sequential frame ID
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
        self.startup_error = None
        
        try:
            if hasattr(self, 'client_mode') and self.client_mode == "toml":
                # Use config file to create clients
                for server_config in self.servers:
                    self.clients.append(ComfyStreamClient(
                        host=server_config["host"],
                        port=server_config["port"],
                        spawn=False,
                        comfyui_log_level=self.comfyui_log_level,
                    ))
                    
            elif hasattr(self, 'client_mode') and self.client_mode == "spawn":
                # Spin up clients as external processes
                ports = []
                cuda_device_list = [d.strip() for d in str(self.cuda_devices).split(',') if d.strip()]
                for device_idx, cuda_device in enumerate(cuda_device_list):
                    for worker_idx in range(self.workers):
                        port = self.workers_start_port + len(ports)
                        ports.append(port)
                        client = ComfyStreamClient(
                            host="127.0.0.1",
                            port=port,
                            spawn=True,
                            comfyui_path=os.path.join(self.workspace, "main.py"),
                            workspace=self.workspace,
                            comfyui_args=[
                                "--disable-cuda-malloc", 
                                "--gpu-only", 
                                "--preview-method", "none", 
                                "--listen", 
                                "--cuda-device", str(cuda_device), 
                                "--fast", 
                                "--enable-cors-header", "\"*\"", 
                                "--port", str(port),
                                "--disable-xformers", 
                            ],
                            comfyui_log_level=self.comfyui_log_level,
                        )
                        self.clients.append(client)
                        logger.info(f"Created worker {worker_idx+1}/{self.workers} for CUDA device {cuda_device} on port {port}")
                
            else:
                raise ValueError(f"Unknown client_mode: {getattr(self, 'client_mode', 'None')}")
            
            # Start all ComfyUI servers in parallel if in spawn mode
            if hasattr(self, 'client_mode') and self.client_mode == "spawn":
                try:
                    # Get all spawn clients
                    spawn_clients = [client for client in self.clients if client.spawn]
                    if spawn_clients:
                        logger.info(f"Starting {len(spawn_clients)} ComfyUI servers in parallel")
                        
                        # First validate all clients (keeping original validation logic)
                        for client in spawn_clients:
                            # These checks are from the original start_server method
                            if not client.comfyui_path:
                                raise ValueError("comfyui_path must be provided when spawn=True")
                            if not os.path.exists(client.comfyui_path):
                                raise FileNotFoundError(f"ComfyUI path does not exist: {client.comfyui_path}")
                        
                        # Start all server processes WITHOUT waiting for them to be ready
                        for client in spawn_clients:
                            client.launch_comfyui_server()
                        
                        # Now wait for all servers to be ready in parallel using thread pool
                        await asyncio.gather(*[
                            asyncio.to_thread(client.wait_for_server_ready) 
                            for client in spawn_clients
                        ])
                        
                except Exception as e:
                    # Clean up any clients that might have started
                    for client in self.clients:
                        if hasattr(client, '_comfyui_proc') and client._comfyui_proc:
                            try:
                                client._comfyui_proc.terminate()
                            except:
                                pass
                    
                    self.clients = []
                    self.startup_error = str(e)
                    logger.error(f"Failed to start ComfyUI servers: {e}")
                    return None
            
            logger.info(f"Initialized {len(self.clients)} clients")
            return self.clients

        except Exception as e:
            self.startup_error = str(e)
            logger.error(f"Error starting clients: {e}")
            self.clients = []
            return None

# For backwards compatibility, maintain the original Pipeline name
Pipeline = MultiServerPipeline