import asyncio
import json
import uuid
import websockets
import base64
import aiohttp
import logging
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional, Union
import random
import time
import subprocess
import os
import socket

from comfystream import tensor_cache
from comfystream.utils_api import convert_prompt
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger(__name__)

class ComfyStreamClient:
    def __init__(
            self,
            host: str = "127.0.0.1",
            port: int = 8188,
            spawn: bool = False,
            comfyui_path: str = None,
            comfyui_args: list = None,
            workspace: str = None
        ):
        """
        Initialize the ComfyStream client to use the ComfyUI API.
        
        Args:
            host: The hostname or IP address of the ComfyUI server
            port: The port number of the ComfyUI server
            spawn: If True, launch a ComfyUI server when start_server is called
            comfyui_path: Path to the ComfyUI main.py file (required if spawn=True)
            comfyui_args: Additional arguments for ComfyUI
            workspace: The workspace directory for ComfyUI
        """
        self.host = host
        self.port = port
        self.spawn = spawn
        self.comfyui_path = comfyui_path
        self.comfyui_args = comfyui_args or []
        self.workspace = workspace
        self._comfyui_proc = None
        
        # Server launch is deferred to start_server method
        
        self.server_address = f"ws://{host}:{port}/ws"
        self.api_base_url = f"http://{host}:{port}/api"
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.current_prompts = []
        self.running_prompts = {}
        self.cleanup_lock = asyncio.Lock()
        self.buffer = BytesIO()
        self.execution_complete_event = asyncio.Event()

        self._ws_listener_task = None
        self._prompt_id = None
        self._current_frame_id = None  # Track the current frame being processed
        self._frame_id_mapping = {}    # Map prompt_ids to frame_ids
    
        logger.info(f"ComfyStreamClient initialized with host: {host}, port: {port}, client_id: {self.client_id}")
    
    async def set_prompts(self, prompts: List[Dict]):
        """Set prompts and run them (compatible with original interface)"""
        # Convert prompts (this already randomizes seeds, but we'll enhance it)
        self.current_prompts = [convert_prompt(prompt) for prompt in prompts]
        
        # Create tasks for each prompt
        for idx in range(len(self.current_prompts)):
            task = asyncio.create_task(self.run_prompt(idx))
            self.running_prompts[idx] = task
            
        logger.info(f"Set {len(self.current_prompts)} prompts for execution")
    
    async def update_prompts(self, prompts: List[Dict]):
        """Update existing prompts (compatible with original interface)"""
        if len(prompts) != len(self.current_prompts):
            raise ValueError(
                "Number of updated prompts must match the number of currently running prompts."
            )
        self.current_prompts = [convert_prompt(prompt) for prompt in prompts]
        logger.info(f"Updated {len(self.current_prompts)} prompts")
    
    async def run_prompt(self, prompt_index: int):
        """Run a prompt continuously, processing new frames as they arrive"""
        logger.info(f"Running prompt {prompt_index}")
        
        # Make sure WebSocket is connected
        await self._connect_websocket()
        
        # Always set execution complete at start to allow first frame to be processed
        self.execution_complete_event.set()
        
        try:
            while True:
                # Wait until we have tensor data available before sending prompt
                if tensor_cache.image_inputs.empty():
                    await asyncio.sleep(0.01)  # Reduced sleep time for faster checking
                    continue
                
                # Clear event before sending a new prompt
                if self.execution_complete_event.is_set():
                    # Reset execution state for next frame
                    self.execution_complete_event.clear()
                    
                    # Queue the prompt with the current frame
                    await self._execute_prompt(prompt_index)
                    
                    # Wait for execution completion with timeout
                    try:
                        logger.debug("Waiting for execution to complete (max 10 seconds)...")
                        await asyncio.wait_for(self.execution_complete_event.wait(), timeout=10.0)
                        logger.debug("Execution complete, ready for next frame")
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for execution, forcing continuation")
                        self.execution_complete_event.set()
                else:
                    # If execution is not complete, check again shortly
                    await asyncio.sleep(0.01)  # Short sleep to prevent CPU spinning
                
        except asyncio.CancelledError:
            logger.info(f"Prompt {prompt_index} execution cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in run_prompt: {str(e)}")
            raise
    
    async def _connect_websocket(self):
        """Connect to the ComfyUI WebSocket endpoint"""
        try:
            if self.ws is not None and self.ws.open:
                return self.ws

            # Close existing connection if any
            if self.ws is not None:
                try:
                    await self.ws.close()
                except:
                    pass
                self.ws = None
            
            logger.info(f"Connecting to WebSocket at {self.server_address}?clientId={self.client_id}")
            
            try:
                # Connect with proper error handling
                self.ws = await websockets.connect(
                    f"{self.server_address}?clientId={self.client_id}",
                    ping_interval=5,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=None,  # No limit on message size
                    ssl=None
                )
                
                logger.info("WebSocket connected successfully")
                
                # Start the listener task if not already running
                if self._ws_listener_task is None or self._ws_listener_task.done():
                    self._ws_listener_task = asyncio.create_task(self._ws_listener())
                    logger.info("Started WebSocket listener task")
                    
                return self.ws
                
            except (websockets.exceptions.WebSocketException, ConnectionError, OSError) as e:
                logger.error(f"WebSocket connection error: {e}")
                self.ws = None
                # Signal execution complete to prevent hanging if connection fails
                self.execution_complete_event.set()
                # Retry after a delay
                await asyncio.sleep(1)
                return await self._connect_websocket()
                
        except Exception as e:
            logger.error(f"Unexpected error in _connect_websocket: {e}")
            self.ws = None
            # Signal execution complete to prevent hanging
            self.execution_complete_event.set()
            return None
    
    async def _ws_listener(self):
        """Listen for WebSocket messages and process them"""
        try:
            logger.info(f"WebSocket listener started")
            while True:
                if self.ws is None:
                    try:
                        await self._connect_websocket()
                    except Exception as e:
                        logger.error(f"Error connecting to WebSocket: {e}")
                        await asyncio.sleep(1)
                        continue
                
                try:
                    # Receive and process messages
                    message = await self.ws.recv()

                    if isinstance(message, str):
                        # Process JSON messages
                        await self._handle_text_message(message)
                    else:
                        # Handle binary data - likely image preview or tensor data
                        await self._handle_binary_message(message)
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WebSocket connection closed")
                    self.ws = None
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error in WebSocket listener: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("WebSocket listener cancelled")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in WebSocket listener: {e}")
    
    async def _handle_text_message(self, message: str):
        """Process text (JSON) messages from the WebSocket"""
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")

            # logger.info(f"Received message type: {message_type}")
            logger.debug(f"{data}")

            # Example output
            '''
            15:15:58 [INFO] Received message type: executing
            15:15:58 [INFO] {'type': 'executing', 'data': {'node': '18', 'display_node': '18', 'prompt_id': '6f983049-dca4-4935-9f36-d2bff7b744fa'}}
            15:15:58 [INFO] Received message type: executed
            15:15:58 [INFO] {'type': 'executed', 'data': {'node': '18', 'display_node': '18', 'output': {'images': [{'source': 'websocket', 'content-type': 'image/png', 'type': 'output'}]}, 'prompt_id': '6f983049-dca4-4935-9f36-d2bff7b744fa'}}
            15:15:58 [INFO] Received message type: execution_success
            15:15:58 [INFO] {'type': 'execution_success', 'data': {'prompt_id': '6f983049-dca4-4935-9f36-d2bff7b744fa', 'timestamp': 1744139758250}}
            '''

            # Handle different message types to have fun with!

            '''
            if message_type == "status":
                # Status message with comfy_ui's queue information
                queue_remaining = data.get("data", {}).get("queue_remaining", 0)
                exec_info = data.get("data", {}).get("exec_info", {})
                if queue_remaining == 0 and not exec_info:
                    logger.info("Queue empty, no active execution")
                else:
                    logger.info(f"Queue status: {queue_remaining} items remaining")
            '''

            '''
            if message_type == "progress":
                if "data" in data and "value" in data["data"]:
                    progress = data["data"]["value"]
                    max_value = data["data"].get("max", 100)
                    # Log the progress for debugging
                    logger.info(f"Progress: {progress}/{max_value}")
            '''

            if message_type == "execution_start":
                if "data" in data and "prompt_id" in data["data"]:
                    self._prompt_id = data["data"]["prompt_id"]
                    logger.info(f"Execution started for prompt {self._prompt_id}")

                    # Let's queue the next prompt here!
                    self.execution_complete_event.set()

            '''
            if message_type == "executing":
                if "data" in data:
                    if "prompt_id" in data["data"]:
                        self._prompt_id = data["data"]["prompt_id"]
                    if "node" in data["data"]:
                        node_id = data["data"]["node"]
                        logger.info(f"Executing node: {node_id}")

                        # Let's check which node_id is a LoadImageBase64 node
                    # and set the execution complete event for that node    
                    for prompt_index, prompt in enumerate(self.current_prompts):
                        for node_id, node in prompt.items():
                            if (node_id == executing_node_id and isinstance(node, dict) and node.get("class_type") in ["LoadImageBase64"]):
                                logger.info(f"Setting execution complete event for LoadImageBase64 node {node_id}")
                                self.execution_complete_event.set()
                                break
            '''

            '''
            if message_type == "executed":
                # This is sent when a node is completely done
                if "data" in data and "node" in data["data"]:
                    node_id = data["data"]["node"]
                    logger.info(f"Node execution complete: {node_id}")
            '''

            '''
            if message_type in ["execution_cached", "execution_error", "execution_complete", "execution_interrupted"]:
                logger.info(f"{message_type} message received for prompt {self._prompt_id}")                
                # Always signal completion for these terminal states
                # self.execution_complete_event.set()
                logger.info(f"Set execution_complete_event from {message_type}")
                pass
            '''

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message[:100]}...")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            # Signal completion on error to prevent hanging
            self.execution_complete_event.set()
    
    async def _handle_binary_message(self, binary_data):
        """Process binary messages from the WebSocket"""
        try:
            # Early return if message is too short
            if len(binary_data) <= 8:
                # self.execution_complete_event.set()
                return
            
            # Extract header data only when needed
            event_type = int.from_bytes(binary_data[:4], byteorder='little')
            format_type = int.from_bytes(binary_data[4:8], byteorder='little')
            data = binary_data[8:]
            
            # Quick check for image format
            is_image = data[:2] in [b'\xff\xd8', b'\x89\x50']
            if not is_image:
                # self.execution_complete_event.set()
                return
            
            # Process image data directly
            try:
                img = Image.open(BytesIO(data))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    
                with torch.no_grad():
                    tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                
                # Try to get frame_id from mapping using current prompt_id
                frame_id = None
                if hasattr(self, '_prompt_id') and self._prompt_id in self._frame_id_mapping:
                    frame_id = self._frame_id_mapping.get(self._prompt_id)
                    logger.debug(f"Using frame_id {frame_id} from prompt_id {self._prompt_id}")
                elif hasattr(self, '_current_frame_id') and self._current_frame_id is not None:
                    frame_id = self._current_frame_id
                    logger.debug(f"Using current frame_id {frame_id}")
                
                # Add to output queue - include frame_id if available
                if frame_id is not None:
                    tensor_cache.image_outputs.put_nowait((frame_id, tensor))
                    logger.debug(f"Added tensor with frame_id {frame_id} to output queue")
                else:
                    tensor_cache.image_outputs.put_nowait(tensor)
                    logger.debug("Added tensor without frame_id to output queue")
                
                # We will execute the next prompt from message_type == "execution_start" instead
                # self.execution_complete_event.set()
                
            except Exception as img_error:
                logger.error(f"Error processing image: {img_error}")
                # self.execution_complete_event.set()
                
        except Exception as e:
            logger.error(f"Error handling binary message: {e}")
            # self.execution_complete_event.set()
    
    async def _execute_prompt(self, prompt_index: int):
        try:
            # Get the prompt to execute
            prompt = self.current_prompts[prompt_index]
            
            # Check if we have a frame waiting to be processed
            if not tensor_cache.image_inputs.empty():
                # Get the most recent frame only
                frame_or_tensor = None
                while not tensor_cache.image_inputs.empty():
                    frame_or_tensor = tensor_cache.image_inputs.get_nowait()
                
                # Extract frame ID if available in side_data
                frame_id = None
                if hasattr(frame_or_tensor, 'side_data'):
                    # Try to get frame_id from side_data
                    if hasattr(frame_or_tensor.side_data, 'frame_id'):
                        frame_id = frame_or_tensor.side_data.frame_id
                        logger.info(f"Found frame_id in side_data: {frame_id}")
                
                # Store current frame ID for binary message handler to use
                self._current_frame_id = frame_id
                
                # Find LoadImageBase64 nodes first
                load_image_nodes = []
                for node_id, node in prompt.items():
                    if isinstance(node, dict) and node.get("class_type") in ["LoadImageBase64"]:
                        load_image_nodes.append(node_id)
                
                if not load_image_nodes:
                    logger.warning("No LoadImageBase64 nodes found in the prompt")
                    self.execution_complete_event.set()
                    return
                
                # Process the tensor ONLY if we have nodes to send it to
                try:
                    # Get the actual tensor data - handle different input types
                    tensor = None
                    
                    # Handle different input types efficiently
                    if hasattr(frame_or_tensor, 'side_data') and hasattr(frame_or_tensor.side_data, 'input'):
                        tensor = frame_or_tensor.side_data.input
                    elif isinstance(frame_or_tensor, torch.Tensor):
                        tensor = frame_or_tensor
                    elif isinstance(frame_or_tensor, np.ndarray):
                        tensor = torch.from_numpy(frame_or_tensor).float()
                    elif hasattr(frame_or_tensor, 'to_ndarray'):
                        frame_np = frame_or_tensor.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                        tensor = torch.from_numpy(frame_np).unsqueeze(0)
                    
                    if tensor is None:
                        logger.error("Failed to get valid tensor data from input")
                        self.execution_complete_event.set()
                        return
                    
                    # Process tensor format only once - streamlined for speed and reliability
                    with torch.no_grad():
                        # Fast tensor normalization to ensure consistent output
                        try:
                            # TODO: Why is the UI sending different sizes? Should be fixed no? This breaks tensorrt
                            #       I'm sometimes seeing (BCHW): torch.Size([1, 384, 384, 3]), H=384, W=3
                            # Ensure minimum size of 512x512

                            # Handle batch dimension if present
                            if len(tensor.shape) == 4:  # BCHW format
                                tensor = tensor[0]  # Take first image from batch
                            
                            # Normalize to CHW format consistently
                            if len(tensor.shape) == 3 and tensor.shape[2] == 3:  # HWC format
                                tensor = tensor.permute(2, 0, 1)  # Convert to CHW
                            
                            # Handle single-channel case
                            if len(tensor.shape) == 3 and tensor.shape[0] == 1:
                                tensor = tensor.repeat(3, 1, 1)  # Convert grayscale to RGB
                            
                            # Ensure tensor is on CPU
                            if tensor.is_cuda:
                                tensor = tensor.cpu()
                            
                            # Always resize to 512x512 for consistency (faster than checking dimensions first)
                            tensor = tensor.unsqueeze(0)  # Add batch dim for interpolate
                            tensor = torch.nn.functional.interpolate(
                                tensor, size=(512, 512), mode='bilinear', align_corners=False
                            )
                            tensor = tensor[0]  # Remove batch dimension

                            # ====
                            # PIL method
                            '''
                            # Direct conversion to PIL without intermediate numpy step for speed
                            tensor_np = (tensor.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).numpy()
                            img = Image.fromarray(tensor_np)
                            img.save(self.buffer, format="JPEG", quality=90, optimize=True)
                            '''
                            
                            # ====
                            # torchvision method (more performant - TODO: need to test further)
                            # Direct conversion to PIL without intermediate numpy step
                            # Fast JPEG encoding with reduced quality for better performance
                            tensor_pil = to_pil_image(tensor.clamp(0, 1))
                            tensor_pil.save(self.buffer, format="JPEG", quality=75, optimize=True)
                            # ====

                            self.buffer.seek(0)
                            img_base64 = base64.b64encode(self.buffer.getvalue()).decode('utf-8')
                            
                        except Exception as e:
                            logger.warning(f"Error in tensor processing: {e}, creating fallback image")
                            # Create a standard 512x512 placeholder if anything fails
                            img = Image.new('RGB', (512, 512), color=(100, 149, 237))
                            self.buffer = BytesIO()
                            img.save(self.buffer, format="JPEG", quality=90)
                            self.buffer.seek(0)
                            img_base64 = base64.b64encode(self.buffer.getvalue()).decode('utf-8')
                        
                        # Add timestamp for cache busting (once, outside the try/except)
                        timestamp = int(time.time() * 1000)
                    
                    # Update all nodes with the SAME base64 string
                    for node_id in load_image_nodes:
                        prompt[node_id]["inputs"]["image"] = img_base64
                        prompt[node_id]["inputs"]["_timestamp"] = timestamp
                        # Use timestamp as cache buster
                        prompt[node_id]["inputs"]["_cache_buster"] = str(timestamp)
                
                except Exception as e:
                    logger.error(f"Error converting tensor to base64: {e}")
                    self.execution_complete_event.set()
                    return
                
                # Execute the prompt via API
                async with aiohttp.ClientSession() as session:
                    api_url = f"{self.api_base_url}/prompt"
                    payload = {
                        "prompt": prompt,
                        "client_id": self.client_id
                    }
                    
                    async with session.post(api_url, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            self._prompt_id = result.get("prompt_id")
                            
                            # Map prompt_id to frame_id for later retrieval
                            if frame_id is not None:
                                self._frame_id_mapping[self._prompt_id] = frame_id
                                logger.info(f"Mapped prompt_id {self._prompt_id} to frame_id {frame_id}")
                        else:
                            error_text = await response.text()
                            logger.error(f"Error queueing prompt: {response.status} - {error_text}")
                            self.execution_complete_event.set()
            else:
                logger.info("No tensor in input queue, skipping prompt execution")
                self.execution_complete_event.set()
                
        except Exception as e:
            logger.error(f"Error executing prompt: {e}")
            self.execution_complete_event.set()
    
    async def cleanup(self):
        """Clean up resources and reset connection state completely."""
        logger.info("Performing client cleanup and connection reset")
        
        # Cancel the WebSocket listener task
        if self._ws_listener_task is not None and not self._ws_listener_task.done():
            self._ws_listener_task.cancel()
            try:
                await self._ws_listener_task
            except asyncio.CancelledError:
                pass
            self._ws_listener_task = None
        
        # Close WebSocket connection
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.ws = None
        
        # Reset all state variables
        self._prompt_id = None
        self._current_frame_id = None
        self._frame_id_mapping = {}
        self.current_prompts = []
        
        # Cancel any running prompt tasks
        for task in self.running_prompts.values():
            if not task.done():
                task.cancel()
        self.running_prompts = {}
        
        # Reset the execution event
        self.execution_complete_event.set()
        
        # Clean up queues
        await self.cleanup_queues()
        
        # Reset buffer
        self.buffer = BytesIO()
        
        logger.info("Client cleanup completed, connection will be reestablished on next use")
    
    async def cleanup_queues(self):
        """Clean up tensor queues"""
        while not tensor_cache.image_inputs.empty():
            tensor_cache.image_inputs.get()

        while not tensor_cache.audio_inputs.empty():
            tensor_cache.audio_inputs.get()

        while tensor_cache.image_outputs.qsize() > 0:
            try:
                await tensor_cache.image_outputs.get()
            except:
                pass

        while tensor_cache.audio_outputs.qsize() > 0:
            try:
                await tensor_cache.audio_outputs.get()
            except:
                pass
        
        logger.info("Tensor queues cleared")
    
    def put_video_input(self, frame):
        if tensor_cache.image_inputs.full():
            tensor_cache.image_inputs.get(block=True)
        tensor_cache.image_inputs.put(frame)
    
    def put_audio_input(self, frame):
        """Put audio frame into tensor cache"""
        tensor_cache.audio_inputs.put(frame)
        
    async def get_video_output(self):
        """Get processed video frame from tensor cache"""
        result = await tensor_cache.image_outputs.get()
        
        # Check if the result is a tuple with frame_id
        if isinstance(result, tuple) and len(result) == 2:
            frame_id, tensor = result
            logger.info(f"Got processed tensor from output queue with frame_id {frame_id}")
            # Return both the frame_id and tensor to help with ordering in the pipeline
            return frame_id, tensor
        else:
            # If it's not a tuple with frame_id, just return the tensor
            logger.info("Got processed tensor from output queue without frame_id")
            return result
    
    async def get_audio_output(self):
        """Get processed audio frame from tensor cache"""
        return await tensor_cache.audio_outputs.get()
    
    async def get_available_nodes(self) -> Dict[int, Dict[str, Any]]:
        """
        Retrieves detailed information about the nodes used in the current prompts
        by querying the ComfyUI /object_info API endpoint.

        Returns:
            A dictionary where keys are prompt indices and values are dictionaries
            mapping node IDs to their information, matching the required UI format.
        
        The idea of this function is to replicate the functionality of comfy embedded client import_all_nodes_in_workspace
        TODO: Why not support ckpt_name and lora_name as dropdown selectors on UI?
        """

        if not self.current_prompts:
            logger.warning("No current prompts set. Cannot get node info.")
            return {}

        all_prompts_nodes_info: Dict[int, Dict[str, Any]] = {}
        all_needed_class_types = set()

        # Collect all unique class types across all prompts first
        for prompt in self.current_prompts:
            for node in prompt.values():
                if isinstance(node, dict) and 'class_type' in node:
                    all_needed_class_types.add(node['class_type'])

        class_info_cache: Dict[str, Any] = {}

        async with aiohttp.ClientSession() as session:
            fetch_tasks = []
            for class_type in all_needed_class_types:
                api_url = f"{self.api_base_url}/object_info/{class_type}"
                fetch_tasks.append(self._fetch_object_info(session, api_url, class_type))

            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Populate cache from results
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    class_type, info = result
                    if info:
                        class_info_cache[class_type] = info
                elif isinstance(result, Exception):
                    logger.error(f"An exception occurred during object_info fetch task: {result}")
        
        # Now, build the output structure for each prompt
        for prompt_index, prompt in enumerate(self.current_prompts):
            nodes_info: Dict[str, Any] = {}
            for node_id, node_data in prompt.items():
                if not isinstance(node_data, dict) or 'class_type' not in node_data:
                    logger.debug(f"Skipping invalid node data for node_id {node_id} in prompt {prompt_index}")
                    continue

                class_type = node_data['class_type']
                # Let's skip the native api i/o nodes for now, subject to change
                if class_type in ['LoadImageBase64', 'SendImageWebsocket']:
                    continue

                node_info = {
                    'class_type': class_type,
                    'inputs': {}
                }

                specific_class_info = class_info_cache.get(class_type)

                if specific_class_info and 'input' in specific_class_info:
                    input_definitions = {}
                    required_inputs = specific_class_info['input'].get('required', {})
                    optional_inputs = specific_class_info['input'].get('optional', {})

                    if isinstance(required_inputs, dict):
                        input_definitions.update(required_inputs)
                    if isinstance(optional_inputs, dict):
                        input_definitions.update(optional_inputs)

                    if 'inputs' in node_data and isinstance(node_data['inputs'], dict):
                        for input_name, input_value in node_data['inputs'].items():
                            input_def = input_definitions.get(input_name)
                            
                            # Format the input value as a tuple if it's a list with node references
                            if isinstance(input_value, list) and len(input_value) == 2 and isinstance(input_value[0], str) and isinstance(input_value[1], int):
                                input_value = tuple(input_value)  # Convert [node_id, output_index] to (node_id, output_index)

                            # Create Enum-like objects for certain types
                            def create_enum_format(type_name):
                                # Format the type as <IO.TYPE_NAME: 'TYPE_NAME'>
                                return f"<IO.{type_name}: '{type_name}'>"
                            
                            input_details = {
                                'value': input_value,
                                'type': 'unknown',  # Default type
                                'min': None, 
                                'max': None,
                                'widget': None  # Default, all widgets should be None to match format
                            }

                            # Parse the definition tuple/list if valid
                            if isinstance(input_def, (list, tuple)) and len(input_def) > 0:
                                config = None
                                # Check for config dict as the second element
                                if len(input_def) > 1 and isinstance(input_def[1], dict):
                                    config = input_def[1]

                                # Check for COMBO type (first element is list/tuple of options)
                                if input_name in ['ckpt_name', 'lora_name']:
                                    # For checkpoint and lora names, use STRING type instead of combo list
                                    input_details['type'] = create_enum_format('STRING')
                                elif isinstance(input_def[0], (list, tuple)):
                                    input_details['type'] = input_def[0]  # Type is the list of options
                                    # Don't set widget for combo
                                else:
                                    # Regular type (string or enum)
                                    input_type_raw = input_def[0]
                                    # Keep raw type name for certain types to match format
                                    if hasattr(input_type_raw, 'name'):
                                        # Special handling for CLIP and STRING to match expected format
                                        type_name = str(input_type_raw.name)
                                        if type_name in ('CLIP', 'STRING'):
                                            # Create Enum-like format that matches format in desired output
                                            input_details['type'] = create_enum_format(type_name)
                                        else:
                                            input_details['type'] = type_name
                                    else:
                                        # For non-enum types
                                        input_details['type'] = str(input_type_raw)

                                    # Extract constraints/widget from config if it exists
                                    if config:
                                        for key in ['min', 'max']:  # Only include these, skip widget/step/round
                                            if key in config:
                                                input_details[key] = config[key]

                            node_info['inputs'][input_name] = input_details
                    else:
                        logger.debug(f"Node {node_id} ({class_type}) has no 'inputs' dictionary.")
                elif class_type not in class_info_cache:
                    logger.warning(f"No cached info found for class_type: {class_type} (node_id: {node_id}).")
                else:
                    logger.debug(f"Class info for {class_type} does not contain an 'input' key.")
                    # If class info exists but no 'input' key, still add node with empty inputs dict

                nodes_info[node_id] = node_info
            
            # Only add if there are any nodes after filtering
            if nodes_info:
                all_prompts_nodes_info[prompt_index] = nodes_info

        return all_prompts_nodes_info

    async def start_server(self):
        """Launch the ComfyUI server if spawn is True"""
        if self.spawn:
            if not self.comfyui_path:
                raise ValueError("comfyui_path must be provided when spawn=True")
            self._launch_comfyui_server()
            self._wait_for_server_ready()
            logger.info("ComfyUI server started successfully")
        else:
            logger.info("Using existing ComfyUI server (spawn=False)")

    async def _fetch_object_info(self, session: aiohttp.ClientSession, url: str, class_type: str) -> Optional[tuple[str, Any]]:
        """Helper function to fetch object info for a single class type."""
        try:
            logger.debug(f"Fetching object info for: {class_type} from {url}")
            async with session.get(url) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                        # Extract the actual node info from the nested structure
                        if class_type in data and isinstance(data[class_type], dict):
                            node_specific_info = data[class_type]
                            logger.debug(f"Successfully fetched and extracted info for {class_type}")
                            return class_type, node_specific_info
                        else:
                             logger.error(f"Unexpected response structure for {class_type}. Key missing or not a dict. Response: {data}")

                    except aiohttp.ContentTypeError:
                         logger.error(f"Failed to decode JSON for {class_type}. Status: {response.status}, Content-Type: {response.headers.get('Content-Type')}, Response: {await response.text()[:200]}...") # Log beginning of text
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON received for {class_type}. Status: {response.status}, Error: {e}, Response: {await response.text()[:200]}...")
                else:
                    error_text = await response.text()
                    logger.error(f"Error fetching info for {class_type}: {response.status} - {error_text[:200]}...")
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error fetching info for {class_type} ({url}): {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching info for {class_type} ({url}): {e}")

        # Return class_type and None if any error occurred
        return class_type, None

    def _launch_comfyui_server(self):
        """Launch ComfyUI as a subprocess"""
        logger.info(f"Spawning ComfyUI server on port {self.port}...")
        
        # Build the command with just the basics
        cmd = [
            "python", self.comfyui_path,
        ]
        
        # Add the arguments from comfyui_args if provided
        if self.comfyui_args:
            cmd.extend(self.comfyui_args)
        else:
            # Only add default arguments if comfyui_args was not provided
            cmd.extend([
                "--disable-cuda-malloc",  # Helps prevent CUDA memory issues
                "--gpu-only",             # Use GPU for all operations when possible
                "--preview-method", "none", # Disable previews to save memory
                "--listen",
                "--port", str(self.port),
                "--fast",
                "--enable-cors-header", "\"*\"",
                "--disable-xformers",     # More compatible with some systems
            ])
            
            # Add workspace if provided and not in comfyui_args
            if self.workspace:
                cmd.extend(["--dir", self.workspace])
            
            # Check if CUDA is available and add device argument
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                cmd.extend(["--cuda-device", cuda_device])
        
        # Always ensure port is set correctly (override if provided in comfyui_args)
        # Remove any existing --port argument
        if "--port" in cmd:
            port_index = cmd.index("--port")
            # Remove both the flag and its value
            if port_index + 1 < len(cmd):
                cmd.pop(port_index + 1)
            cmd.pop(port_index)
        
        # Add our port
        cmd.extend(["--port", str(self.port)])
        
        # Start the process
        try:
            logger.info(f"Starting ComfyUI with command: {' '.join(cmd)}")
            self._comfyui_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
                text=True,
                bufsize=1,  # Line buffered
            )
            
            # Start a thread to log output
            def log_output(stream, level):
                for line in iter(stream.readline, ''):
                    # Check for known error patterns and log them at appropriate levels
                    if "error" in line.lower() or "exception" in line.lower():
                        logger.error(f"ComfyUI[{self.port}]: {line.strip()}")
                    else:
                        logger.debug(f"ComfyUI[{self.port}]: {line.strip()}")

            import threading
            threading.Thread(target=log_output, args=(self._comfyui_proc.stdout, logging.INFO), daemon=True).start()
            threading.Thread(target=log_output, args=(self._comfyui_proc.stderr, logging.INFO), daemon=True).start()
            
            logger.info(f"Started ComfyUI process with PID {self._comfyui_proc.pid}")
        except Exception as e:
            logger.error(f"Failed to spawn ComfyUI: {e}")
            raise
            
    def _wait_for_server_ready(self, timeout=60, check_interval=0.5):
        """Wait until the ComfyUI server is accepting connections"""
        logger.info(f"Waiting for ComfyUI server on port {self.port} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if process is still running
            if self._comfyui_proc and self._comfyui_proc.poll() is not None:
                return_code = self._comfyui_proc.poll()
                logger.error(f"ComfyUI process exited with code {return_code} before it was ready")
                raise RuntimeError(f"ComfyUI process exited with code {return_code}")
                
            # Try to connect to the server
            try:
                with socket.create_connection((self.host, self.port), timeout=2):
                    logger.info(f"ComfyUI server on port {self.port} is now accepting connections")
                    return
            except (ConnectionRefusedError, socket.timeout, OSError):
                # Sleep and try again
                time.sleep(check_interval)
                
        # If we get here, the server didn't start in time
        logger.error(f"Timed out waiting for ComfyUI server on port {self.port}")
        if self._comfyui_proc:
            self._comfyui_proc.terminate()
            self._comfyui_proc = None
        raise RuntimeError(f"Timed out waiting for ComfyUI server on port {self.port}")