import torch
import numpy as np
import base64
import logging
import json
import traceback
import sys

logger = logging.getLogger(__name__)

# Log when the module is loaded
logger.debug("------------------ SendTensorWebSocket Module Loaded ------------------")

class SendTensorWebSocket:
    def __init__(self):
        # Output directory is not needed as we send via WebSocket
        logger.debug("SendTensorWebSocket instance created")
        pass

    @classmethod
    def INPUT_TYPES(cls):
        logger.debug("SendTensorWebSocket.INPUT_TYPES called")
        return {
            "required": {
                # Accept IMAGE input (typical output from VAE Decode)
                "tensor": ("IMAGE", ),
            },
            "hidden": {
                # These are needed for ComfyUI execution context
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()  # No direct output connection to other nodes
    FUNCTION = "save_tensor"
    OUTPUT_NODE = True
    CATEGORY = "ComfyStream/native"

    def save_tensor(self, tensor, prompt=None, extra_pnginfo=None):
        logger.debug("========== SendTensorWebSocket.save_tensor STARTED ==========")
        logger.info(f"SendTensorWebSocket received input. Type: {type(tensor)}")
        logger.debug(f"SendTensorWebSocket node is processing tensor with id: {id(tensor)}")
        
        # Log memory usage for debugging
        if torch.cuda.is_available():
            try:
                logger.debug(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                logger.debug(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            except Exception as e:
                logger.error(f"Error checking CUDA memory: {e}")
        
        if tensor is None:
            logger.error("SendTensorWebSocket received None tensor.")
            # Return error directly without ui nesting
            return {"comfystream_tensor_output": {"error": "Input tensor was None"}}

        try:
            # Log details about the tensor before processing
            logger.debug(f"Process tensor of type: {type(tensor)}")
            
            if isinstance(tensor, torch.Tensor):
                logger.debug("Processing torch.Tensor...")
                logger.info(f"Input tensor details: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
                
                # Additional handling for IMAGE-type tensors (0-1 float values, BCHW format)
                if len(tensor.shape) == 4:  # BCHW format (batch)
                    logger.debug(f"Tensor is batched (BCHW): {tensor.shape}")
                    logger.info(f"Tensor appears to be IMAGE batch. Min: {tensor.min().item()}, Max: {tensor.max().item()}")
                    logger.debug(f"First batch slice: min={tensor[0].min().item()}, max={tensor[0].max().item()}")
                    tensor = tensor[0]  # Select first image from batch
                    logger.debug(f"Selected first batch element. New shape: {tensor.shape}")
                
                if len(tensor.shape) == 3:  # CHW format (single image)
                    logger.debug(f"Tensor is CHW format: {tensor.shape}")
                    logger.info(f"Tensor appears to be single IMAGE. Min: {tensor.min().item()}, Max: {tensor.max().item()}")
                    
                    # Log first few values for debugging
                    logger.debug(f"First few values: {tensor.flatten()[:10].tolist()}")
                
                # Ensure the tensor is on CPU and detached
                logger.debug(f"Moving tensor to CPU. Current device: {tensor.device}")
                try:
                    tensor = tensor.cpu().detach()
                    logger.debug(f"Tensor moved to CPU successfully: {tensor.device}")
                except Exception as e:
                    logger.error(f"Error moving tensor to CPU: {e}")
                    logger.error(traceback.format_exc())
                    return {"comfystream_tensor_output": {"error": f"CPU transfer error: {str(e)}"}}
                
                # Convert to numpy
                logger.debug("Converting tensor to numpy array...")
                try:
                    np_array = tensor.numpy()
                    logger.debug(f"Conversion to numpy successful: shape={np_array.shape}, dtype={np_array.dtype}")
                    logger.debug(f"NumPy array memory usage: {np_array.nbytes / 1024**2:.2f} MB")
                except Exception as e:
                    logger.error(f"Error converting tensor to numpy: {e}")
                    logger.error(traceback.format_exc())
                    return {"comfystream_tensor_output": {"error": f"NumPy conversion error: {str(e)}"}}
                
                # Encode the tensor
                logger.debug("Converting numpy array to bytes...")
                try:
                    tensor_bytes = np_array.tobytes()
                    logger.debug(f"Tensor converted to bytes: {len(tensor_bytes)} bytes")
                except Exception as e:
                    logger.error(f"Error converting numpy array to bytes: {e}")
                    logger.error(traceback.format_exc())
                    return {"comfystream_tensor_output": {"error": f"Bytes conversion error: {str(e)}"}}
                
                logger.debug("Encoding bytes to base64...")
                try:
                    b64_data = base64.b64encode(tensor_bytes).decode('utf-8')
                    b64_size = len(b64_data)
                    logger.debug(f"Base64 encoding successful: {b64_size} characters")
                    if b64_size > 100:
                        logger.debug(f"Base64 sample: {b64_data[:50]}...{b64_data[-50:]}")
                except Exception as e:
                    logger.error(f"Error encoding to base64: {e}")
                    logger.error(traceback.format_exc())
                    return {"comfystream_tensor_output": {"error": f"Base64 encoding error: {str(e)}"}}
                
                # Prepare metadata
                shape = list(np_array.shape)
                dtype = str(np_array.dtype)
                
                logger.info(f"SendTensorWebSocket prepared tensor: shape={shape}, dtype={dtype}")
                
                # Construct the return value with simplified structure (no ui nesting)
                success_output = {
                    "comfystream_tensor_output": {
                        "b64_data": b64_data,
                        "shape": shape,
                        "dtype": dtype
                    }
                }
                
                # Log the structure of the output (avoid logging the actual base64 data which is large)
                output_structure = {
                    "comfystream_tensor_output": {
                        "b64_data": f"(base64 string of {b64_size} bytes)",
                        "shape": shape,
                        "dtype": dtype
                    }
                }
                logger.info(f"SendTensorWebSocket returning SUCCESS data structure: {json.dumps(output_structure)}")
                logger.debug("========== SendTensorWebSocket.save_tensor COMPLETED SUCCESSFULLY ==========")
                
                return success_output
            
            elif isinstance(tensor, np.ndarray):
                logger.debug("Processing numpy.ndarray...")
                logger.info(f"Input is numpy array: shape={tensor.shape}, dtype={tensor.dtype}")
                
                # Log memory details
                logger.debug(f"NumPy array memory usage: {tensor.nbytes / 1024**2:.2f} MB")
                logger.debug(f"First few values: {tensor.flatten()[:10].tolist()}")
                
                # Handle numpy array directly
                logger.debug("Converting numpy array to bytes...")
                try:
                    tensor_bytes = tensor.tobytes()
                    logger.debug(f"NumPy array converted to bytes: {len(tensor_bytes)} bytes")
                except Exception as e:
                    logger.error(f"Error converting numpy array to bytes: {e}")
                    logger.error(traceback.format_exc())
                    return {"comfystream_tensor_output": {"error": f"NumPy to bytes error: {str(e)}"}}
                
                logger.debug("Encoding bytes to base64...")
                try:
                    b64_data = base64.b64encode(tensor_bytes).decode('utf-8')
                    b64_size = len(b64_data)
                    logger.debug(f"Base64 encoding successful: {b64_size} characters")
                    if b64_size > 100:
                        logger.debug(f"Base64 sample: {b64_data[:50]}...{b64_data[-50:]}")
                except Exception as e:
                    logger.error(f"Error encoding numpy to base64: {e}")
                    logger.error(traceback.format_exc())
                    return {"comfystream_tensor_output": {"error": f"NumPy base64 encoding error: {str(e)}"}}
                
                shape = list(tensor.shape)
                dtype = str(tensor.dtype)
                
                logger.debug("Constructing success output for numpy array...")
                success_output = {
                    "comfystream_tensor_output": {
                        "b64_data": b64_data,
                        "shape": shape,
                        "dtype": dtype
                    }
                }
                logger.info(f"SendTensorWebSocket returning SUCCESS from numpy array: shape={shape}, dtype={dtype}")
                logger.debug("========== SendTensorWebSocket.save_tensor COMPLETED SUCCESSFULLY ==========")
                return success_output
                
            elif isinstance(tensor, list):
                logger.debug("Processing list input...")
                logger.info(f"Input is a list of length {len(tensor)}")
                
                if len(tensor) > 0:
                    first_item = tensor[0]
                    logger.debug(f"First item type: {type(first_item)}")
                    
                    if isinstance(first_item, torch.Tensor):
                        logger.debug("Processing first tensor from list...")
                        logger.debug(f"First tensor details: shape={first_item.shape}, dtype={first_item.dtype}, device={first_item.device}")
                        
                        # Log first few values
                        logger.debug(f"First few values: {first_item.flatten()[:10].tolist()}")
                        
                        # Process first tensor in the list
                        try:
                            logger.debug("Moving tensor to CPU and detaching...")
                            np_array = first_item.cpu().detach().numpy()
                            logger.debug(f"Conversion successful: shape={np_array.shape}, dtype={np_array.dtype}")
                        except Exception as e:
                            logger.error(f"Error processing first tensor in list: {e}")
                            logger.error(traceback.format_exc())
                            return {"comfystream_tensor_output": {"error": f"List tensor processing error: {str(e)}"}}
                        
                        try:
                            logger.debug("Converting numpy array to bytes...")
                            tensor_bytes = np_array.tobytes()
                            logger.debug(f"Converted to bytes: {len(tensor_bytes)} bytes")
                        except Exception as e:
                            logger.error(f"Error converting list tensor to bytes: {e}")
                            logger.error(traceback.format_exc())
                            return {"comfystream_tensor_output": {"error": f"List tensor bytes conversion error: {str(e)}"}}
                        
                        try:
                            logger.debug("Encoding bytes to base64...")
                            b64_data = base64.b64encode(tensor_bytes).decode('utf-8')
                            b64_size = len(b64_data)
                            logger.debug(f"Base64 encoding successful: {b64_size} characters")
                        except Exception as e:
                            logger.error(f"Error encoding list tensor to base64: {e}")
                            logger.error(traceback.format_exc())
                            return {"comfystream_tensor_output": {"error": f"List tensor base64 encoding error: {str(e)}"}}
                        
                        shape = list(np_array.shape)
                        dtype = str(np_array.dtype)
                        
                        logger.debug("Constructing success output for list tensor...")
                        success_output = {
                            "comfystream_tensor_output": {
                                "b64_data": b64_data,
                                "shape": shape,
                                "dtype": dtype
                            }
                        }
                        logger.info(f"SendTensorWebSocket returning SUCCESS from list's first tensor: shape={shape}, dtype={dtype}")
                        logger.debug("========== SendTensorWebSocket.save_tensor COMPLETED SUCCESSFULLY ==========")
                        return success_output
                    else:
                        logger.error(f"First item in list is not a tensor but {type(first_item)}")
                        if hasattr(first_item, '__dict__'):
                            logger.debug(f"First item attributes: {dir(first_item)}")
                
                # If we got here, couldn't process the list
                logger.error(f"Unable to process list input: invalid content types")
                list_types = [type(x).__name__ for x in tensor[:3]]
                error_msg = f"Unsupported list content: {list_types}..."
                logger.debug("========== SendTensorWebSocket.save_tensor FAILED ==========")
                return {"comfystream_tensor_output": {"error": error_msg}}
            
            else:
                # Unsupported type
                error_msg = f"Unsupported tensor type: {type(tensor)}"
                logger.error(error_msg)
                if hasattr(tensor, '__dict__'):
                    logger.debug(f"Tensor attributes: {dir(tensor)}")
                
                logger.debug("========== SendTensorWebSocket.save_tensor FAILED ==========")
                return {"comfystream_tensor_output": {"error": error_msg}}

        except Exception as e:
            logger.exception(f"Error serializing tensor in SendTensorWebSocket: {e}")
            
            # Get detailed exception info
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = ''.join(tb_lines)
            logger.debug(f"Exception traceback:\n{tb_text}")
            
            error_output = {"comfystream_tensor_output": {"error": f"{str(e)} - See save_tensor_websocket_debug.log for details"}}
            logger.info(f"SendTensorWebSocket returning ERROR data: {error_output}")
            logger.debug("========== SendTensorWebSocket.save_tensor FAILED WITH EXCEPTION ==========")
            return error_output