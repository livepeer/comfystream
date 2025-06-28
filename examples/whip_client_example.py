#!/usr/bin/env python3
"""
Simple WHIP client example for ComfyStream.

This example demonstrates how to use the WHIP (WebRTC-HTTP Ingestion Protocol)
endpoint to ingest media streams to ComfyStream using standard HTTP requests.
"""

import asyncio
import json
import logging
import requests
from typing import Optional
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WHIPClient:
    """Simple WHIP client implementation."""
    
    def __init__(self, whip_url: str, prompts=None):
        self.whip_url = whip_url
        self.prompts = prompts or []
        self.pc = None
        self.resource_url = None
        
    async def publish(self, media_source: Optional[str] = None):
        """Publish media to WHIP endpoint."""
        try:
            # Create peer connection
            self.pc = RTCPeerConnection()
            
            # Add media tracks if source provided
            if media_source is not None:
                player = MediaPlayer(media_source)
                if hasattr(player, 'video') and player.video:
                    self.pc.addTrack(player.video)
                    logger.info("Added video track")
                if hasattr(player, 'audio') and player.audio:
                    self.pc.addTrack(player.audio)
                    logger.info("Added audio track")
            
            # Create offer
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
            
            # Prepare WHIP request
            headers = {'Content-Type': 'application/sdp'}
            
            # Add query parameters for configuration
            params = {}
            if self.prompts:
                params['prompts'] = json.dumps(self.prompts)
            
            # Send WHIP request
            logger.info(f"Sending WHIP request to {self.whip_url}")
            response = requests.post(
                self.whip_url,
                data=self.pc.localDescription.sdp,
                headers=headers,
                params=params
            )
            
            if response.status_code == 201:
                # Success - get resource URL and SDP answer
                self.resource_url = response.headers.get('Location')
                answer_sdp = response.text
                
                logger.info(f"WHIP session created: {self.resource_url}")
                
                # Set remote description
                answer = RTCSessionDescription(sdp=answer_sdp, type='answer')
                await self.pc.setRemoteDescription(answer)
                
                # Log ICE server configuration if provided
                if 'Link' in response.headers:
                    logger.info(f"ICE servers provided: {response.headers['Link']}")
                
                return True
            else:
                logger.error(f"WHIP request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during WHIP publish: {e}")
            return False
    
    async def unpublish(self):
        """Terminate WHIP session."""
        if self.resource_url:
            try:
                logger.info(f"Terminating WHIP session: {self.resource_url}")
                response = requests.delete(self.resource_url)
                if response.status_code == 200:
                    logger.info("WHIP session terminated successfully")
                else:
                    logger.warning(f"WHIP termination returned: {response.status_code}")
            except Exception as e:
                logger.error(f"Error terminating WHIP session: {e}")
        
        if self.pc:
            await self.pc.close()
            logger.info("Peer connection closed")


async def main():
    """Main example function."""
    # Configure WHIP endpoint
    whip_url = "http://localhost:8889/whip"

    # Example ComfyUI prompts
    example_prompts = [{
        "1": {
            "inputs": {
            "image": "example.png"
            },
            "class_type": "LoadImage",
            "_meta": {
            "title": "Load Image"
            }
        },
        "2": {
            "inputs": {
            "engine": "depth_anything_vitl14-fp16.engine",
            "images": [
                "1",
                0
            ]
            },
            "class_type": "DepthAnythingTensorrt",
            "_meta": {
            "title": "Depth Anything Tensorrt"
            }
        },
        "3": {
            "inputs": {
            "unet_name": "static-dreamshaper8_SD15_$stat-b-1-h-512-w-512_00001_.engine",
            "model_type": "SD15"
            },
            "class_type": "TensorRTLoader",
            "_meta": {
            "title": "TensorRT Loader"
            }
        },
        "5": {
            "inputs": {
            "text": "the hulk",
            "clip": [
                "23",
                0
            ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
            "title": "CLIP Text Encode (Prompt)"
            }
        },
        "6": {
            "inputs": {
            "text": "",
            "clip": [
                "23",
                0
            ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
            "title": "CLIP Text Encode (Prompt)"
            }
        },
        "7": {
            "inputs": {
            "seed": 785664736216738,
            "steps": 1,
            "cfg": 1,
            "sampler_name": "lcm",
            "scheduler": "normal",
            "denoise": 1,
            "model": [
                "24",
                0
            ],
            "positive": [
                "9",
                0
            ],
            "negative": [
                "9",
                1
            ],
            "latent_image": [
                "16",
                0
            ]
            },
            "class_type": "KSampler",
            "_meta": {
            "title": "KSampler"
            }
        },
        "8": {
            "inputs": {
            "control_net_name": "control_v11f1p_sd15_depth_fp16.safetensors"
            },
            "class_type": "ControlNetLoader",
            "_meta": {
            "title": "Load ControlNet Model"
            }
        },
        "9": {
            "inputs": {
            "strength": 1,
            "start_percent": 0,
            "end_percent": 1,
            "positive": [
                "5",
                0
            ],
            "negative": [
                "6",
                0
            ],
            "control_net": [
                "10",
                0
            ],
            "image": [
                "2",
                0
            ]
            },
            "class_type": "ControlNetApplyAdvanced",
            "_meta": {
            "title": "Apply ControlNet"
            }
        },
        "10": {
            "inputs": {
            "backend": "inductor",
            "fullgraph": False,
            "mode": "reduce-overhead",
            "controlnet": [
                "8",
                0
            ]
            },
            "class_type": "TorchCompileLoadControlNet",
            "_meta": {
            "title": "TorchCompileLoadControlNet"
            }
        },
        "11": {
            "inputs": {
            "vae_name": "taesd"
            },
            "class_type": "VAELoader",
            "_meta": {
            "title": "Load VAE"
            }
        },
        "13": {
            "inputs": {
            "backend": "inductor",
            "fullgraph": True,
            "mode": "reduce-overhead",
            "compile_encoder": True,
            "compile_decoder": True,
            "vae": [
                "11",
                0
            ]
            },
            "class_type": "TorchCompileLoadVAE",
            "_meta": {
            "title": "TorchCompileLoadVAE"
            }
        },
        "14": {
            "inputs": {
            "samples": [
                "7",
                0
            ],
            "vae": [
                "13",
                0
            ]
            },
            "class_type": "VAEDecode",
            "_meta": {
            "title": "VAE Decode"
            }
        },
        "15": {
            "inputs": {
            "images": [
                "14",
                0
            ]
            },
            "class_type": "PreviewImage",
            "_meta": {
            "title": "Preview Image"
            }
        },
        "16": {
            "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {
            "title": "Empty Latent Image"
            }
        },
        "23": {
            "inputs": {
            "clip_name": "CLIPText/model.fp16.safetensors",
            "type": "stable_diffusion",
            "device": "default"
            },
            "class_type": "CLIPLoader",
            "_meta": {
            "title": "Load CLIP"
            }
        },
        "24": {
            "inputs": {
            "use_feature_injection": False,
            "feature_injection_strength": 0.8,
            "feature_similarity_threshold": 0.98,
            "feature_cache_interval": 4,
            "feature_bank_max_frames": 4,
            "model": [
                "3",
                0
            ]
            },
            "class_type": "FeatureBankAttentionProcessor",
            "_meta": {
            "title": "Feature Bank Attention Processor"
            }
        }
    }]

    # Create WHIP client
    client = WHIPClient(whip_url, example_prompts)

    try:
        # Option 1: Publish webcam (requires camera)
        # success = await client.publish("0")  # Camera device 0

        # Option 2: Publish test pattern
        # success = await client.publish("testsrc")  # Synthetic test source

        # Option 3: Publish file
        # success = await client.publish("path/to/video.mp4")

        # For this example, just test the signaling without media
        success = await client.publish()

        if success:
            logger.info("WHIP publishing started successfully!")

            # Keep session active for 30 seconds
            logger.info("Keeping session active for 30 seconds...")
            await asyncio.sleep(30)
        else:
            logger.error("Failed to start WHIP publishing")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        await client.unpublish()


if __name__ == "__main__":
    print("ComfyStream WHIP Client Example")
    print("===============================")
    print()
    print("This example demonstrates how to use WHIP to ingest streams to ComfyStream.")
    print("Make sure ComfyStream server is running at http://localhost:8889")
    print()
    
    asyncio.run(main()) 
