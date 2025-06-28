import copy

from typing import Dict, Any
from comfy.api.components.schema.prompt import Prompt, PromptDictInput


def create_load_tensor_node():
    return {
        "inputs": {},
        "class_type": "LoadTensor",
        "_meta": {"title": "LoadTensor"},
    }


def create_save_tensor_node(inputs: Dict[Any, Any]):
    return {
        "inputs": inputs,
        "class_type": "SaveTensor",
        "_meta": {"title": "SaveTensor"},
    }
DEFAULT_PROMPT = """
{
  "1": {
    "inputs": {
      "images": [
        "2",
        0
      ]
    },
    "class_type": "SaveTensor",
    "_meta": {
      "title": "SaveTensor"
    }
  },
  "2": {
    "inputs": {},
    "class_type": "LoadTensor",
    "_meta": {
      "title": "LoadTensor"
    }
  }
}
"""
DEFAULT_SD_PROMPT = """
{
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
    }
"""


def convert_prompt(prompt: PromptDictInput) -> Prompt:
    # Validate the schema
    Prompt.validate(prompt)

    prompt = copy.deepcopy(prompt)

    num_primary_inputs = 0
    num_inputs = 0
    num_outputs = 0

    keys = {
        "PrimaryInputLoadImage": [],
        "LoadImage": [],
        "PreviewImage": [],
        "SaveImage": [],
    }
    
    for key, node in prompt.items():
        class_type = node.get("class_type")

        # Collect keys for nodes that might need to be replaced
        if class_type in keys:
            keys[class_type].append(key)

        # Count inputs and outputs
        if class_type == "PrimaryInputLoadImage":
            num_primary_inputs += 1
        elif class_type in ["LoadImage", "LoadTensor", "LoadAudioTensor"]:
            num_inputs += 1
        elif class_type in ["PreviewImage", "SaveImage", "SaveTensor", "SaveAudioTensor"]:
            num_outputs += 1

    # Only handle single primary input
    if num_primary_inputs > 1:
        raise Exception("too many primary inputs in prompt")

    # If there are no primary inputs, only handle single input
    if num_primary_inputs == 0 and num_inputs > 1:
        raise Exception("too many inputs in prompt")

    # Only handle single output for now
    if num_outputs > 1:
        raise Exception("too many outputs in prompt")

    if num_primary_inputs + num_inputs == 0:
        raise Exception("missing input")

    if num_outputs == 0:
        raise Exception("missing output")

    # Replace nodes
    for key in keys["PrimaryInputLoadImage"]:
        prompt[key] = create_load_tensor_node()

    if num_primary_inputs == 0 and len(keys["LoadImage"]) == 1:
        prompt[keys["LoadImage"][0]] = create_load_tensor_node()

    for key in keys["PreviewImage"] + keys["SaveImage"]:
        node = prompt[key]
        prompt[key] = create_save_tensor_node(node["inputs"])

    # Validate the processed prompt input
    prompt = Prompt.validate(prompt)

    return prompt
