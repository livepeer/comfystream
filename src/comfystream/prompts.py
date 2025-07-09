

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

INVERTED_PROMPT = """
{
  "1": {
    "inputs": {
      "images": [
        "3",
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
  },
  "3": {
    "inputs": {
      "image": [
        "2",
        0
      ]
    },
    "class_type": "ImageInvert",
    "_meta": {
      "title": "Invert Image"
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
      "fullgraph": false,
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
      "fullgraph": true,
      "mode": "reduce-overhead",
      "compile_encoder": true,
      "compile_decoder": true,
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
      "use_feature_injection": false,
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