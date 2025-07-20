"""Video stream utility nodes for ComfyStream"""

from .primary_input_load_image import PrimaryInputLoadImage, NODE_DISPLAY_NAME_MAPPINGS as primary_input_display_mappings

NODE_CLASS_MAPPINGS = {"PrimaryInputLoadImage": PrimaryInputLoadImage}

# Combine display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(primary_input_display_mappings)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
