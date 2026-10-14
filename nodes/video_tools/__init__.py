"""File and frame nodes for batch video tools."""

from .container import VideoContainerEdit
from .frames import LoadVideoFrame, RotateFlipImage, SaveVideoFrame

NODE_CLASS_MAPPINGS = {
    "LoadVideoFrame": LoadVideoFrame,
    "SaveVideoFrame": SaveVideoFrame,
    "RotateFlipImage": RotateFlipImage,
    "VideoContainerEdit": VideoContainerEdit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoFrame": "Load Video Frame",
    "SaveVideoFrame": "Save Video Frame",
    "RotateFlipImage": "Rotate Or Flip Image",
    "VideoContainerEdit": "Video Container Edit",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
