import torch
import numpy as np

class SideData:
    pass

class FrameProxy:
    def __init__(self, tensor, width, height, pts=None, time_base=None):
        self.width = width
        self.height = height
        self.pts = pts
        self.time_base = time_base
        self.side_data = SideData()
        self.side_data.input = tensor.clone().cpu()
        self.side_data.skipped = True

    @staticmethod
    def avframe_to_frameproxy(frame):
        frame_np = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
        tensor = torch.from_numpy(frame_np).unsqueeze(0)
        return FrameProxy(
            tensor=tensor.clone().cpu(),
            width=frame.width,
            height=frame.height,
            pts=getattr(frame, "pts", None),
            time_base=getattr(frame, "time_base", None)
        )