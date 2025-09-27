import torch
import numpy as np

from queue import Queue
from asyncio import Queue as AsyncQueue

from typing import Union

# TODO: improve eviction policy fifo might not be the best, skip alternate frames instead
# Increased queue size to support batch processing (up to 8 images at a time)
image_inputs: Queue[Union[torch.Tensor, np.ndarray]] = Queue(maxsize=1)
image_outputs: AsyncQueue[Union[torch.Tensor, np.ndarray]] = AsyncQueue()

audio_inputs: Queue[Union[torch.Tensor, np.ndarray]] = Queue()
audio_outputs: AsyncQueue[Union[torch.Tensor, np.ndarray]] = AsyncQueue()

text_outputs: AsyncQueue[str] = AsyncQueue()
