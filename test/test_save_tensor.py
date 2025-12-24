import asyncio

import pytest
import torch

from comfystream import tensor_cache
from nodes.tensor_utils.save_tensor import SaveTensor


async def _drain_image_queue():
    while True:
        try:
            tensor_cache.image_outputs.get_nowait()
        except asyncio.QueueEmpty:
            break


@pytest.mark.asyncio
async def test_save_tensor_splits_batched_images():
    await _drain_image_queue()

    images = torch.rand(2, 8, 8, 3)

    SaveTensor().execute(images)

    first = await tensor_cache.image_outputs.get()
    second = await tensor_cache.image_outputs.get()

    assert torch.equal(first, images[0])
    assert torch.equal(second, images[1])
    assert tensor_cache.image_outputs.empty()


@pytest.mark.asyncio
async def test_save_tensor_passthrough_single_image():
    await _drain_image_queue()

    images = torch.rand(1, 4, 4, 3)

    SaveTensor().execute(images)

    queued = await tensor_cache.image_outputs.get()

    assert torch.equal(queued, images)
    assert tensor_cache.image_outputs.empty()

