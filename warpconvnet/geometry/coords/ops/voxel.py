from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


from warpconvnet.geometry.coords.ops.batch_index import (
    batch_index_from_indices,
    batch_index_from_offset,
    batch_indexed_coordinates,
    offsets_from_batch_index,
)

from warpconvnet.utils.unique import unique_hashmap


@torch.no_grad()
def voxel_downsample_random_indices(
    batched_points: Float[Tensor, "N 3"],  # noqa: F821
    offsets: Int[Tensor, "B + 1"],  # noqa: F821
    voxel_size: Optional[float] = None,
) -> Tuple[Int[Tensor, "M"], Int[Tensor, "B + 1"]]:  # noqa: F821
    """
    Args:
        batched points: Float[Tensor, "N 3"] - batched points
        offsets: Int[Tensor, "B + 1"] - offsets for each batch
        voxel_size: Optional[float] - voxel size. Will quantize the points if voxel_size is provided.

    Returns:
        unique_indices: sorted indices of unique voxels.
        batch_offsets: Batch offsets.
    """

    # Floor the points to the voxel coordinates
    N = len(batched_points)
    B = len(offsets) - 1
    device = str(batched_points.device)
    assert (
        offsets[-1] == N
    ), f"Offsets {offsets} does not match the number of points {N}"

    if voxel_size is not None:
        voxel_coords = torch.floor(batched_points / voxel_size).int()
    else:
        voxel_coords = batched_points.int()
    batch_index = batch_index_from_offset(offsets).to(device)
    voxel_coords = torch.cat([batch_index.unsqueeze(1), voxel_coords], dim=1)

    unique_indices, hash_table = unique_hashmap(voxel_coords)
    # unique_indices is sorted

    if B == 1:
        batch_offsets = torch.IntTensor([0, len(unique_indices)])
    else:
        _, batch_counts = torch.unique(batch_index[unique_indices], return_counts=True)
        batch_counts = batch_counts.cpu()
        batch_offsets = torch.cat(
            (batch_counts.new_zeros(1), batch_counts.cumsum(dim=0))
        )

    return unique_indices, batch_offsets
