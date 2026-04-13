from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from jaxtyping import Int
from torch import Tensor

from warpconvnet.geometry.coords.search.torch_hashmap import HashMethod, TorchHashTable


def unique_hashmap(
    bcoords: Int[Tensor, "N 4"],  # noqa: F821
    hash_method: HashMethod = HashMethod.CITY,
) -> Tuple[Int[Tensor, "M"], TorchHashTable]:  # noqa: F821
    """
    Args:
        bcoords: Batched coordinates.
        hash_method: Hash method.

    Returns:
        unique_indices: bcoords[unique_indices] == unique
        hash_table: Hash table.
    """
    # Append batch index to the coordinates
    # assert "cpu" in str(
    #     bcoords.device
    # ), f"Batched coordinates must be on cuda device, got {bcoords.device}"
    table = TorchHashTable(2 * len(bcoords), hash_method)
    table.insert(bcoords)
    return table.unique_index, table  # this is a torch tensor
