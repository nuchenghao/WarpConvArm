import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.ravel import ravel_multi_index
from warpconvnet.geometry.coords.search.torch_discrete import (
    generate_kernel_map,
    kernel_offsets_from_size,
    _kernel_map_from_offsets,
    _kernel_map_from_size,
)
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.ops.batch_index import batch_indexed_coordinates


@pytest.fixture
def device():
    """Fixture for device configuration."""
    return "cpu"


def test_kernel_map_from_offset(setup_voxels, device):
    """Test kernel map generation using offset method."""
    voxels: Voxels = setup_voxels.to(device)

    kernel_offsets = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=torch.int32,
        device=device,
    )

    bcoords = batch_indexed_coordinates(voxels.coordinate_tensor, voxels.offsets)
    voxel_hashmap = (
        voxels.coordinate_hashmap
    )  # Batch information was already included when constructing the hash table.

    kernel_map: IntSearchResult = _kernel_map_from_offsets(
        voxel_hashmap,
        bcoords,
        kernel_offsets,
    )

    tot_num_maps = kernel_map.offsets[-1].item()
    assert tot_num_maps == len(kernel_map.in_maps)
    assert tot_num_maps == len(kernel_map.out_maps)
