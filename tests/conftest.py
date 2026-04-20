import pytest
import torch
from warpconvnet.geometry.types.voxels import Voxels


@pytest.fixture
def device():
    """Fixture for device configuration."""
    return "cpu"


def _create_voxels_data(
    B: int,
    min_N: int,
    max_N: int,
    C: int,
    device: str = "cpu",
    voxel_size: float = 0.01,
):
    """Helper function to create voxels data with given parameters."""
    torch.manual_seed(0)
    Ns = torch.randint(min_N, max_N, (B,))
    coords = [(torch.rand((int(N.item()), 3)) / voxel_size).int() for N in Ns]
    features = [torch.rand((int(N.item()), C)) for N in Ns]
    return Voxels(coords, features, device=device).unique()


@pytest.fixture
def setup_voxels(device):
    """Setup medium test voxels with random coordinates and features."""
    return _create_voxels_data(B=30, min_N=1000000, max_N=10000000, C=7, device=device)
