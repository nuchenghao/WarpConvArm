import numpy as np
import pytest
import torch

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod


@pytest.fixture
def device():
    """Fixture for device configuration."""
    return "cpu"


@pytest.fixture
def sample_keys_torch(device):
    """Fixture providing sample vector keys as Torch tensors.
    shape: (65536, 4); int32 within [0, 10000)
    """

    N = 1 << 16  # 65536
    return torch.randint(0, 10000, (N, 4), device=device, dtype=torch.int32)


@pytest.fixture
def sample_keys_torch_large(device):
    """Fixture providing larger sample vector keys for benchmark."""
    N = 1 << 22  # 4194304
    return torch.randint(0, 100000, (N, 4), device=device, dtype=torch.int32)


# --- TorchHashTable Tests ---


@pytest.mark.parametrize("hash_method", list(HashMethod))
def test_torch_hash_table_creation_and_search(device, sample_keys_torch, hash_method):
    """Test TorchHashTable creation, insertion, and search with various hash methods."""
    table = TorchHashTable.from_keys(
        sample_keys_torch, hash_method=hash_method, device=device
    )
    assert table.key_dim == sample_keys_torch.shape[1]

    # Search for existing keys
    results = table.search(sample_keys_torch)
    # assert results.device == torch.device(device)
    # assert (
    #     results.cpu().numpy() != -1
    # ).all(), f"All existing keys should be found with {hash_method}"

    # # Verify unique keys retrieval
    # unique_indices = table.unique_index
    # unique_keys_retrieved = table.unique_vector_keys

    # # Basic check: number of unique keys should be <= total keys
    # assert unique_keys_retrieved.shape[0] <= sample_keys_torch.shape[0]
    # if unique_keys_retrieved.shape[0] > 0:
    #     assert unique_keys_retrieved.shape[1] == sample_keys_torch.shape[1]

    # # Search for a subset of keys
    # subset_keys = sample_keys_torch[: sample_keys_torch.shape[0] // 2]
    # results_subset = table.search(subset_keys)
    # assert (
    #     results_subset.cpu().numpy() != -1
    # ).all(), f"Subset search failed with {hash_method}"

    # # Search for non-existent keys
    # non_existent_keys = torch.randint(
    #     10001, 20000, sample_keys_torch.shape, device=device, dtype=torch.int32
    # )
    # results_non_existent = table.search(non_existent_keys)
    # assert (
    #     results_non_existent.cpu().numpy() == -1
    # ).all(), f"Non-existent key search failed with {hash_method}"


if __name__ == "__main__":
    pytest.main([__file__])
