import hashlib
import logging
import math
import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Dict

import numpy as np
import torch

from jaxtyping import Int
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable, HashMethod
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.ntuple import ntuple

logger = logging.getLogger(__name__)


@torch.no_grad()
def kernel_offsets_from_size(
    kernel_size: Tuple[int, ...],
    kernel_dilation: Tuple[int, ...],
    center_offset: Optional[Tuple[int, ...]] = None,
    device: Optional[torch.device] = None,  # Added device argument
) -> Int[Tensor, "K D+1"]:
    """
    Generate the offset coordinates of each kernel position relative to the center point, based on kernel_size and kernel_dilation
    What is returned is not the actual point coordinates, but an offset table.
    Later, during sparse convolution search, the system will query the input-coordinate hash table using “a given output coordinate plus these offsets” to find its neighbors

    For example:
        kernel_size = (3, 3)
        kernel_dilation = (1, 1)                                kernel_dilation = (2, 2)
        center_offset = None

        The return value with batch dimension:
        tensor([                                                tensor([
            [0, -1, -1],                                            [0, -2, -2],
            [0, -1,  0],                                            [0, -2,  0],
            [0, -1,  1],                                            [0, -2,  2],
            [0,  0, -1],                                            [0,  0, -2],
            [0,  0,  0],                                            [0,  0,  0],
            [0,  0,  1],                                            [0,  0,  2],
            [0,  1, -1],                                            [0,  2, -2],
            [0,  1,  0],                                            [0,  2,  0],
            [0,  1,  1],                                            [0,  2,  2],
        ], dtype=torch.int32)                                   ], dtype=torch.int32)
    """

    assert len(kernel_size) == len(kernel_dilation)
    num_spatial_dims = len(kernel_size)  # Record the number of spatial dimensions: 2 for 2D and 3 for 3D

    # Create meshgrid for arbitrary dimensions
    ranges = [torch.arange(size, dtype=torch.int32, device="cpu") for size in kernel_size]
    grids = torch.meshgrid(*ranges, indexing="ij")
    flattened_grids = [grid.flatten() for grid in grids]

    if center_offset is None:
        # center odd-sized kernels and 0 for even-sized kernels
        center_offset = [(s - 1) // 2 if s % 2 == 1 else 0 for s in kernel_size]
    assert len(center_offset) == num_spatial_dims

    # Create offsets for each dimension
    offsets = [(grid - center_offset[i]) * kernel_dilation[i] for i, grid in enumerate(flattened_grids)]

    # Add batch dimension (zeros)
    offsets = [torch.zeros_like(offsets[0])] + offsets

    return torch.stack(offsets, dim=1).contiguous().to(device)


@torch.no_grad()
def _kernel_map_search_to_result(
    found_in_coord_index: Int[Tensor, "K M"],
    identity_map_index: Optional[int] = None,
    return_type: Literal["indices", "offsets"] = "offsets",
) -> Int[Tensor, "K M"] | IntSearchResult:
    """
    Processes the raw found_in_coord_index tensor into the desired format.

    The found_in_coord_index is a tensor of shape (K, M) where K is the number of kernel offsets and M is the number of query coordinates.
    The value is the index of the input coordinate if the kernel offset is found in the query coordinate, otherwise -1.
    We remove the -1 values and return valid indices for each kernel offset.

    The return value:
        in_maps: All input indices that resulted in a hit.
        out_maps: The corresponding output or query indices.
        offsets: Defines the segment within these two arrays belonging to each kernel offset, [ offset[i], offset[i+1] ) belonging to kernel_offsets[i]

    Example:
        found_in_coord_index =
            tensor([
                [10, -1,  4, -1, -1,  8],
                [-1,  3, -1,  6,  9, -1],
            ], dtype=torch.int32)

        in_maps  = [10, 4, 8, 3, 6, 9]
        out_maps = [ 0, 2, 5, 1, 3, 4]
        offsets  = [ 0, 3, 6]


    """

    target_device = found_in_coord_index.device
    K, M = found_in_coord_index.shape  # K is the number of offsets, and M is the number of queries

    if return_type == "indices":
        return found_in_coord_index

    assert return_type == "offsets"  # Default path. It compresses the dense [K, M] matrix into valid mappings grouped by kernel offset.

    # Convert the original matrix into a boolean mask, where 'hit' positions are set to True and positions with -1 are set to False
    found_in_coord_index_bool = found_in_coord_index >= 0  # identify which positions are valid hits

    # get the index of the non zero elements
    # For each row corresponding to a kernel offset, compute the cumulative count of valid hits from left to right.
    # Then, subtract one so that the first hit is indexed as 0, the second as 1, and so on.
    # For the example, `mapped_indices` is
    # tensor(
    #     [
    #         [0, 0, 1, 1, 1, 2],
    #         [-1, 0, 0, 1, 2, 2],
    #     ],
    #     dtype=torch.int32,
    # )
    mapped_indices = torch.cumsum(found_in_coord_index_bool.to(torch.int32), dim=1, dtype=torch.int32) - 1
    # Need to handle rows with zero valid maps correctly (cumsum results in -1)
    # Clamp minimum value to 0 after subtracting 1
    mapped_indices = torch.clamp(mapped_indices, min=-1)  # Keep -1 for rows with no hits

    # Count valid maps per kernel offset row
    # If mapped_indices is -1 everywhere in a row, max will be -1, add 1 -> 0 count.
    num_valid_maps = mapped_indices.max(dim=1).values + 1

    # Calculate offsets
    offsets = torch.cumsum(num_valid_maps, dim=0, dtype=torch.int32)  # Compute the prefix sum of the hit counts for each row
    # Prepend a zero to the beginning. This way, offsets[i] and offsets[i+1] can represent the start and end indices of the $i$-th offset within the flattened result.
    # For the example, offsets is tensor([0, 3, 6], dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32, device=target_device), offsets], dim=0)
    # Retrieve the last offset value—which represents the total number of valid mappings—and convert the single-element tensor into a Python integer
    num_total_maps = offsets[-1].item()

    # Allocate output tensors
    in_maps = torch.empty(num_total_maps, dtype=torch.int32, device=target_device)
    out_maps = torch.empty(num_total_maps, dtype=torch.int32, device=target_device)

    if num_total_maps > 0:
        # Launch CUDA kernel to gather results
        found_in_coord_index_cont = found_in_coord_index.contiguous()
        mapped_indices_cont = mapped_indices.contiguous()
        offsets_cont = offsets.contiguous()

        # Organize the previously prepared 'intermediate results' into the final sparse mapping arrays, in_maps and out_maps
        _C.coords.map_found_indices_to_inoutmaps(
            found_in_coord_index_cont,
            mapped_indices_cont,
            offsets_cont,
            in_maps,
            out_maps,
            K,
            M,
        )

    return IntSearchResult(
        in_maps,
        out_maps,
        offsets,
        identity_map_index=identity_map_index,
    )


@torch.no_grad()
def _kernel_map_from_offsets(
    hashtable: TorchHashTable,  # Use TorchHashTable
    batched_query_coords: Int[Tensor, "N D_1"],
    kernel_offsets: Int[Tensor, "K D_1"],
    identity_map_index: Optional[int] = None,
    return_type: Literal["indices", "offsets"] = "offsets",
) -> Int[Tensor, "K N"] | IntSearchResult:
    """
    Based on a set of query coordinates (batched_query_coords) and kernel offsets (kernel_offsets),
    look up the input points corresponding to each query point under each offset in the hashtable.
    This process constructs the kernel map required for sparse convolution.

    For each output point and each kernel offset, search the input coordinate set for a corresponding input point.
    If a match is found, record the connection between the input index and the output index.
    """

    target_device = hashtable.device
    assert target_device == batched_query_coords.device, f"{target_device} != {batched_query_coords.device}"
    assert target_device == kernel_offsets.device, f"{target_device} != {kernel_offsets.device}"
    assert batched_query_coords.shape[1] == kernel_offsets.shape[1]
    assert batched_query_coords.ndim == 2
    assert kernel_offsets.ndim == 2
    assert batched_query_coords.dtype == torch.int32
    assert kernel_offsets.dtype == torch.int32

    if hashtable._table_kvs is None or hashtable._vector_keys is None:
        raise RuntimeError("Input TorchHashTable must be populated before calling kernel map functions.")

    if identity_map_index is not None:
        # Assert that the number of elements in the hashtable and the query coordinates are the same
        assert identity_map_index < kernel_offsets.shape[0], "Identity map index must be less than the number of kernel offsets"
        iden_offset = kernel_offsets[identity_map_index]
        # assert that iden_offset is all zeros
        assert torch.all(iden_offset == 0), "Identity map offset must be all zeros"

    num_query_coords = batched_query_coords.shape[0]
    key_dim = batched_query_coords.shape[1]
    num_kernel_offsets = kernel_offsets.shape[0]

    # Allocate a raw search result tensor with a shape of [K, N].
    # found_in_coord_index[k, n] represents the index of the input point hit by the k-th kernel offset for the n-th query/output point.
    # If there is no hit, it is -1.
    found_in_coord_index = torch.empty(
        (num_kernel_offsets, num_query_coords),
        dtype=torch.int32,
        device=target_device,
    )

    # Launch the kernel
    # torch::Tensor is not the actual data buffer; it is an object handle that points to an underlying storage.
    _C.coords.kernel_map_offset(
        hashtable._table_kvs.contiguous(),
        hashtable.vector_keys.contiguous(),
        batched_query_coords.contiguous(),
        kernel_offsets.contiguous(),
        found_in_coord_index,
        num_query_coords,
        key_dim,
        num_kernel_offsets,
        hashtable.capacity,
        hashtable.hash_method.value,
    )

    # Reformat the raw matching matrix `found_in_coord_index` obtained from the previous search into a result format better suited for subsequent sparse convolution operations
    # If return_type == indices, directly return the found_in_coord_index[K, N] tensor
    # If return_type == "offsets", compress the result into an IntSearchResult
    return _kernel_map_search_to_result(
        found_in_coord_index,
        identity_map_index=identity_map_index,
        return_type=return_type,
    )


@torch.no_grad()
def _kernel_map_from_size(
    hashtable: TorchHashTable,  # Use TorchHashTable
    batched_query_coords: Int[Tensor, "N D_1"],
    kernel_sizes: Tuple[int, ...],
    identity_map_index: Optional[int] = None,
    return_type: Literal["indices", "offsets"] = "offsets",
    threads_per_block_x: int = 64,
    threads_per_block_y: int = 8,
    skip_symmetric_kernel_map: bool = False,
) -> Int[Tensor, "K N"] | IntSearchResult:
    """ """
    target_device = hashtable.device
    assert str(target_device) == str(batched_query_coords.device)
    assert batched_query_coords.dtype == torch.int32

    if hashtable._table_kvs is None or hashtable._vector_keys is None:
        raise RuntimeError("Input TorchHashTable must be populated before calling kernel map functions.")

    num_dims = batched_query_coords.shape[1]
    assert len(kernel_sizes) == num_dims - 1, f"kernel_size ({len(kernel_sizes)}) must match spatial dims ({num_dims - 1})"

    # Check if we should skip symmetric kernel parts
    if skip_symmetric_kernel_map:
        assert all(k % 2 == 1 for k in kernel_sizes), f"Kernel sizes must be odd for symmetric skipping. Got {kernel_sizes}"

    num_offsets = np.prod(kernel_sizes).item()

    # --- Specialized 4D Case ---
    # 4 does not mean four spatial dimensions; it means D + 1 == 4, i.e., the most common 3D spatial coordinate format [batch, x, y, z].
    if num_dims == 4:
        num_query_coords = batched_query_coords.shape[0]

        if skip_symmetric_kernel_map:
            # For symmetric kernels, only use the first half (excluding center)
            num_offsets = num_offsets // 2
            # Identity map is the center of the kernel
            if identity_map_index is not None:
                assert identity_map_index == num_offsets

        # Prepare kernel size tensor
        kernel_size_tensor = torch.tensor(kernel_sizes, dtype=torch.int32, device=target_device)

        if return_type == "indices":
            # For "indices" return type, use the original search kernel
            found_in_coord_index = torch.empty(
                (num_offsets, num_query_coords),
                dtype=torch.int32,
                device=target_device,
            )
            _C.coords.kernel_map_size_4d(
                hashtable._table_kvs.contiguous(),
                hashtable.vector_keys.contiguous(),
                batched_query_coords.contiguous(),
                kernel_size_tensor,
                found_in_coord_index,
                num_query_coords,
                hashtable.capacity,
                num_offsets,
                hashtable.hash_method.value,
                threads_per_block_x,
                threads_per_block_y,
            )
            return found_in_coord_index

        # Fused kernel map path: 2 CUDA kernel launches (count + scatter)
        # instead of search + postprocess_count + cumsum + postprocess_scatter.
        # No K*M intermediate matrix is allocated.
        if hasattr(_C.coords, "fused_kernel_map") and not skip_symmetric_kernel_map:
            in_maps, out_maps, offsets = _C.coords.fused_kernel_map(
                batched_query_coords.contiguous(),
                hashtable._table_kvs.contiguous(),
                hashtable.vector_keys.contiguous(),
                hashtable.capacity,
                list(kernel_sizes),
                hashtable.hash_method.value,
            )
            return IntSearchResult(
                in_maps,
                out_maps,
                offsets,
                identity_map_index=identity_map_index,
            )

        # Fallback: Search-once + fused postprocess path for "offsets" return type.
        # Single hash table search pass -> postprocess count -> cumsum -> postprocess scatter.
        # This eliminates the second hash table search pass (the dominant cost).
        table_kvs = hashtable._table_kvs.contiguous()
        vector_keys = hashtable.vector_keys.contiguous()
        query_coords = batched_query_coords.contiguous()
        capacity = hashtable.capacity
        hash_method_val = hashtable.hash_method.value

        # Step 1: Single search pass -> K*M intermediate
        found_in_coord_index = torch.empty(
            (num_offsets, num_query_coords),
            dtype=torch.int32,
            device=target_device,
        )
        _C.coords.kernel_map_size_4d(
            table_kvs,
            vector_keys,
            query_coords,
            kernel_size_tensor,
            found_in_coord_index,
            num_query_coords,
            capacity,
            num_offsets,
            hash_method_val,
            threads_per_block_x,
            threads_per_block_y,
        )

        # Step 2: Fused count on intermediate (sequential scan, no hash access)
        counts = torch.zeros(num_offsets, dtype=torch.int32, device=target_device)
        _C.coords.postprocess_count(
            found_in_coord_index,
            counts,
            num_offsets,
            num_query_coords,
        )

        # Step 3: Prefix sum + allocate
        offsets = torch.zeros(num_offsets + 1, dtype=torch.int32, device=target_device)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        num_total_maps = offsets[-1].item()

        # Allocate output
        in_maps = torch.empty(num_total_maps, dtype=torch.int32, device=target_device)
        out_maps = torch.empty(num_total_maps, dtype=torch.int32, device=target_device)

        if num_total_maps > 0:
            # Step 4: Fused scatter on intermediate (sequential scan, no hash access)
            scatter_counters = torch.zeros(num_offsets, dtype=torch.int32, device=target_device)
            _C.coords.postprocess_scatter(
                found_in_coord_index,
                offsets,
                scatter_counters,
                in_maps,
                out_maps,
                num_offsets,
                num_query_coords,
            )

        result = IntSearchResult(
            in_maps,
            out_maps,
            offsets,
            identity_map_index=identity_map_index,
        )
        # Cache the pair_table (found_in_coord_index) for mask-based GEMM.
        # This avoids expensive Python reconstruction in _kernel_map_to_mask_data.
        result._pair_table = found_in_coord_index  # [K, N_out], int32, -1=invalid
        return result

    # --- Generic Case (Fallback to offset method) ---
    else:
        # kernel_offsets_tensor is a local offset template relative to each query coordinate
        # In this context, `batched_query_coords` usually refers to the output coordinates.
        kernel_offsets_tensor = kernel_offsets_from_size(kernel_sizes, (1,) * len(kernel_sizes), device=target_device)

        # If skipping symmetric parts, reduce the kernel offsets
        if skip_symmetric_kernel_map:
            num_offsets = num_offsets // 2
            kernel_offsets_tensor = kernel_offsets_tensor[:num_offsets]

        # Call the offset-based function
        return _kernel_map_from_offsets(
            hashtable,
            batched_query_coords,
            kernel_offsets_tensor,
            return_type=return_type,
            identity_map_index=identity_map_index,
        )


@torch.no_grad()
def generate_kernel_map(
    batch_indexed_in_coords: Int[Tensor, "N D_1"],
    batch_indexed_out_coords: Int[Tensor, "M D_1"],
    in_to_out_stride_ratio: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    kernel_dilation: Optional[Tuple[int, ...]] = None,
    kernel_center_offset: Optional[Tuple[int, ...]] = None,
    method: Literal["offset", "size"] = "size",  # Size is the default fastest method
    hash_method: HashMethod = HashMethod.CITY,  # Allow selecting hash method
    skip_symmetric_kernel_map: bool = False,
) -> IntSearchResult:
    """
    Given the input active points `batch_indexed_in_coords` and output active points `batch_indexed_out_coords`, construct the `kernel_map` (connection table) required for sparse convolution.
    This connection table informs the subsequent convolution operators: for each output point and each specific kernel offset, which corresponding input point was hit
    """

    target_device = batch_indexed_in_coords.device
    assert target_device == batch_indexed_out_coords.device
    assert batch_indexed_in_coords.dtype == torch.int32
    assert batch_indexed_out_coords.dtype == torch.int32
    if skip_symmetric_kernel_map:
        assert len(batch_indexed_in_coords) == len(
            batch_indexed_out_coords
        ), "You can only skip symmetric kernel map if the input and output coordinates are the same."
        assert all(k % 2 == 1 for k in kernel_size), "Kernel size must be odd for symmetric skipping."

    # Create a TorchHashTable for the input coordinates
    hashtable = TorchHashTable.from_keys(batch_indexed_in_coords, hash_method=hash_method, device=target_device)

    # Subtract the batch column from the total number of columns to obtain the number of spatial dimensions
    num_spatial_dims = batch_indexed_out_coords.shape[1] - 1
    assert len(in_to_out_stride_ratio) == num_spatial_dims

    # If the input and output are not on the same coordinate scale, perform scale alignment first
    if not all(s == 1 for s in in_to_out_stride_ratio):
        # Construct a scaling vector in the form of [1, sx, sy, sz]. The leading 1 is for the batch column, ensuring that the batch IDs remain unscaled
        stride_tensor = torch.tensor(
            [1] + list(ntuple(in_to_out_stride_ratio, ndim=num_spatial_dims)),
            dtype=torch.int32,
            device=target_device,
        )
        # Scale the output coordinates back to the input coordinate system.
        # For example, with a stride of 2, an output point at (b, 5, 7, 9) would be mapped to (b, 10, 14, 18) in the input grid.
        strided_out_coords = batch_indexed_out_coords * stride_tensor
    else:
        strided_out_coords = batch_indexed_out_coords

    # assume there is no 'identity mapping for the center offset'
    identity_map_index = None
    # Check if kernel is odd and potentially symmetric
    is_odd_kernel = all(k % 2 == 1 for k in kernel_size)
    # Use `the same quantity` as an approximate signal to infer that this may be a same-coordinate-set scenario
    same_in_out_coords = batch_indexed_in_coords.shape[0] == batch_indexed_out_coords.shape[0]
    # If there appears to be a center point and the input and output may use the same coordinate set, record the index of the center offset.
    if is_odd_kernel and same_in_out_coords:
        total_kernels = int(np.prod(kernel_size))  # Compute the total number of kernel positions
        center_idx = total_kernels // 2  # For kernel offsets flattened in rule-defined order, the middle position corresponds to the center point.
        identity_map_index = center_idx  # Record the identity map corresponding to the center offset. Some subsequent optimizations will use it.

    # Force the symmetric kernel skipping to be False if the kernel is not odd
    if skip_symmetric_kernel_map and not is_odd_kernel:
        skip_symmetric_kernel_map = False

    if method == "offset":
        # This method generates offsets and launches the custom kernel_map_offset kernel
        if kernel_dilation is None:
            kernel_dilation = (1,) * num_spatial_dims

        kernel_offsets_tensor = kernel_offsets_from_size(
            kernel_size,
            kernel_dilation,
            center_offset=kernel_center_offset,
            device=target_device,
        )
        if identity_map_index is not None:
            kernel_offsets_tensor = kernel_offsets_tensor[:center_idx]

        return _kernel_map_from_offsets(
            hashtable,
            strided_out_coords,  # Use strided coordinates
            kernel_offsets_tensor,
            return_type="offsets",
            identity_map_index=identity_map_index,
        )
    elif method == "size":
        # This method uses _kernel_map_from_size, which has the 4D specialization
        assert kernel_dilation is None or all(
            s == 1 for s in kernel_dilation
        ), "Kernel dilation is not supported with method='size'. Use method='offset' instead."
        assert kernel_center_offset is None, "Custom kernel_center_offset is not supported with method='size'. Use method='offset' instead."
        return _kernel_map_from_size(
            hashtable,
            strided_out_coords,
            kernel_size,
            return_type="offsets",
            skip_symmetric_kernel_map=skip_symmetric_kernel_map,
            identity_map_index=identity_map_index,
        )
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'offset', or 'size'.")


def _int_sequence_hash(arr: Sequence[int]) -> int:  # noqa: F821
    x = hash(arr[0])
    for i in range(1, len(arr)):
        x = (x * 31 + hash(arr[i])) & 0xFFFFFFFF  # Keep it within 32-bit range
    return x


def string_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) & 0xFFFFFFFF
