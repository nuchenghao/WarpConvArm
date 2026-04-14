from typing import Optional, Tuple

import torch
from torch import Tensor


import warpconvnet._C as _C
from warpconvnet.utils.type_cast import _min_dtype
from warpconvnet.geometry.coords.search.search_results import IntSearchResult


def _build_pair_table(
    kernel_map: IntSearchResult,
    N_out: int,
    device: torch.device,
) -> Tensor:
    """Build the forward pair_table [K * N_out] from kernel_map."""
    K = len(kernel_map)
    if hasattr(kernel_map, "_pair_table") and kernel_map._pair_table is not None:
        return kernel_map._pair_table.reshape(-1).contiguous()

    pair_table = torch.empty(K * N_out, dtype=torch.int32, device=device)
    pair_table.fill_(-1)
    L = kernel_map.in_maps.shape[0]
    if L > 0 and hasattr(_C.gemm, "csr_to_pair_table_cuda"):
        offsets_gpu = kernel_map.offsets.to(device=device, dtype=torch.int32)
        _C.gemm.csr_to_pair_table_cuda(
            kernel_map.in_maps.int(),
            kernel_map.out_maps.int(),
            offsets_gpu,
            pair_table,
            N_out,
            K,
        )
    return pair_table


def _build_mask_and_argsort(
    pair_table: Tensor,
    N: int,
    K: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Build pair_mask and mask_argsort from a pair_table [K * N]."""
    pair_mask = torch.zeros(N, dtype=torch.int32, device=device)
    if K <= 32 and hasattr(_C.gemm, "build_pair_mask_cuda"):
        _C.gemm.build_pair_mask_cuda(pair_table, pair_mask, K)
    elif K <= 32:
        pair_table_2d = pair_table.reshape(K, N)
        valid = pair_table_2d >= 0
        bit_positions = (
            1 << torch.arange(K, device=device, dtype=torch.int32)
        ).unsqueeze(1)
        pair_mask = (valid.int() * bit_positions).sum(dim=0).int()
    mask_argsort = torch.argsort(pair_mask, stable=True).int()
    return pair_mask, mask_argsort


def _kernel_map_to_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert IntSearchResult to mask-based pair_table + mask + argsort.

    Returns:
        pair_table: [K * N_out] int32, flattened
        pair_mask: [N_out] int32 (uint32 bitmask)
        mask_argsort: [N_out] int32 permutation
    """
    K = len(kernel_map)
    N_out = num_out_coords
    pair_table = _build_pair_table(kernel_map, N_out, device)
    pair_mask, mask_argsort = _build_mask_and_argsort(pair_table, N_out, K, device)
    return pair_table, pair_mask, mask_argsort


def _get_mask_data(
    kernel_map: IntSearchResult,
    num_out_coords: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Get or compute mask data, cached on the kernel_map object."""
    if kernel_map._mask_data is None:
        kernel_map._mask_data = _kernel_map_to_mask_data(
            kernel_map, num_out_coords, device
        )
    return kernel_map._mask_data


def _mask_implicit_gemm_forward_logic(
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype] = None,
    block_size: int = 16,
    mma_tile: int = 3,
) -> Tensor:
    """Forward pass using mask-based fused implicit GEMM."""
    device = in_features.device
    feature_dtype = compute_dtype if compute_dtype is not None else in_features.dtype
    min_dtype = _min_dtype(feature_dtype, weight.dtype)

    _in_features = in_features.contiguous().detach().to(dtype=min_dtype)
    _weight = weight.contiguous().detach().to(dtype=min_dtype)

    N_in, C_in = _in_features.shape
    K, _, C_out = _weight.shape

    output = torch.zeros((num_out_coords, C_out), dtype=min_dtype, device=device)

    if num_out_coords == 0 or K == 0 or C_in == 0 or C_out == 0 or N_in == 0:
        return output.to(dtype=in_features.dtype)

    pair_table, pair_mask, mask_argsort = _get_mask_data(
        kernel_map, num_out_coords, device
    )

    # Auto-pad unaligned channels for CuTe tensor core eligibility
    _has_cute = hasattr(_C.gemm, "cute_gemm_mask_fwd")
    _use_cuda_cute_rules = _has_cute and device.type == "cpu"
    vec_width = 16 // _in_features.element_size()  # 8 for fp16/bf16
    orig_C_in, orig_C_out = C_in, C_out
    needs_padding = (C_in % vec_width != 0) or (C_out % vec_width != 0)
    if (
        needs_padding
        and _use_cuda_cute_rules
        and min_dtype in (torch.float16, torch.bfloat16)
    ):
        target_cin = ((C_in + vec_width - 1) // vec_width) * vec_width
        target_cout = ((C_out + vec_width - 1) // vec_width) * vec_width
        _in_features = torch.nn.functional.pad(_in_features, (0, target_cin - C_in))
        _weight = torch.nn.functional.pad(
            _weight, (0, target_cout - C_out, 0, target_cin - C_in)
        )
        output = torch.zeros(
            (num_out_coords, target_cout), dtype=min_dtype, device=device
        )
        C_in, C_out = target_cin, target_cout
    aligned = True  # After padding, always aligned

    if _has_cute:
        # CuTe kernels require fp16/bf16 — downcast fp32 inputs
        if min_dtype == torch.float32 and _use_cuda_cute_rules:
            _in_features = _in_features.half()
            _weight = _weight.half()
            output = output.half()
            min_dtype = torch.float16
            # Recheck padding after dtype change
            vec_width = 16 // _in_features.element_size()
            needs_padding = (orig_C_in % vec_width != 0) or (
                orig_C_out % vec_width != 0
            )
            if needs_padding:
                C_in_pad = ((orig_C_in + vec_width - 1) // vec_width) * vec_width
                C_out_pad = ((orig_C_out + vec_width - 1) // vec_width) * vec_width
                _in_features = torch.nn.functional.pad(
                    _in_features, (0, C_in_pad - orig_C_in)
                )
                _weight = torch.nn.functional.pad(
                    _weight, (0, C_out_pad - orig_C_out, 0, C_in_pad - orig_C_in)
                )
                output = torch.zeros(
                    (num_out_coords, C_out_pad), dtype=min_dtype, device=device
                )
                C_in, C_out = C_in_pad, C_out_pad
        status = _C.gemm.cute_gemm_mask_fwd(
            _in_features,
            _weight,
            output,
            pair_table,
            pair_mask,
            mask_argsort,
            K,
            mma_tile,
            1.0,
        )
        if status == 0:
            if needs_padding:
                output = output[:, :orig_C_out]
            return output.to(dtype=in_features.dtype)
        raise RuntimeError(
            f"cute_gemm_mask_fwd failed with status {status} "
            f"(N={num_out_coords}, C_in={C_in}, C_out={C_out}, K={K})"
        )

    raise RuntimeError(
        f"mask_implicit_gemm requires CuTe backend (C_in={C_in}, C_out={C_out})"
    )
