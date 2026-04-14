from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.logger import get_logger

from .explicit import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
)
from .mask_gemm import (
    _mask_implicit_gemm_forward_logic,
)


def _execute_forward(
    algo: str,
    params: Dict[str, Any],
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    fwd_block_size: Optional[int],
) -> Tensor:
    """Dispatch forward pass to the selected algorithm."""
    if algo == "explicit_gemm":
        return _explicit_gemm_forward_logic(
            in_features, weight, kernel_map, num_out_coords, compute_dtype
        )
    elif algo == "mask_implicit_gemm":
        return _mask_implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            block_size=params.get("block_size", 16),
            mma_tile=params.get("mma_tile", 3),
        )
    else:
        raise ValueError(f"Unsupported forward algorithm: {algo}")


def _execute_backward(
    algo: str,
    params: Dict[str, Any],
    grad_output: Tensor,
    in_features: Tensor,
    weight: Tensor,
    kernel_map: IntSearchResult,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    device: torch.device,
    needs_input_grad: Tuple[bool, ...],
    weight_T: Optional[Tensor] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Dispatch backward pass to the selected algorithm.

    Args:
        weight_T: Pre-computed weight.transpose(1,2).contiguous() to avoid
            redundant copies when dgrad and wgrad are dispatched separately.

    Returns (grad_in_features, grad_weight). Either can be None if the
    corresponding needs_input_grad flag is False AND the algorithm supports it.
    """
    if algo == "explicit_gemm":
        return _explicit_gemm_backward_logic(
            grad_output, in_features, weight, kernel_map, compute_dtype, device
        )

    else:
        raise ValueError(f"Unsupported backward algorithm: {algo}")
