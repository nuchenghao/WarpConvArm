from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult
from warpconvnet.utils.logger import get_logger

from .explicit import (
    _explicit_gemm_forward_logic,
    _explicit_gemm_backward_logic,
    _explicit_gemm_forward_grouped,
    _explicit_gemm_backward_grouped,
)
from .implicit_direct import (
    _implicit_gemm_forward_logic,
    _implicit_gemm_backward_logic,
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
    groups: int = 1,
    use_fp16_accum: bool = False,
) -> Tensor:
    """Dispatch forward pass to the selected algorithm."""
    # if groups > 1 and algo != "production":
    #     raise ValueError(f"Group convolution (groups={groups}) only supported with algo='production', " f"got '{algo}'")
    if groups > 1:
        C_in_g = weight.shape[2]
        C_out_g = weight.shape[3]
        if C_in_g < 8 or C_out_g < 8:
            raise ValueError(f"Group convolution requires per-group channels >= 8 " f"(got C_in/G={C_in_g}, C_out/G={C_out_g}). " f"Reduce groups or increase channels.")
    if algo == "explicit_gemm":
        return _explicit_gemm_forward_logic(in_features, weight, kernel_map, num_out_coords, compute_dtype)
    elif algo == "implicit_gemm":
        return _implicit_gemm_forward_logic(in_features, weight, kernel_map, num_out_coords, compute_dtype)
    elif algo == "explicit_gemm_grouped":
        return _explicit_gemm_forward_grouped(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
            saturation_m=params.get("saturation_m", 5000),
        )
    elif algo == "mask_implicit_gemm":
        return _mask_implicit_gemm_forward_logic(
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            compute_dtype,
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
    use_fp16_accum: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Dispatch backward pass to the selected algorithm.

    Args:
        weight_T: Pre-computed weight.transpose(1,2).contiguous() to avoid
            redundant copies when dgrad and wgrad are dispatched separately.

    Returns (grad_in_features, grad_weight). Either can be None if the
    corresponding needs_input_grad flag is False AND the algorithm supports it.
    """
    if algo == "explicit_gemm":
        return _explicit_gemm_backward_logic(grad_output, in_features, weight, kernel_map, compute_dtype, device)
    elif algo == "implicit_gemm":
        return _implicit_gemm_backward_logic(
            grad_output,
            in_features,
            weight,
            kernel_map,
            num_out_coords,
            split_k_factor=params.get("split_k_factor", 4),
            compute_dtype=compute_dtype,
        )
    elif algo == "explicit_gemm_grouped":
        return _explicit_gemm_backward_grouped(
            grad_output,
            in_features,
            weight,
            kernel_map,
            compute_dtype,
            device,
            saturation_m=params.get("saturation_m", 5000),
        )
    else:
        raise ValueError(f"Unsupported backward algorithm: {algo}")
