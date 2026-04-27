from tokenize import group
from typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float

from enum import Enum

import torch
from torch import Tensor
from torch.autograd import Function

import warpconvnet._C as _C
from warpconvnet.geometry.coords.search.search_results import IntSearchResult

# from warpconvnet.utils.benchmark_cache import (
#     SpatiallySparseConvConfig,
# )
from warpconvnet.utils.ntuple import _pad_tuple
from warpconvnet.utils.logger import get_logger

from .dispatch import _execute_forward, _execute_backward
from .algo_params import (
    SPARSE_CONV_AB_ALGO_MODE,
    SPARSE_CONV_ATB_ALGO_MODE,
    _get_adaptive_AB_params,
    _filter_benchmark_params_by_env_config,
)


from .autotune import (
    _run_forward_benchmarks,
)

logger = get_logger(__name__)


class UnifiedSpatiallySparseConvFunction(Function):
    @staticmethod
    def forward(
        ctx,  # The context object of a PyTorch custom Function, used to save data for the backward() pass
        in_features: Float[Tensor, "N C_in"],  # Input features of sparse points/voxels
        weight: Float[Tensor, "K C_in C_out"],  # Standard convolutions typically use the [K, C_in, C_out] format
        kernel_map: IntSearchResult,  # Contains the sparse convolution adjacency relationships, such as in_maps, out_maps, and offsets
        num_out_coords: int,
        fwd_algo: Union[str, List[Union[str, SPARSE_CONV_AB_ALGO_MODE]]],
        dgrad_algo: Union[str, List[Union[str, SPARSE_CONV_AB_ALGO_MODE]]],
        wgrad_algo: Union[str, List[Union[str, SPARSE_CONV_ATB_ALGO_MODE]]],
        compute_dtype: Optional[torch.dtype],  # Computational precision, typically resolved by an external helper under AMP
        voxel_size: Optional[Tuple[int, ...]] = None,  # Passed to the AB adaptive candidate selection for the shape-aware strategy
        groups: int = 1,
        use_fp16_accum: bool = False,
    ) -> Float[Tensor, "M C_out"]:
        output_feature_tensor = None

        # Normalize input algos to strings for benchmarking and caching
        def _to_algo_str_list(
            x: Union[str, List[Union[str, Enum]], Enum],
        ) -> Union[str, List[str]]:
            if isinstance(x, list):
                return [a.value if isinstance(a, Enum) else str(a) for a in x]
            return x.value if isinstance(x, Enum) else str(x)

        fwd_algo = _to_algo_str_list(fwd_algo)
        dgrad_algo = _to_algo_str_list(dgrad_algo)
        wgrad_algo = _to_algo_str_list(wgrad_algo)

        C_in = in_features.shape[1]
        C_out = weight.shape[-1] * groups if groups > 1 else weight.shape[2]
        kv = weight.shape[0]

        if compute_dtype is not None:
            _weight_cast = weight.contiguous().to(dtype=compute_dtype)
        else:
            _weight_cast = weight.contiguous()
        fwd_params = {}

        logger.debug(
            f"[dispatch] FWD forced algo={fwd_algo} requested={fwd_algo} "
            f"N_in={in_features.shape[0]} N_out={num_out_coords} "
            f"C_in={C_in} C_out={C_out} kv={kv} dtype={in_features.dtype}"
        )
        output_feature_tensor = _execute_forward(
            fwd_algo,
            fwd_params,
            in_features,
            _weight_cast,
            kernel_map,
            num_out_coords,
            compute_dtype,
            groups=groups,
            use_fp16_accum=use_fp16_accum,
        )
        # Save only the state needed by the explicit backward path.
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(in_features, weight)
            ctx.kernel_map = kernel_map
            ctx.use_fp16_accum = use_fp16_accum
            ctx.groups = groups
            ctx.config_params_for_bwd = {
                "num_in_coords": in_features.shape[0],
                "num_out_coords": num_out_coords,
                "in_channels": C_in,
                "out_channels": C_out,
                "kernel_volume": kv,
                "compute_dtype": compute_dtype,
                "device": in_features.device,
                "dgrad_algo": dgrad_algo,
                "wgrad_algo": wgrad_algo,
            }

        return output_feature_tensor

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, "M C_out"]) -> Tuple[
        Optional[Float[Tensor, "N C_in"]],
        Optional[Float[Tensor, "K C_in C_out"]],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        kernel_map = getattr(ctx, "kernel_map", None)
        config_params = getattr(ctx, "config_params_for_bwd", None)
        if kernel_map is None or config_params is None or len(ctx.saved_tensors) == 0:
            # Forward ran without grad (e.g., frozen backbone with torch.no_grad())
            return _pad_tuple(None, None, 13)

        in_features, weight = ctx.saved_tensors
        num_out_coords = config_params["num_out_coords"]
        compute_dtype = config_params["compute_dtype"]
        device = config_params["device"]
        dgrad_algo = config_params["dgrad_algo"]
        wgrad_algo = config_params["wgrad_algo"]
        bwd_params = {}

        grad_in_features, grad_weight = None, None

        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return _pad_tuple(None, None, 13)

        use_fp16_accum = getattr(ctx, "use_fp16_accum", False)
        N_in, C_in = in_features.shape
        K = weight.shape[0]
        C_out = weight.shape[2]
        if num_out_coords == 0 or K == 0 or C_in == 0 or C_out == 0 or N_in == 0 or grad_output.shape[0] == 0:
            grad_in_final = torch.zeros_like(in_features) if ctx.needs_input_grad[0] else None
            grad_weight_final = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
            return _pad_tuple(grad_in_final, grad_weight_final, 13)

        if dgrad_algo == wgrad_algo:
            logger.debug(f"[dispatch] BWD forced algo=explicit_gemm params={{}} " f"N_in={N_in} N_out={num_out_coords} C_in={C_in} C_out={C_out} kv={K}")
            grad_in_features, grad_weight = _execute_backward(
                dgrad_algo,
                bwd_params,
                grad_output,
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
                device,
                needs_input_grad=(ctx.needs_input_grad[0], ctx.needs_input_grad[1]),
                use_fp16_accum=use_fp16_accum,
            )
            if not ctx.needs_input_grad[0]:
                grad_in_features = None
            if not ctx.needs_input_grad[1]:
                grad_weight = None

            ctx.kernel_map = None
            ctx.config_params_for_bwd = None

            return _pad_tuple(grad_in_features, grad_weight, 13)
        else:
            pass


# Algorithm execution dispatch moved to dispatch.py
# _execute_forward and _execute_backward are imported from there.
# This comment replaces ~300 lines of dispatch code that was extracted.
_DISPATCH_MOVED = True  # sentinel to confirm extraction
