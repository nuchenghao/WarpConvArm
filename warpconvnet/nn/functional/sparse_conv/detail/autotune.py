from typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float

from enum import Enum

import torch
from torch import Tensor
from warpconvnet.utils.logger import get_logger
from warpconvnet.utils.timer import Timer

from .explicit import (
    _explicit_gemm_forward_logic,
)

from .algo_params import (
    _get_filtered_AB_params,
)

logger = get_logger(__name__)
# Benchmark iterations for auto-tuning. More iterations = more reliable
# winner selection but slower first-iteration auto-tune.
_BENCHMARK_NUM_WARMUP = 2
_BENCHMARK_NUM_ITERS = 5

# ---------------------------------------------------------------------------
# Forward benchmark runner
# ---------------------------------------------------------------------------


def _run_forward_benchmarks(
    in_features: Float[Tensor, "N C_in"],
    weight: Float[Tensor, "K C_in C_out"],
    kernel_map,
    num_out_coords: int,
    compute_dtype: Optional[torch.dtype],
    warmup_iters: int = _BENCHMARK_NUM_WARMUP,
    benchmark_iters: int = _BENCHMARK_NUM_ITERS,
    custom_params: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Benchmark different forward algorithms and return sorted results (best first)."""
    warmup_iters = max(warmup_iters, 1)
    benchmark_iters = max(benchmark_iters, 1)

    all_benchmark_results: List[Tuple[str, Dict[str, Any], float]] = []
    timer = Timer()

    def _execute_single_fwd_pass(
        algo_mode: str, params_config: Dict[str, Any]
    ) -> Optional[int]:
        if algo_mode == "explicit_gemm":
            _ = _explicit_gemm_forward_logic(
                in_features,
                weight,
                kernel_map,
                num_out_coords,
                compute_dtype,
            )
        else:
            raise ValueError(
                f"Unsupported algo_mode in _execute_single_fwd_pass: {algo_mode}"
            )

    params_to_use = (
        custom_params if custom_params is not None else _get_filtered_AB_params()
    )
    # Filter out IMPLICIT_GEMM when dtype is float64 (unsupported by kernels)
    dtype_to_check = compute_dtype if compute_dtype is not None else in_features.dtype
    if dtype_to_check == torch.float64:
        params_to_use = [
            (algo, cfg) for (algo, cfg) in params_to_use if algo != "implicit_gemm"
        ]

    # Note: no alignment filter for mask_implicit_gemm — both CUTLASS and mask
    # kernels auto-pad unaligned channels internally (see cutlass.py, mask_gemm.py).
    if False:
        params_to_use = [
            (algo, cfg) for (algo, cfg) in params_to_use if algo != "mask_implicit_gemm"
        ]

    global _AUTOTUNE_BANNER_SHOWN
    num_candidates = len(params_to_use)
    N_in = in_features.shape[0]
    C_in_val = in_features.shape[1]
    C_out_val = weight.shape[2]
    if not _AUTOTUNE_BANNER_SHOWN:
        logger.warning(
            "WarpConvNet: Auto-tuning sparse convolution algorithms. "
            "The first few iterations will be slow while optimal kernels are selected. "
            "Results are cached to ~/.cache/warpconvnet/ for future runs."
        )
        _AUTOTUNE_BANNER_SHOWN = True
    logger.info(
        f"Auto-tuning forward (N={N_in}, C_in={C_in_val}, C_out={C_out_val}, "
        f"{num_candidates} candidates)..."
    )

    for idx, (algo_mode, params_config) in enumerate(params_to_use, 1):
        # Warmup runs
        status = None
        try:
            for _ in range(warmup_iters):
                status = _execute_single_fwd_pass(algo_mode, params_config)
                if isinstance(status, int) and status != 0:
                    break
            # Sync to catch async CUDA errors from this candidate
            # torch.cuda.synchronize()
        except (RuntimeError, Exception) as e:
            logger.debug(
                f"  [{idx}/{num_candidates}] {algo_mode} — skipped (error: {e})"
            )
            # Clear CUDA error state to prevent corruption of subsequent candidates.
            # cudaGetLastError() resets the error flag; synchronize() then succeeds.
            # try:
            #     torch.cuda.synchronize()
            # except Exception:
            #     pass  # Clear error state by consuming the sync exception  # Clear again after sync failure
            continue

        if isinstance(status, int) and status != 0:
            logger.debug(
                f"  [{idx}/{num_candidates}] {algo_mode} — skipped (unsupported)"
            )
            continue

        # Benchmark runs — collect all times and take median for robustness
        iter_times = []

        try:
            for _ in range(benchmark_iters):
                with timer:
                    _execute_single_fwd_pass(algo_mode, params_config)
                iter_times.append(timer.elapsed_time)
            # Sync to catch async errors
            # torch.cuda.synchronize()
        except (RuntimeError, Exception) as e:
            logger.debug(
                f"  [{idx}/{num_candidates}] {algo_mode} — failed during benchmark (error: {e})"
            )
            # try:
            #     torch.cuda.synchronize()
            # except Exception:
            #     pass  # Clear error state by consuming the sync exception
            continue

        if iter_times:
            median_time_ms = sorted(iter_times)[len(iter_times) // 2]
            all_benchmark_results.append((algo_mode, params_config, median_time_ms))
            _param_str = ", ".join(f"{k}={v}" for k, v in params_config.items())
            logger.debug(
                f"  [{idx}/{num_candidates}] {algo_mode}"
                + (f" ({_param_str})" if _param_str else "")
                + f" — {median_time_ms:.2f}ms"
            )

    if not all_benchmark_results:
        logger.warning("No forward benchmark succeeded. Falling back to explicit_gemm.")
        with timer:
            _execute_single_fwd_pass("explicit_gemm", {})
        all_benchmark_results.append(("explicit_gemm", {}, timer.elapsed_time))

    # Sort results by time (3rd element of tuple), ascending
    all_benchmark_results.sort(key=lambda x: x[2])

    best_algo, best_params, overall_best_time_ms = all_benchmark_results[0]
    _best_param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
    logger.info(
        f"Auto-tune forward complete: {best_algo}"
        + (f" ({_best_param_str})" if _best_param_str else "")
        + f" — {overall_best_time_ms:.2f}ms"
    )
    return all_benchmark_results
