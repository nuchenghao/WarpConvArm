from typing import Any, Dict, List, Tuple, Union

from enum import Enum


class SPARSE_CONV_AB_ALGO_MODE(Enum):
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    CUTLASS_IMPLICIT_GEMM = "cutlass_implicit_gemm"
    CUTE_IMPLICIT_GEMM = "cute_implicit_gemm"
    EXPLICIT_GEMM_GROUPED = "explicit_gemm_grouped"
    IMPLICIT_GEMM_GROUPED = "implicit_gemm_grouped"
    CUTLASS_GROUPED_HYBRID = "cutlass_grouped_hybrid"
    CUTE_GROUPED = "cute_grouped"
    CUTE_IMPLICIT_GEMM_SM90 = "cute_implicit_gemm_sm90"
    CUTE_GROUPED_SM90 = "cute_grouped_sm90"
    MASK_IMPLICIT_GEMM = "mask_implicit_gemm"
    AUTO = "auto"  # Benchmark and select the best algorithm
    ALL = "all"  # Benchmark ALL candidates (slow, exhaustive)
    TRIMMED = "trimmed"  # Benchmark reduced set (excludes dead-weight)


class SPARSE_CONV_ATB_ALGO_MODE(Enum):
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    CUTLASS_IMPLICIT_GEMM = "cutlass_implicit_gemm"
    CUTE_IMPLICIT_GEMM = "cute_implicit_gemm"
    EXPLICIT_GEMM_GROUPED = "explicit_gemm_grouped"
    IMPLICIT_GEMM_GROUPED = "implicit_gemm_grouped"
    CUTLASS_GROUPED_HYBRID = "cutlass_grouped_hybrid"
    CUTE_GROUPED = "cute_grouped"
    CUTE_IMPLICIT_GEMM_SM90 = "cute_implicit_gemm_sm90"
    CUTE_GROUPED_SM90 = "cute_grouped_sm90"
    MASK_IMPLICIT_GEMM = "mask_implicit_gemm"
    AUTO = "auto"  # Benchmark and select the best algorithm
    ALL = "all"  # Benchmark ALL candidates (slow, exhaustive)
    TRIMMED = "trimmed"  # Benchmark reduced set (excludes dead-weight)


def _filter_benchmark_params_by_env_config(
    all_params: List[Tuple[Union[str, Any], Dict[str, Any]]],
    env_config: Union[str, List[Union[str, Any]]],
    is_forward: bool = True,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Filter benchmark parameters based on environment variable configuration.

    Args:
        all_params: All available benchmark parameters (the reduced "auto" set)
        env_config: Environment variable value (string or list of algorithm names)
        is_forward: Whether this is for AB pass (affects which exhaustive set to use)

    Returns:
        Filtered list of benchmark parameters
    """
    if env_config == "all":
        # When "all", use the full exhaustive candidate set (nothing excluded)
        full_params = _ALL_AB_PARAMS if is_forward else _ALL_ATB_PARAMS
        return [(str(algo), params) for algo, params in full_params]

    if env_config in ("auto", "trimmed"):
        # "auto" and "trimmed" both use dimension-aware candidate selection.
        # The caller (unified.py) already selected the right params via
        # _get_adaptive_*_params or _get_trimmed_*_params.
        return [(str(algo), params) for algo, params in all_params]

    # Convert environment config to list of algorithm names
    if isinstance(env_config, str):
        target_algos = [env_config]
    else:
        target_algos = [str(a) for a in env_config]

    if not target_algos:
        logger.warning("No valid algorithms found, using all algorithms")
        return all_params

    # Filter parameters to only include target algorithms
    filtered_params: List[Tuple[str, Dict[str, Any]]] = []
    for algo_tag, params in all_params:
        algo_str = str(algo_tag)
        if algo_str in target_algos:
            filtered_params.append((algo_str, params))

    if not filtered_params:
        logger.warning(f"No benchmark parameters found for algorithms {target_algos}, using all algorithms")
        return all_params

    return filtered_params


def _get_filtered_AB_params() -> List[Tuple[str, Dict[str, Any]]]:
    """Get AB benchmark parameters filtered by environment variable.

    For "auto", returns the static superset of all adaptive candidates.
    For "all", returns the full exhaustive set.
    """
    return _filter_benchmark_params_by_env_config(_AB_PARAMS_AUTO, WARPCONVNET_AB_ALGO_MODE, is_forward=True)


import math as _math


def _get_adaptive_AB_params(
    in_channels: int,
    out_channels: int,
    kernel_volume: int,
    num_in_coords: int = 0,
    voxel_size: Union[Tuple[int, ...], None] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get AB (gather-scatter) auto candidates — most aggressive trimming.

    Based on 301-config analysis (SM 8.9, cuBLAS 12.9.1.4):
      mask: 66% wins (dominates ch<=128 at all N, ch<=256 at small-medium N)
      cutlass: 21% (dominates ch>256 at large N)
      cutlass_grouped: 10% (wins ch 129-256 at large N)
      cute_grouped: 2% (marginal, dropped from auto)

    Auto picks only the dominant winner per region. 4-5 candidates.
    """
    max_ch = max(in_channels, out_channels)
    log_n = _math.ceil(_math.log2(num_in_coords)) if num_in_coords > 1 else 0

    if kernel_volume >= 64:
        params: List[Tuple[str, Dict[str, Any]]] = []
        params.extend(_AB_MASK_IMPLICIT_GEMM)
        # params.extend(_AB_CUTLASS_IMPLICIT)
        # params.extend(_AB_CUTE_SM90)
        # params.extend(_AB_CUTE_GROUPED_SM90)
        return params

    # ch <= 128: mask wins 90-100% at all N sizes
    # Include cutlass as fallback in case mask fails (e.g., unsupported config)
    if max_ch <= 128:
        params = []
        params.extend(_AB_MASK_IMPLICIT_GEMM)
        # params.extend(_AB_CUTLASS_IMPLICIT)
        # params.extend(_AB_CUTE_SM90)
        # params.extend(_AB_CUTE_GROUPED_SM90)
        return params

    # ch 129-256: mask dominates small/medium N, cutlass_grouped at large N
    if max_ch <= 256:
        params = []
        params.extend(_AB_MASK_IMPLICIT_GEMM)
        if log_n == 0 or log_n > 17:
            params.extend([("cutlass_grouped_hybrid", {"saturation_m": 5000})])
        # params.extend(_AB_CUTLASS_IMPLICIT)
        # params.extend(_AB_CUTE_SM90)
        # params.extend(_AB_CUTE_GROUPED_SM90)
        return params

    # ch > 256: cute_grouped wins at ch>=384, cutlass at ch=512 large N,
    # mask still competitive at ch=256-384 small N
    params = []
    params.extend(_AB_MASK_IMPLICIT_GEMM)
    # params.extend(_AB_CUTE_GROUPED)
    # params.extend(_AB_CUTLASS_IMPLICIT)
    # params.extend(_AB_CUTE_SM90)
    # params.extend(_AB_CUTE_GROUPED_SM90)
    return params
