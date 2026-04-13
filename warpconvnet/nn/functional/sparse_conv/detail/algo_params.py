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
