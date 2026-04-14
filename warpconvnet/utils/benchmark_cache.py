import re
import math
import os
import threading
import time
import atexit
import enum
from pathlib import Path
from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    Sequence,
    TypeVar,
    Generic,
    Callable,
    Iterable,
    List,
)
from dataclasses import dataclass

import msgpack
import torch

_SPARSE_CONV_CONFIG_DTYPE_TO_INT = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
    torch.float64: 3,
}

from warpconvnet.utils.logger import get_logger
from warpconvnet.constants import (
    WARPCONVNET_BENCHMARK_CACHE_DIR,
    WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE,
)

from warpconvnet.utils.dist import _get_current_rank, _is_rank_zero

logger = get_logger(__name__, rank_zero_only=False)


def _int_sequence_hash(arr: Sequence[int]) -> int:  # noqa: F821
    """Hash a sequence of ints into a single 32‑bit value."""
    x = hash(arr[0])
    for i in range(1, len(arr)):
        x = (x * 31 + hash(arr[i])) & 0xFFFFFFFF  # Keep it within 32-bit range
    return x


@dataclass
class SpatiallySparseConvConfig:
    log_num_in_coords: int
    log_num_out_coords: int
    in_channels: int
    out_channels: int
    kernel_volume: int
    in_dtype: torch.dtype
    sm_capability: Tuple[int, int]

    def __init__(
        self,
        num_in_coords: int,
        num_out_coords: int,
        in_channels: int,
        out_channels: int,
        kernel_volume: int,
        in_dtype: torch.dtype,
    ):
        # Use clamped log10 bins: ceil(log10(N)) clamped to [3, inf).
        # This treats all N < 10K identically (bin 3-4), which covers the
        # deep encoder/decoder levels where N varies most across batches.
        # Bins: N<10K → 4, N=10K-100K → 5, N=100K-1M → 6
        _log10_in = math.ceil(math.log10(max(num_in_coords, 1)))
        _log10_out = math.ceil(math.log10(max(num_out_coords, 1)))
        self.log_num_in_coords = max(_log10_in, 4)  # clamp: treat N<10K as one bin
        self.log_num_out_coords = max(_log10_out, 4)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_volume = kernel_volume
        assert (
            in_dtype in _SPARSE_CONV_CONFIG_DTYPE_TO_INT
        ), f"Unsupported in_dtype: {in_dtype}"
        self.in_dtype = in_dtype
        # self.sm_capability = _get_sm_capability()

    def __hash__(self):
        return _int_sequence_hash(
            [
                self.log_num_in_coords,
                self.log_num_out_coords,
                self.in_channels,
                self.out_channels,
                self.kernel_volume,
                _SPARSE_CONV_CONFIG_DTYPE_TO_INT[self.in_dtype],
                self.sm_capability[0] * 10 + self.sm_capability[1],
            ]
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SpatiallySparseConvConfig):
            return False
        return (
            self.log_num_in_coords == other.log_num_in_coords
            and self.log_num_out_coords == other.log_num_out_coords
            and self.in_channels == other.in_channels
            and self.out_channels == other.out_channels
            and self.kernel_volume == other.kernel_volume
            and self.in_dtype == other.in_dtype
            and self.sm_capability == other.sm_capability
        )


class GenericBenchmarkCache(Generic[K, V]):
    """
    A generic on-disk benchmark cache that supports arbitrary namespaces and key/value types.

    - Thread-safe updates with a background saver on rank 0
    - Atomic writes using a temporary file rename
    - File format kept separate from sparse conv cache: benchmark_cache_generic.pkl

    Stored file schema (v3.0):
        {
            "namespaces": { str: { K: V, ... }, ... },
            "timestamp": float,
            "version": WARPCONVNET_BENCHMARK_CACHE_VERSION
        }
    """

    def __init__(self, cache_dir: str = WARPCONVNET_BENCHMARK_CACHE_DIR):
        # Use override cache directory if available (for debugging multi-GPU issues)
        if WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE:
            cache_dir = WARPCONVNET_BENCHMARK_CACHE_DIR_OVERRIDE
            logger.debug(f"Using override cache directory: {cache_dir}")

        self.cache_dir = Path(cache_dir).expanduser().resolve()  # Resolve symlinks
        self.cache_file = self.cache_dir / "benchmark_cache_generic.msgpack"
        self.lock = threading.Lock()

        current_rank = _get_current_rank()

        # In-memory accumulated results to be flushed by background saver
        self._results: Dict[str, Dict[K, V]] = {}
        # Optional namespace -> validator function(value) that raises on invalid
        self._validators: Dict[str, Callable[[V], None]] = {}
        # Callbacks invoked after disk merge so consumers can refresh their in-memory state.
        # Signature: callback(namespace: str, merged_dict: Dict[K, V])
        self._on_merge_callbacks: List[Callable[[str, Dict], None]] = []

        # Periodic save settings
        self.save_interval = 60.0
        self.last_save_time = 0.0
        self.pending_changes = False
        self._shutdown_requested = False

        # Background thread
        self._save_thread = None
        self._save_condition = threading.Condition(self.lock)

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Preload from disk (single load, also used for init logging)
        try:
            self._results = self.load_cache()
            total_entries = sum(len(ns_dict) for ns_dict in self._results.values())
            logger.info(
                f"[Rank {current_rank}] Loaded benchmark cache: "
                f"{len(self._results)} namespaces, {total_entries} entries from {self.cache_file}"
            )
        except Exception as e:
            logger.info(f"[Rank {current_rank}] Failed to load cache: {e}")
            self._results = {}
        if not self._results and not self.cache_file.exists():
            logger.info(
                f"[Rank {current_rank}] No existing cache found, will create: {self.cache_file}"
            )

        # All ranks run background saver — each rank auto-tunes independently
        # and writes results to disk with file locking. On save, the full disk
        # cache is read back into memory so all ranks benefit from each other's
        # auto-tune results.
        self._start_background_saver()
        atexit.register(self._save_on_exit)
        logger.debug(f"[Rank {current_rank}] Started background saver")


# Global generic cache instance
_generic_benchmark_cache: Optional[GenericBenchmarkCache[Any, Any]] = None


def get_generic_benchmark_cache() -> GenericBenchmarkCache[Any, Any]:
    global _generic_benchmark_cache
    if _generic_benchmark_cache is None:
        _generic_benchmark_cache = GenericBenchmarkCache()
    return _generic_benchmark_cache


def generic_benchmark_update_entry(
    namespace: str, key: Any, value: Any, force: bool = False
) -> None:
    cache = get_generic_benchmark_cache()
    cache.update_entry(namespace, key, value, force=force)
