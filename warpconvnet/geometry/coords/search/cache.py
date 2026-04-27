from typing import Tuple, Optional
from jaxtyping import Int

from dataclasses import dataclass

from torch import Tensor

from warpconvnet.geometry.coords.search.utils import _int_tensor_hash

from warpconvnet.geometry.coords.search.search_results import (
    IntSearchResult,
)

from warpconvnet.geometry.coords.search.torch_discrete import (
    _int_sequence_hash,
    string_hash,
)


@dataclass
class IntSearchCacheKey:
    """
    It packages the 'configuration of a kernel map search task' into a hashable object to serve as a dictionary key for caching IntSearchResult.
    This is used in conjunction with the IntSearchCache(dict) class.
    """

    kernel_size: Tuple[int, ...]
    kernel_dilation: Tuple[int, ...]
    transposed: bool
    generative: bool
    stride_mode: str
    skip_symmetric_kernel_map: bool
    in_offsets: Int[Tensor, "B+1"]  # noqa: F821
    out_offsets: Int[Tensor, "B+1"]  # noqa: F821

    def __init__(
        self,
        kernel_size,
        kernel_dilation,
        transposed,
        generative,
        stride_mode,
        skip_symmetric_kernel_map,
        in_offsets,
        out_offsets,
    ):
        # Instead of packing the full coordinates into the key, it only includes the 'convolution configuration + batch segmentation structure.' T
        # his represents a 'lightweight key' design
        self.kernel_size = kernel_size
        self.kernel_dilation = kernel_dilation
        self.transposed = transposed
        self.generative = generative
        self.stride_mode = stride_mode
        self.skip_symmetric_kernel_map = skip_symmetric_kernel_map
        self.in_offsets = in_offsets.detach().int()
        self.out_offsets = out_offsets.detach().int()

    def __hash__(self):
        # For a dictionary to perform efficient lookups, its keys must be hashable. The approach here is to mix the hashes of individual fields using a bitwise XOR (^) operation
        return int(
            _int_sequence_hash(self.kernel_size)
            ^ _int_sequence_hash(self.kernel_dilation)
            ^ hash(self.transposed)
            ^ hash(self.generative)
            ^ string_hash(self.stride_mode)  # Use string_hash for stride_mode
            ^ hash(self.skip_symmetric_kernel_map)
            ^ _int_sequence_hash(self.in_offsets.tolist())
            ^ _int_sequence_hash(self.out_offsets.tolist())
        )

    def __eq__(self, other: "IntSearchCacheKey"):
        return (
            self.kernel_size == other.kernel_size
            and self.kernel_dilation == other.kernel_dilation
            and self.transposed == other.transposed
            and self.generative == other.generative
            and self.stride_mode == other.stride_mode
            and self.skip_symmetric_kernel_map == other.skip_symmetric_kernel_map
            and self.in_offsets.equal(other.in_offsets)
            and self.out_offsets.equal(other.out_offsets)
        )

    def __repr__(self):
        return f"IntSearchCacheKey(kernel_size={self.kernel_size}, kernel_dilation={self.kernel_dilation}, transposed={self.transposed}, generative={self.generative}, stride_mode={self.stride_mode}, skip_symmetric_kernel_map={self.skip_symmetric_kernel_map}, num_in={self.in_offsets[-1]}, num_out={self.out_offsets[-1]})"


class IntSearchCache(dict):

    def get(self, key: IntSearchCacheKey) -> Optional[IntSearchResult]:
        return super().get(key, None)

    def put(self, key: IntSearchCacheKey, value: IntSearchResult):
        super().__setitem__(key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)} keys)"
