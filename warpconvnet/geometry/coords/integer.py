from typing import List, Tuple, Optional, Union
from jaxtyping import Bool, Float, Int

import torch
from torch import Tensor

from warpconvnet.geometry.base.coords import Coords

from warpconvnet.geometry.coords.search.torch_hashmap import TorchHashTable
from warpconvnet.geometry.utils.list_to_batch import list_to_cat_tensor
from warpconvnet.utils.ntuple import ntuple
from warpconvnet.geometry.coords.ops.batch_index import (
    batch_indexed_coordinates,
)
from warpconvnet.geometry.coords.ops.serialization import POINT_ORDERING, encode


class IntCoords(Coords):
    voxel_size: float
    tensor_stride: Optional[Tuple[int, ...]]
    _hashmap: Optional[TorchHashTable]

    def __init__(
        self,
        batched_tensor: List[Float[Tensor, "N D"]] | Float[Tensor, "N D"],
        offsets: Optional[Union[List[int], Int[Tensor, "B+1"]]] = None,
        voxel_size: Optional[float] = None,
        tensor_stride: Optional[Union[int, Tuple[int, ...]]] = None,
        device: Optional[str] = None,
    ):
        """

        Args:
            batched_tensor: provides the coordinates of the points
            offsets: provides the offsets for each batch
            voxel_size: provides the size of the voxel for converting the coordinates to points
            tensor_stride: provides the stride of the tensor for converting the coordinates to points
        """
        if isinstance(batched_tensor, list):
            assert (
                offsets is None
            ), "If batched_tensors is a list, offsets must be None."
            batched_tensor, offsets, _ = list_to_cat_tensor(batched_tensor)

        if isinstance(offsets, list):
            offsets = torch.LongTensor(offsets, requires_grad=False)

        if device is not None:
            batched_tensor = batched_tensor.to(device)

        self.offsets = offsets.cpu()
        self.batched_tensor = batched_tensor
        self.voxel_size = voxel_size
        # Convert the tensor stride to ntuple
        if tensor_stride is not None:
            self.tensor_stride = ntuple(
                tensor_stride, ndim=self.batched_tensor.shape[1]
            )
        else:
            self.tensor_stride = None

        self.check()

    def check(self):
        Coords.check(self)
        assert self.batched_tensor.dtype in [
            torch.int32,
            torch.int64,
        ], "Discrete coordinates must be integers"
        if self.tensor_stride is not None:
            assert isinstance(self.tensor_stride, (int, tuple))

    @property
    def stride(self):
        return self.tensor_stride

    @property
    def num_spatial_dims(self):
        return self.batched_tensor.shape[1]

    def set_tensor_stride(self, tensor_stride: Union[int, Tuple[int, ...]]):
        self.tensor_stride = ntuple(tensor_stride, ndim=self.num_spatial_dims)

    @property
    def hashmap(self) -> TorchHashTable:
        if not hasattr(self, "_hashmap") or self._hashmap is None:
            bcoords = batch_indexed_coordinates(
                self.batched_tensor, self.offsets
            )  # add additional batch index into the keys
            self._hashmap = TorchHashTable.from_keys(bcoords, device=bcoords.device)
        return self._hashmap

    def sort(self, ordering: POINT_ORDERING = POINT_ORDERING.MORTON_XYZ) -> "IntCoords":
        result = encode(
            self.batched_tensor,
            batch_offsets=self.offsets,
            order=ordering,
            return_perm=True,
        )
        return self.__class__(
            self.batched_tensor[result.perm],
            self.offsets,
            voxel_size=self.voxel_size,
            tensor_stride=self.tensor_stride,
        )
