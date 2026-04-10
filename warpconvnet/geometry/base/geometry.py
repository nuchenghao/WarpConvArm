from dataclasses import dataclass, field
from typing import Any, Dict, Union, Optional

import torch
from torch import Tensor


def amp_aware_dtype(func):
    """Decorator to handle dtype conversion based on autocast context.

    Usage:
        @amp_aware_dtype
        def features(self) -> Tensor:
            return self.batched_features.batched_tensor
    """

    def wrapper(self, *args, **kwargs):
        tensor = func(self, *args, **kwargs)
        if torch.is_autocast_enabled():
            amp_dtype = torch.get_autocast_gpu_dtype()
            if amp_dtype is not None:
                return tensor.to(dtype=amp_dtype)
        return tensor

    return wrapper


@dataclass
class Geometry:
    """A base class for all geometry objects such as sparse voxels, points, etc.

    This class provides a unified interface for handling different types of geometric data
    with associated features. It supports both concatenated and padded feature representations.

    Args:
        batched_coordinates (Coords): Coordinate information for the geometry.
        batched_features (Union[CatFeatures, PadFeatures, Tensor]): Feature data associated with the coordinates.
        **kwargs: Additional arguments to be stored as extra attributes.

    Properties:
        num_spatial_dims (int): Number of spatial dimensions in the coordinates.
        coordinate_tensor (Tensor): The raw coordinate tensor.
        features (Tensor): The raw feature tensor.
        device: The device where the tensors are stored.
        num_channels (int): Number of feature channels.
        batch_size (int): Size of the batch.
        dtype: Data type of the features.
    """
