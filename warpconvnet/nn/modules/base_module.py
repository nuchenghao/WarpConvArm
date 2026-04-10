from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from warpconvnet.geometry.base.geometry import Geometry


class BaseSpatialModule(nn.Module):
    """Base module for spatial features. The input must be an instance of `BatchedSpatialFeatures`."""

    @property
    def device(self):
        """Returns the device that the model is on."""
        return next(self.parameters()).device

    def forward(self, x: Geometry):
        """Forward pass."""
        raise NotImplementedError
