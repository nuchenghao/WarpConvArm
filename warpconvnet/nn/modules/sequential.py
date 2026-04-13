import torch
import torch.nn as nn
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.modules.base_module import BaseSpatialModule


def run_forward(module: nn.Module, x: Geometry, in_sf: Geometry):
    if isinstance(module, BaseSpatialModule) and isinstance(x, Geometry):
        return module(x), in_sf
    elif not isinstance(module, BaseSpatialModule) and isinstance(x, Geometry):
        in_sf = x
        x = module(x.feature_tensor)
    elif isinstance(x, torch.Tensor) and isinstance(module, BaseSpatialModule):
        x = in_sf.replace(batched_features=x)
        x = module(x)
    else:
        x = module(x)

    return x, in_sf


class Sequential(nn.Sequential, BaseSpatialModule):
    """
    Sequential module that allows for spatial and non-spatial layers to be chained together.

    If the module has multiple consecutive non-spatial layers, then it will not create an intermediate
    spatial features object and will become more efficient.
    """

    def forward(self, x: Geometry):
        assert isinstance(
            x, Geometry
        ), f"Expected BatchedSpatialFeatures, got {type(x)}"

        in_sf = x
        for module in self:
            x, in_sf = run_forward(module, x, in_sf)

        if isinstance(x, torch.Tensor):
            x = in_sf.replace(batched_features=x)

        return x
