import torch
import torch.nn as nn


class Sequential(nn.Sequential, BaseSpatialModule):
    """
    Sequential module that allows for spatial and non-spatial layers to be chained together.

    If the module has multiple consecutive non-spatial layers, then it will not create an intermediate
    spatial features object and will become more efficient.
    """
