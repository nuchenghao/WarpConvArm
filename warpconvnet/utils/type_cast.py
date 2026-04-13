from typing import Optional

import torch


def _maybe_cast(
    tensor: torch.Tensor, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Fast dtype conversion if needed."""
    return tensor if dtype is None else tensor.to(dtype=dtype)
