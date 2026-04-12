import os

import torch

_SKIP_EXTENSION = os.environ.get("WARPCONVNET_SKIP_EXTENSION", "0") == "1"

if not _SKIP_EXTENSION:
    try:
        from . import _C
    except ImportError as exc:
        raise ImportError(
            "Failed to import the compiled WarpConvNet extension. Build it via "
            "`python setup.py build_ext --inplace` or install the pre-built wheel."
        ) from exc
else:
    _C = None  # type: ignore[assignment]
