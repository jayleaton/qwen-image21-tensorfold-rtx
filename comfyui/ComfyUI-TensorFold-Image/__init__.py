"""ComfyUI custom node: TensorFold image engine (tfimage). Points at the tfimage checkout (TFIMAGE_REPO or ../..)."""

import os
import sys
from pathlib import Path

_repo = Path(os.environ.get("TFIMAGE_REPO") or Path(__file__).resolve().parents[2])
for _p in (_repo, _repo / "vendor" / "TensorFold" / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tfimage.comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
