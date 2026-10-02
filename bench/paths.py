r"""Where things are, from the environment (no machine-specific defaults).

COMFYUI_PORTABLE   the ComfyUI Windows portable folder (holds python_embeded\ and ComfyUI\)
QWEN_IMAGE21_GGUF  the Qwen-Image 2.1 GGUF (default: <ComfyUI>\models\diffusion_models\qwen-image-2.1-Q6_K.gguf)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def portable() -> Path:
    p = os.environ.get("COMFYUI_PORTABLE")
    if not p:
        sys.exit("set COMFYUI_PORTABLE to your ComfyUI portable folder (the one holding python_embeded and ComfyUI)")
    return Path(p)


def comfy() -> Path:
    return portable() / "ComfyUI"


def python() -> Path:
    return portable() / "python_embeded" / "python.exe"


def gguf() -> str:
    return os.environ.get("QWEN_IMAGE21_GGUF") or str(comfy() / "models" / "diffusion_models" / "qwen-image-2.1-Q6_K.gguf")
