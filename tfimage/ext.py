"""TensorFold's CUDA extensions without a compiler: import a prebuilt module when one is there, else build it.

TensorFold builds its kernels with torch's JIT ``load`` (ninja + nvcc + MSVC). ComfyUI's portable Python has none of
those, but it runs the same torch (2.10+cu130, cp313), so modules built once in this repo's venv import as they are.
``use_prebuilt`` wraps ``tensorfold.cuda.build.load``: a module ``<dir>/<name>/<name>.pyd`` that exists is imported
directly; anything missing falls through to TensorFold's own build.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[1] / ".build" / "torch_ext"
SUFFIX = ".pyd" if os.name == "nt" else ".so"


def _import(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    import torch  # noqa: F401  (its DLLs, cudart64_13 included, must be loaded first)

    loader = importlib.machinery.ExtensionFileLoader(name, str(path))
    spec = importlib.util.spec_from_file_location(name, str(path), loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def use_prebuilt(directory: str | Path | None = None) -> Path:
    """Route TensorFold's extension loads to prebuilt modules under ``directory`` (default: this repo's build dir)."""

    root = Path(directory or os.environ.get("TFIMAGE_EXT_DIR") or DEFAULT_DIR)
    from tensorfold.cuda import build

    if getattr(build.load, "_tfimage", False):
        return root
    original = build.load

    def load(name, sources, *args, **kwargs):
        built = root / name / f"{name}{SUFFIX}"
        if built.is_file():
            return _import(name, built)
        kwargs.setdefault("build_directory", None)
        if kwargs["build_directory"] is None:
            (root / name).mkdir(parents=True, exist_ok=True)
            kwargs["build_directory"] = str(root / name)
        return original(name, sources, *args, **kwargs)

    load._tfimage = True
    build.load = load
    # modules that bound ``load`` at import time
    for mod in list(sys.modules.values()):
        if getattr(mod, "load", None) is original and getattr(mod, "__name__", "").startswith("tensorfold."):
            mod.load = load
    return root
