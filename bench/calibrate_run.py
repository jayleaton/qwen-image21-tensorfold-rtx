"""Calibrate the cached NVFP4 engine's static input scales from runs/ref/contexts.pt, and re-save it in the cache.

Also copies the contexts to the cache dir so future conversions (any GGUF / precision) calibrate on their own.
"""
import argparse, shutil, sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths
from tfimage import ext, store
from tfimage.calibrate import calibrate
ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="nvfp4")
ap.add_argument("--gguf", default=None, help="default: QWEN_IMAGE21_GGUF, else the one in COMFYUI_PORTABLE")
a = ap.parse_args()
a.gguf = a.gguf or paths.gguf()
ext.use_prebuilt()
ctx_src = "runs/ref/contexts.pt"; shutil.copy(ctx_src, store.cache_dir() / "calib_contexts.pt")
path = store.cache_dir() / (store.key(a.gguf, a.spec) + ".safetensors")
m = store.load_or_convert(a.gguf, a.spec)
t = time.perf_counter(); acts = calibrate(m, torch.load(ctx_src)); print(f"calibrated {len(acts)} linears in {time.perf_counter() - t:.0f}s")
vals = sorted(acts.values()); print("act range", vals[0], vals[len(vals) // 2], vals[-1])
store.save(m, path); print("saved", path)
