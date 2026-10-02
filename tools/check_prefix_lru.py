"""Prefix LRU: alternating prompts of different lengths and a reused one must not error and must reuse."""
import torch
from tfimage import ext, store
ext.use_prebuilt()
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench")); import paths  # noqa: E401,E402
m = store.load_or_convert(paths.gguf(), "nvfp4")
ctx = torch.load("runs/ref/contexts.pt")
x = torch.randn(1, 64, 32, 32, device="cuda", dtype=torch.bfloat16); t = torch.tensor([0.5], device="cuda")
outs = [m(x, t, ctx[i].cuda()) for i in (0, 1, 0, 2, 3, 4, 5, 1)]
print("prefixes kept", len(m.prefixes), "repeat equal", torch.equal(outs[0], outs[2]))
