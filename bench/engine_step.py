"""Standalone engine step timing (no ComfyUI): converted engine from the cache, ref dump's context, 1024^2 latent."""
import argparse, time, torch
from tfimage import ext, store
ap = argparse.ArgumentParser(); ap.add_argument("--spec", default="nvfp4"); ap.add_argument("--hw", type=int, nargs=2, default=[64, 64])
ap.add_argument("--profile", action="store_true"); ap.add_argument("--steps", type=int, default=5)
a = ap.parse_args()
ext.use_prebuilt()
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent)); import paths  # noqa: E401,E402
GGUF = paths.gguf()
t0 = time.perf_counter(); m = store.load_or_convert(GGUF, a.spec); torch.cuda.synchronize(); print(f"load {time.perf_counter() - t0:.1f}s")
ctx_file = store.cache_dir() / "calib_contexts.pt"           # a real prompt embedding if calibration ran
c = (torch.load(ctx_file)[0] if ctx_file.is_file() else torch.randn(1, 40, 4096, dtype=torch.bfloat16)).cuda(); x = torch.randn(1, 64, *a.hw, device="cuda", dtype=torch.bfloat16)
def step(t): return m(x, torch.tensor([t], device="cuda"), c)
step(0.9); torch.cuda.synchronize()
ts = []
for i in range(a.steps):
    t = time.perf_counter(); step(0.5); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
print("step s:", [round(v, 3) for v in ts], "peak alloc GiB", torch.cuda.max_memory_allocated() / 2**30)
if a.profile:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        step(0.5); torch.cuda.synchronize()
    rows = sorted([(e.key, e.device_time_total / 1e3, e.count) for e in prof.key_averages() if e.device_time_total > 0 and not e.key.startswith("aten::")], key=lambda r: -r[1])
    print(f"kernel total {sum(r[1] for r in rows):.1f} ms")
    for k, ms, n in rows[:25]: print(f"{ms:8.2f} ms x{n:<5d} {k[:110]}")
