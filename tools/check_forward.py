"""tfimage Qwen-Image 2.1 forward vs ComfyUI's on the dumped inputs (runs/ref/forward.pt), per backend."""
import sys, time, torch
from tfimage.qwen_image21 import QwenImage21
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench")); import paths  # noqa: E401,E402
GGUF = paths.gguf()
ref = torch.load("runs/ref/forward.pt")
def cmp(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return f"relL2 {((a - b).norm() / b.norm()).item():.4f} cos {torch.nn.functional.cosine_similarity(a, b, dim=0).item():.5f}"
for kind in sys.argv[1:] or ["bf16", "fp8", "nvfp4"]:
    t0 = time.perf_counter(); m = QwenImage21.from_gguf(GGUF, kind); torch.cuda.synchronize()
    print(f"[{kind}] load {time.perf_counter() - t0:.1f}s, linears {m.nbytes() / 2**30:.2f} GiB, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    x, t, c = ref["x"].cuda(), ref["t"].cuda(), ref["context"].cuda()
    out = m(x, t, c).cpu()
    print(f"[{kind}] t2i  {cmp(out, ref['out'])}")
    out_r = m(x, t, c, [ref["ref"].cuda()], [ref["slot"]]).cpu()
    print(f"[{kind}] edit {cmp(out_r, ref['out_ref'])}   (edit vs t2i ref out: {cmp(ref['out_ref'], ref['out'])})")
    del m; torch.cuda.empty_cache()
