"""DiT-shaped GEMMs: cuBLAS bf16 / torch fp8 vs TensorFold NVFP4 W4A4 and FP8 prompt GEMMs (incl. row quantization)."""
import torch
from tensorfold.cuda.nvfp4 import checkpoint as ck
from tensorfold.cuda.nvfp4.linear import Fp4Linear

def bench(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it

def nvfp4_weight(w):
    """bf16 [N, K] -> (codes [N, K/2], e4m3 scales [N, K/16], global) via TF's own row quantizer."""
    n, k = w.shape
    g = float(w.abs().amax()) / (6 * 448)
    r = ck.quant4(w, g)
    s = r.scales[:, :n, :].permute(1, 0, 2).reshape(n, k // 16)   # [K/64, mpad, 4] -> [N, K/16]
    return r.codes, s, g

def deq(codes, s, g):
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=codes.device)
    c = torch.stack([codes & 15, codes >> 4], -1).flatten(1).long()
    return lut[c] * s.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1) * g

M = 4096 + 64
torch.manual_seed(0)
for K, N, name in [(4096, 12288, "qkv"), (4096, 4096, "out"), (4096, 24576, "gate_up"), (12288, 4096, "down")]:
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    fl = 2 * M * K * N
    tb = bench(lambda: x @ w.t())
    codes, s, g = nvfp4_weight(w)
    wd = deq(codes, s, g)
    print(f"  weight nvfp4 rel err {((wd - w.float()).norm() / w.float().norm()).item():.4f}")
    act = float(x.abs().amax()) / (6 * 448)
    lin = Fp4Linear.from_checkpoint(codes, s.view(torch.float8_e4m3fn), g, act=act)
    y4 = ck.prompt(ck.A4, x, lin, tile=0)
    tiles = {}
    for tile in [0, 1, 2, 3, 11, 12, 13, 14]:
        try: tiles[tile] = bench(lambda: ck.prompt(ck.A4, x, lin, tile=tile))
        except Exception as e: tiles[tile] = None
    best = min((t, k) for k, t in tiles.items() if t)
    ref = x.float() @ w.float().t()
    err4 = ((ck.prompt(ck.A4, x, lin, tile=best[1]).float() - ref).norm() / ref.norm()).item()
    rows = ck.quant4(x, act, ck._tb(best[1]))
    tq = bench(lambda: ck.quant4(x, act, ck._tb(best[1])))
    print(f"{name:8s} K={K:5d} N={N:5d} bf16 {tb:6.2f} ms {fl/tb/1e9:5.0f} TF/s | nvfp4 W4A4 best tile {best[1]} {best[0]:6.2f} ms "
          f"{fl/best[0]/1e9:5.0f} TF/s (quant {tq:.2f} ms) relerr {err4:.4f} | tiles {{{', '.join(f'{k}:{v:.2f}' for k, v in tiles.items() if v)}}}")
