"""Measured tensor-core ceilings on this GPU for DiT-shaped GEMMs (M=tokens, K/N from Qwen-Image 2.1)."""
import torch, time
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
def bench(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it
M = 4096 + 300
for K, N, name in [(4096, 12288, "qkv"), (4096, 4096, "out"), (4096, 24576, "gate_up"), (12288, 4096, "down")]:
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    t = bench(lambda: a @ w.t()); fl = 2 * M * K * N
    a8 = a.to(torch.float8_e4m3fn); w8 = w.to(torch.float8_e4m3fn); one = torch.ones((), device="cuda")
    t8 = bench(lambda: torch._scaled_mm(a8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
    print(f"{name:8s} M={M} K={K} N={N}  bf16 {t:6.2f} ms {fl/t/1e9:6.1f} TF/s   fp8 {t8:6.2f} ms {fl/t8/1e9:6.1f} TF/s")
