import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
B,H,N,P,D=1,32,4096,66,128
def t(e):
    s=torch.cuda.Event(True);f=torch.cuda.Event(True)
    for _ in range(2): e()
    torch.cuda.synchronize(); s.record()
    for _ in range(5): e()
    f.record(); torch.cuda.synchronize(); return s.elapsed_time(f)/5
for layout in ("BNHD-view","BHND-contig"):
    if layout=="BNHD-view":
        q=torch.randn(B,N,H,D,device="cuda",dtype=torch.bfloat16).transpose(1,2); k=torch.randn(B,N+P,H,D,device="cuda",dtype=torch.bfloat16).transpose(1,2); v=torch.randn_like(k)
    else:
        q=torch.randn(B,H,N,D,device="cuda",dtype=torch.bfloat16); k=torch.randn(B,H,N+P,D,device="cuda",dtype=torch.bfloat16); v=torch.randn_like(k)
    for be in (SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel([be]): ms=t(lambda: F.scaled_dot_product_attention(q,k,v))
            print(layout, be.name, f"{ms:.2f} ms/layer")
        except Exception as e: print(layout, be.name, "FAIL", str(e).splitlines()[0][:120])
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e: print("no flash_attn", e)
