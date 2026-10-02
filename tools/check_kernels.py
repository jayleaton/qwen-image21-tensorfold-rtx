"""tfimage fused kernels vs plain torch (fp32 inside) on small DiT-shaped tensors."""
import torch, torch.nn.functional as F
from tfimage import kernels as K
from tfimage.qwen_image21 import _rope_table, Config
torch.manual_seed(0); cfg = Config()
B, N, D, H, Dh = 2, 300, 4096, 32, 128
x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16) * 3; s = torch.randn(B, 1, D, device="cuda") * 0.2
ref = (F.layer_norm(x.float(), (D,), eps=1e-6) * (1 + s)).to(torch.bfloat16)
print("adaln max|d|", (K.adaln(x, s, 1e-6).float() - ref.float()).abs().max().item(), "ref absmax", ref.float().abs().max().item())
qkv = torch.randn(B, N, 3, H, Dh, device="cuda", dtype=torch.bfloat16); w = torch.rand(Dh, device="cuda") + 0.5
ids = torch.stack([torch.arange(N, device="cuda").float()] * 3, -1); cos, sin = _rope_table(ids, cfg)
xf = qkv[:, :, 1].float(); xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w
x0, x1 = xf[..., 0::2], xf[..., 1::2]; c, si = cos[None, :, None], sin[None, :, None]
ref = torch.stack([c * x0 - si * x1, si * x0 + c * x1], -1).flatten(-2).to(torch.bfloat16)
buf = torch.zeros(B, N + 7, H, Dh, device="cuda", dtype=torch.bfloat16)
K.rms_rope(qkv, 1, w, cos, sin, 1e-6, buf[:, 7:])
print("rms_rope max|d|", (buf[:, 7:].float() - ref.float()).abs().max().item(), "prefix rows untouched", buf[:, :7].abs().max().item() == 0)
gu = torch.randn(B * N, 2 * 12288, device="cuda", dtype=torch.bfloat16) * 2
g, u = gu.split(12288, -1); ref = F.silu(g) * u
print("swiglu max|d|", (K.swiglu(gu).float() - ref.float()).abs().max().item(), "ref absmax", ref.float().abs().max().item())
