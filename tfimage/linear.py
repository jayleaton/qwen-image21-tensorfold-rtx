"""Linear layers for DiT steps: bf16 (reference), FP8 W8A8 and NVFP4 W4A4 on TensorFold's prompt GEMMs.

A DiT step multiplies thousands of rows by every weight, so it is compute bound: the win is the tensor-core rate of the
operand format (bf16 ~85, FP8 ~170, NVFP4 ~400 TF/s on an RTX 5070 Ti), not the bytes read. Activations are quantized
under a per-layer input scale ``act``: measured on the fly (``act=None``, one host sync a call, for calibration) or
fixed from a calibration run (no sync, CUDA-graph safe). Per-16 (NVFP4) block scales still follow every row's range.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

FP4_MAX, E4M3_MAX = 6.0, 448.0
TILES = (0, 1, 2, 12, 13)        # TensorFold prompt tiles worth trying on sm_120 (12-13: bulk-copy tiles)


def _ck():
    from tensorfold.cuda.nvfp4 import checkpoint
    return checkpoint


@lru_cache(maxsize=1)
def nvfp4_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def nvfp4_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """bf16 [N, K] -> (e2m1 codes [N, K/2], e4m3 scales [N, K/16], global scale), ModelOpt's NVFP4 layout."""

    ck = _ck()
    n, k = w.shape
    g = max(float(w.float().abs().amax()), 1e-12) / (FP4_MAX * E4M3_MAX)
    rows = ck.quant4(w.to(torch.bfloat16).contiguous(), g)
    scales = rows.scales[:, :n, :].permute(1, 0, 2).reshape(n, k // 16)
    return rows.codes, scales.view(torch.float8_e4m3fn), g


class Linear:
    """y = x @ W^T for 2-D bf16 rows; ``kind`` names the backend."""

    kind = "base"
    n: int
    k: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError


class Bf16Linear(Linear):
    kind = "bf16"

    def __init__(self, w: torch.Tensor):
        self.w = w.to(torch.bfloat16).contiguous()
        self.n, self.k = self.w.shape

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w if self.w.device == x.device else self.w.to(x.device, non_blocking=True)
        return F.linear(x, w)

    def nbytes(self) -> int:
        return self.w.numel() * 2


class Fp8Linear(Linear):
    """W8A8 e4m3: one weight scale and one input scale (tensor-wise, what cuBLASLt takes on sm_120), ``_scaled_mm``."""

    kind = "fp8"

    def __init__(self, w: torch.Tensor, act: float | None = None):
        w = w.float()
        self.n, self.k = w.shape
        s = (w.abs().amax() / E4M3_MAX).clamp_min(1e-12)
        self.w8 = (w / s).to(torch.float8_e4m3fn).contiguous()
        self.ws = s.view(()).contiguous()                       # fp32 scalar
        self.act = act
        self._a = None

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act is None:
            return (x.abs().amax().float() / E4M3_MAX).clamp_min(1e-12)
        if self._a is None or self._a.device != x.device:
            self._a = torch.tensor(self.act, dtype=torch.float32, device=x.device)
        return self._a

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        a = self._act(x)
        m = x.shape[0]
        pad = -m % 16
        xq = (x.float() / a).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        if pad:
            xq = F.pad(xq.view(torch.uint8), (0, 0, 0, pad)).view(torch.float8_e4m3fn)
        y = torch._scaled_mm(xq, self.w8.t(), scale_a=a, scale_b=self.ws, out_dtype=torch.bfloat16)
        return y[:m] if pad else y

    def nbytes(self) -> int:
        return self.w8.numel() + 4


class Nvfp4Linear(Linear):
    """W4A4 NVFP4 (e2m1 values, e4m3 scale per 16, fp32 global) on TensorFold's block-scaled FP4 prompt GEMM."""

    kind = "nvfp4"

    def __init__(self, w: torch.Tensor, act: float | None = None, tile: int = 12):
        from tensorfold.cuda.nvfp4.linear import Fp4Linear

        codes, scales, g = nvfp4_weight(w)
        self.lin = Fp4Linear.from_checkpoint(codes, scales, g, act=act if act is not None else 1.0)
        self.n, self.k = self.lin.n, self.lin.k
        self.dynamic = act is None
        self.tile = tile
        self.amax = 0.0                 # largest input scale seen (calibration)

    def set_act(self, act: float | None) -> None:
        self.dynamic = act is None
        if act is not None:
            self.lin.act = float(act)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic:
            a = max(float(x.abs().amax()), 1e-6) / (FP4_MAX * E4M3_MAX)
            self.amax = max(self.amax, a)
            self.lin.act = a
        return _ck().prompt(_ck().A4, x, self.lin, tile=self.tile)

    def nbytes(self) -> int:
        return self.lin.words.numel() * 4 + self.lin.bs.numel()


def swiglu_mlp(gate_up: Linear, down: Linear, x: torch.Tensor):
    """down(silu(gate) * up) in one TensorFold pass when both are static-scale NVFP4: the gate/up GEMM's epilogue
    writes the SwiGLU rows as NVFP4 under down's input scale, so no bf16 [M, 2F] or [M, F] tensor exists. None if
    the layers cannot take it (the caller runs the unfused path)."""

    if not (isinstance(gate_up, Nvfp4Linear) and isinstance(down, Nvfp4Linear)) or gate_up.dynamic or down.dynamic:
        return None
    f = gate_up.n // 2
    if f % 64 or down.k != f:
        return None
    if getattr(gate_up, "_halves", None) is None:
        t = f // 64
        gate_up._halves = (gate_up.lin.tiles(0, t), gate_up.lin.tiles(t, 2 * t))
    gate, up = gate_up._halves
    return _ck().mlp_prompt(x, gate, up, down.lin)


def make(kind: str, w: torch.Tensor, act: float | None = None) -> Linear:
    """A backend by name for a bf16 [N, K] weight (dims NVFP4 cannot tile fall back to FP8, then bf16)."""

    n, k = w.shape
    if kind == "nvfp4" and nvfp4_supported() and k % 128 == 0 and n % 64 == 0:
        return Nvfp4Linear(w, act)
    if kind in ("nvfp4", "fp8") and k % 16 == 0 and n % 16 == 0:
        return Fp8Linear(w, act)
    return Bf16Linear(w)
