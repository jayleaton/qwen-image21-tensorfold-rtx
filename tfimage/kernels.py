"""Fused elementwise kernels for a DiT block (Triton): each reads its input once and writes its output once.

Unfused, the fp32 LayerNorm / RMSNorm / RoPE / SwiGLU chains of a Qwen-Image 2.1 step cost ~400 ms of temporaries on
an RTX 5070 Ti, three times the NVFP4 GEMMs. These kernels keep the reference arithmetic (fp32 inside, bf16 out).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _adaln_kernel(x_ptr, s_ptr, o_ptr, rows_per_b, D: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = row // rows_per_b
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row.to(tl.int64) * D + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    s = tl.load(s_ptr + b * D + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * tl.rsqrt(var + eps) * (1.0 + s)
    tl.store(o_ptr + row.to(tl.int64) * D + cols, y.to(tl.bfloat16), mask=mask)


def adaln(x: torch.Tensor, scale: torch.Tensor, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """LayerNorm(x) * (1 + scale): x [B, N, D] bf16, scale [B, 1, D] (any float) -> bf16 [B, N, D]."""

    B, N, D = x.shape
    x = x.contiguous()
    s = scale.reshape(B, D).float().contiguous()
    out = torch.empty_like(x) if out is None else out
    _adaln_kernel[(B * N,)](x, s, out, N, D, eps, BLOCK=triton.next_power_of_2(D), num_warps=8)
    return out


@triton.jit
def _rms_rope_kernel(src_ptr, w_ptr, cos_ptr, sin_ptr, dst_ptr, src_b_stride, src_row_stride, src_off, dst_b_stride,
                     dst_row_stride, N, H: tl.constexpr, D: tl.constexpr, eps):
    # one (row, head): RMSNorm over D in fp32, weight, then RoPE on interleaved pairs (2i, 2i+1)
    pid = tl.program_id(0)
    row = pid // H
    head = pid % H
    n = row % N
    bi = (row // N).to(tl.int64)
    half = tl.arange(0, D // 2)
    base = src_ptr + bi * src_b_stride + n.to(tl.int64) * src_row_stride + src_off + head * D
    x0 = tl.load(base + 2 * half).to(tl.float32)
    x1 = tl.load(base + 2 * half + 1).to(tl.float32)
    rr = tl.rsqrt((tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0)) / D + eps)
    x0 = x0 * rr * tl.load(w_ptr + 2 * half)
    x1 = x1 * rr * tl.load(w_ptr + 2 * half + 1)
    c = tl.load(cos_ptr + n * (D // 2) + half)
    s = tl.load(sin_ptr + n * (D // 2) + half)
    out = dst_ptr + bi * dst_b_stride + n.to(tl.int64) * dst_row_stride + head * D
    tl.store(out + 2 * half, (c * x0 - s * x1).to(tl.bfloat16))
    tl.store(out + 2 * half + 1, (s * x0 + c * x1).to(tl.bfloat16))


def rms_rope(src: torch.Tensor, part: int, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float,
             dst: torch.Tensor) -> None:
    """RMSNorm + RoPE of ``src``'s ``part`` (0 = q, 1 = k of a [B, N, 3, H, D] qkv) into ``dst`` [B, N, H, D]
    (``dst`` may be a row slice of a larger [B, P + N, H, D] buffer: its strides are used as they are)."""

    B, N, _, H, D = src.shape
    _rms_rope_kernel[(B * N * H,)](src, w, cos, sin, dst, src.stride(0), src.stride(1), part * H * D, dst.stride(0),
                                   dst.stride(1), N, H, D, eps, num_warps=1)


@triton.jit
def _swiglu_kernel(gu_ptr, o_ptr, F: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < F
    g = tl.load(gu_ptr + row.to(tl.int64) * 2 * F + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row.to(tl.int64) * 2 * F + F + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32) * u      # silu rounded to bf16 as the reference does
    tl.store(o_ptr + row.to(tl.int64) * F + cols, y.to(tl.bfloat16), mask=mask)


def swiglu(gu: torch.Tensor) -> torch.Tensor:
    """[M, 2F] = [gate | up] bf16 -> silu(gate) * up, [M, F] bf16."""

    M, F2 = gu.shape
    F = F2 // 2
    out = torch.empty((M, F), dtype=torch.bfloat16, device=gu.device)
    BLOCK = 2048
    _swiglu_kernel[(M, triton.cdiv(F, BLOCK))](gu.contiguous(), out, F, BLOCK=BLOCK, num_warps=8)
    return out
