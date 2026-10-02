"""GGUF diffusion checkpoints (ComfyUI-GGUF / sd.cpp layout) -> torch tensors, dequantized on the GPU.

Only the types image GGUFs use are decoded here (BF16, F16, F32, Q8_0, Q6_K, Q5_K, Q4_K); anything else is refused by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch

QK_K = 256


def _f16(b: torch.Tensor) -> torch.Tensor:
    return b.contiguous().view(torch.float16).float()


def _q8_0(raw: torch.Tensor) -> torch.Tensor:
    blk = raw.view(-1, 34)
    return _f16(blk[:, :2]) * blk[:, 2:].view(torch.int8).float()


def _q6_k(raw: torch.Tensor) -> torch.Tensor:
    # block: ql[128] qh[64] scales[16] (int8) d (f16) -> 256 values, ggml's dequantize_row_q6_K order
    blk = raw.view(-1, 210)
    ql, qh = blk[:, :128].int(), blk[:, 128:192].int()
    sc = blk[:, 192:208].view(torch.int8).float()
    d = _f16(blk[:, 208:210])
    ql = ql.view(-1, 2, 64)                              # two 128-value halves
    qh = qh.view(-1, 2, 32)
    lo = torch.stack([ql[:, :, :32] & 15, ql[:, :, 32:] & 15, ql[:, :, :32] >> 4, ql[:, :, 32:] >> 4], dim=2)
    hi = torch.stack([(qh >> s) & 3 for s in (0, 2, 4, 6)], dim=2)
    q = (lo | (hi << 4)) - 32                            # [B, 2, 4, 32]
    s = sc.view(-1, 2, 4, 2).repeat_interleave(16, dim=3)   # scale index is = l/16 within each 32-run, +2 per run
    return (d.view(-1, 1, 1, 1) * s * q).reshape(-1, QK_K)


def _scale_min_k4(sc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 12 bytes -> 8 six-bit scales and mins (ggml get_scale_min_k4)
    q = sc.int()
    s = torch.empty(sc.shape[0], 8, dtype=torch.int32, device=sc.device)
    m = torch.empty_like(s)
    s[:, :4], m[:, :4] = q[:, :4] & 63, q[:, 4:8] & 63
    s[:, 4:] = (q[:, 8:12] & 15) | ((q[:, :4] >> 6) << 4)
    m[:, 4:] = (q[:, 8:12] >> 4) | ((q[:, 4:8] >> 6) << 4)
    return s.float(), m.float()


def _q4_k(raw: torch.Tensor) -> torch.Tensor:
    blk = raw.view(-1, 144)
    d, dmin = _f16(blk[:, :2]), _f16(blk[:, 2:4])
    s, m = _scale_min_k4(blk[:, 4:16])
    qs = blk[:, 16:].int().view(-1, 4, 32)
    q = torch.stack([qs & 15, qs >> 4], dim=2).view(-1, 8, 32).float()
    return (d.view(-1, 1, 1) * s.unsqueeze(-1) * q - dmin.view(-1, 1, 1) * m.unsqueeze(-1)).reshape(-1, QK_K)


def _q5_k(raw: torch.Tensor) -> torch.Tensor:
    blk = raw.view(-1, 176)
    d, dmin = _f16(blk[:, :2]), _f16(blk[:, 2:4])
    s, m = _scale_min_k4(blk[:, 4:16])
    qh = blk[:, 16:48].int()
    qs = blk[:, 48:].int().view(-1, 4, 32)
    lo = torch.stack([qs & 15, qs >> 4], dim=2).view(-1, 8, 32)
    hi = torch.stack([(qh >> j) & 1 for j in range(8)], dim=1)
    q = (lo | (hi << 4)).float()
    return (d.view(-1, 1, 1) * s.unsqueeze(-1) * q - dmin.view(-1, 1, 1) * m.unsqueeze(-1)).reshape(-1, QK_K)


DECODERS = {"Q8_0": _q8_0, "Q6_K": _q6_k, "Q5_K": _q5_k, "Q4_K": _q4_k}


def dequantize(kind: str, data: np.ndarray, shape: tuple[int, ...], device="cuda",
               dtype=torch.bfloat16) -> torch.Tensor:
    """One GGUF tensor's bytes -> a ``shape`` tensor (torch order: the reverse of GGUF's) on ``device``."""

    raw = torch.from_numpy(np.ascontiguousarray(data).view(np.uint8).reshape(-1)).to(device, non_blocking=True)
    if kind in ("BF16", "F16", "F32"):
        src = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[kind]
        return raw.view(src).view(shape).to(dtype)
    if kind not in DECODERS:
        raise ValueError(f"GGUF tensor type {kind} is not decoded by tfimage (decoded: BF16, F16, F32, {', '.join(DECODERS)})")
    return DECODERS[kind](raw).view(shape).to(dtype)


def tensors(path: str | Path, device="cuda", dtype=torch.bfloat16) -> Iterator[tuple[str, torch.Tensor]]:
    """(name, tensor) for every tensor of a GGUF file, dequantized one at a time on ``device``."""

    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    for t in reader.tensors:
        shape = tuple(int(v) for v in reversed(t.shape.tolist()))
        yield t.name, dequantize(t.tensor_type.name, t.data, shape, device, dtype)
