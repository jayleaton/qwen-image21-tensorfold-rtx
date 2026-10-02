"""Flow-matching sampling as ComfyUI runs Qwen-Image 2.1: shifted sigmas (shift 0.69), "simple" schedule, Euler."""

from __future__ import annotations

import math

import torch

SHIFT = 0.69


def sigmas(steps: int, shift: float = SHIFT, table: int = 10000) -> list[float]:
    t = torch.arange(1, table + 1, dtype=torch.float32) / table
    s = math.exp(shift) / (math.exp(shift) + (1 / t - 1))
    stride = table / steps
    return [float(s[-(1 + int(i * stride))]) for i in range(steps)] + [0.0]


@torch.inference_mode()
def euler(model, noise: torch.Tensor, context: torch.Tensor, steps: int = 25, callback=None, **kw) -> torch.Tensor:
    """Text-to-image latent from ``noise`` (unit Gaussian, [B, 64, H, W]); the model predicts velocity."""

    sig = sigmas(steps)
    x = noise.float() * sig[0]
    for i in range(steps):
        t = torch.full((x.shape[0],), sig[i], device=x.device)
        v = model(x.to(torch.bfloat16), t, context, **kw).float()
        x = x + (sig[i + 1] - sig[i]) * v
        if callback:
            callback(i, x)
    return x
