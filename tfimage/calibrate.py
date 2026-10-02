"""Static input scales for NVFP4 linears: the largest per-layer input range over real denoising runs, times a margin.

With per-16 e4m3 block scales the global input scale only sets the range: a 4x margin leaves blocks down to ~1e-4 of
the calibrated maximum at full scale precision and makes clipping unlikely for prompts and sizes outside the set.
"""

from __future__ import annotations

import torch

from . import linear as L
from .sampling import euler

MARGIN = 4.0
SIZES = ((64, 64), (76, 62), (48, 80))          # 1024^2, 1216x992 (a common portrait size), 768x1280 latents


def calibrate(model, contexts: list[torch.Tensor], steps: int = 12, sizes=SIZES, margin: float = MARGIN, log=print):
    lins = [(name, lin) for name, lin in model.linears() if isinstance(lin, L.Nvfp4Linear)]
    for _, lin in lins:
        lin.set_act(None)
        lin.amax = 0.0
    g = torch.Generator(device="cuda").manual_seed(1234)
    for i, ctx in enumerate(contexts):
        hw = sizes[i % len(sizes)]
        noise = torch.randn((1, 64, *hw), generator=g, device="cuda")
        model.drop_prefixes()
        euler(model, noise, ctx.cuda(), steps)
        log(f"[tfimage] calibration {i + 1}/{len(contexts)} at {hw[0] * 16}x{hw[1] * 16}")
    acts = {name: lin.amax * margin for name, lin in lins}
    for name, lin in lins:
        lin.set_act(acts[name])
    model.drop_prefixes()
    return acts
