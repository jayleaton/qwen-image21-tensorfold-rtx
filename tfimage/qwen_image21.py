"""Qwen-Image 2.1 DiT on tfimage linears: 32 single-stream blocks (dim 4096, 32 heads x 128, SwiGLU 12288).

The sequence is [text with reference latents spliced in at their slots, target latent]. Text and references attend
block-causally among themselves and are modulated at t = 0, so their keys and values never depend on the step or the
noise: they are computed once per (prompt, references, target shape) and kept, and every step runs the target rows
only, attending to [kept prefix, target]. Norms, RoPE and the modulation run in fp32 like the reference, linears in
the backend chosen per layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import kernels as K
from . import linear as L


@dataclass(frozen=True)
class Config:
    layers: int = 32
    dim: int = 4096
    heads: int = 32
    head_dim: int = 128
    mlp: int = 12288
    in_ch: int = 64
    axes: tuple[int, int, int] = (16, 56, 56)
    theta: float = 10000.0
    eps: float = 1e-6


@dataclass
class Block:
    qkv: L.Linear                  # [q; k; v] rows of one input
    out: L.Linear
    gate_up: L.Linear              # [gate; up]
    down: L.Linear
    norm_q: torch.Tensor           # fp32 [head_dim]
    norm_k: torch.Tensor


@dataclass(eq=False)
class Prefix:
    """Kept keys/values of one prefix: per block (k, v) as [B, P, H, D] bf16, plus what identifies it."""

    context: torch.Tensor
    refs: list[torch.Tensor]
    slots: tuple[int, ...]
    target_hw: tuple[int, int]
    kv: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    def matches(self, context, refs, slots, hw) -> bool:
        return (self.target_hw == hw and self.slots == slots and self.context.shape == context.shape
                and len(self.refs) == len(refs) and torch.equal(self.context, context)
                and all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(self.refs, refs)))


def _rope_table(ids: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """ids [N, 3] -> cos, sin [N, head_dim / 2] (fp32; frequencies in fp64 as the reference computes them)."""

    angles = []
    for axis, dim in enumerate(cfg.axes):
        scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=ids.device)
        omega = 1.0 / (cfg.theta ** scale)
        angles.append(ids[:, axis].to(torch.float64)[:, None] * omega[None])
    a = torch.cat(angles, dim=1)
    return torch.cos(a).float(), torch.sin(a).float()


def policy(spec: str, layers: int):
    """``"nvfp4"`` or ``"nvfp4:down=fp8,out=fp8,edge=2"``: a base kind, per-projection overrides, and ``edge`` blocks at
    each end kept at ``edge_kind`` (default fp8). Returns (block, projection) -> kind."""

    base, _, rest = spec.partition(":")
    over = dict(kv.split("=") for kv in rest.split(",") if kv)
    edge = int(over.pop("edge", 0))
    edge_kind = over.pop("edge_kind", "fp8")

    def pick(i: int, name: str) -> str:
        if i < edge or i >= layers - edge:
            return edge_kind
        return over.get(name, base)
    return pick


class QwenImage21:
    """The DiT with weights in ``kind`` ("bf16", "fp8", "nvfp4"); small layers stay bf16."""

    def __init__(self, w: dict[str, torch.Tensor], kind: str = "nvfp4", device="cuda", cfg: Config = Config(),
                 acts: dict[str, float] | None = None):
        self.cfg, self.kind, self.device = cfg, kind, torch.device(device)
        pick = policy(kind, cfg.layers)
        acts = acts or {}
        dev = lambda name: w[name].to(self.device, torch.bfloat16)       # noqa: E731
        self.img_in = dev("img_in.weight")
        self.txt_norm = w["txt_in.text_norm.weight"].to(self.device, torch.float32) + 1.0   # zero-centred RMSNorm
        self.txt_in1, self.txt_in2 = dev("txt_in.in_layer.weight"), dev("txt_in.out_layer.weight")
        self.t1 = dev("time_text_embed.timestep_embedder.linear_1.weight")
        self.t2 = dev("time_text_embed.timestep_embedder.linear_2.weight")
        self.mod = dev("modulation.1.weight")
        self.norm_out = dev("norm_out.linear.weight")
        self.proj_out = dev("proj_out.weight")
        self.blocks: list[Block] = []
        for i in range(cfg.layers):
            p = f"transformer_blocks.{i}."
            q = torch.cat([w[p + "attn.to_q.weight"], w[p + "attn.to_k.weight"], w[p + "attn.to_v.weight"]])
            gu = torch.cat([w[p + "img_mlp.gate_layer.weight"], w[p + "img_mlp.proj.weight"]])
            def mk(t, name, i=i):
                k = pick(i, name)
                return L.make(k, t if k == "bf16" else t.to(self.device), acts.get(f"{i}.{name}"))
            self.blocks.append(Block(mk(q, "qkv"), mk(w[p + "attn.to_out.0.weight"], "out"), mk(gu, "gate_up"),
                                     mk(w[p + "img_mlp.out.weight"], "down"),
                                     w[p + "attn.norm_q.weight"].to(self.device, torch.float32),
                                     w[p + "attn.norm_k.weight"].to(self.device, torch.float32)))
            for name in [n for n in list(w) if n.startswith(p)]:
                del w[name]                       # free the bf16 copy as soon as a block is quantized
        self.prefix: Prefix | None = None
        self.prefixes: list[Prefix] = []        # most recent last; cond and uncond each keep theirs

    MAX_PREFIXES = 4

    def drop_prefixes(self) -> None:
        self.prefix, self.prefixes = None, []

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_gguf(cls, path: str | Path, kind: str = "nvfp4", device="cuda", acts=None) -> "QwenImage21":
        from .gguf_load import tensors

        stage = "cpu" if kind.split(":")[0] == "bf16" else device      # bf16 (14 GB) cannot sit on a 16 GB card beside activations
        w = {name: t.to(stage) for name, t in tensors(path, device=device)}
        if stage == "cpu":
            torch.cuda.empty_cache()
        return cls(w, kind, device, acts=acts)

    def linears(self):
        for i, b in enumerate(self.blocks):
            for name in ("qkv", "out", "gate_up", "down"):
                yield f"{i}.{name}", getattr(b, name)

    def nbytes(self) -> int:
        return sum(lin.nbytes() for _, lin in self.linears())

    # ---------------------------------------------------------------- pieces
    def _time(self, timesteps: torch.Tensor) -> torch.Tensor:
        t = ((timesteps * 1000).to(torch.bfloat16) / 1000).to(torch.bfloat16)
        t = torch.cat([t, t.new_zeros(1)]).float() * 1000                   # last row: t = 0 for the prefix
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16)
        return F.linear(F.silu(F.linear(emb, self.t1)), self.t2)           # [B + 1, dim]

    def _text(self, context: torch.Tensor) -> torch.Tensor:
        c = context.float()
        c = c * torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + self.cfg.eps) * self.txt_norm
        return F.linear(F.gelu(F.linear(c.to(torch.bfloat16), self.txt_in1), approximate="tanh"), self.txt_in2)

    def _image_ids(self, h: int, w: int, pos: int, target_hw: tuple[int, int]) -> torch.Tensor:
        dev = self.device
        hh = torch.arange(h, device=dev, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - target_hw[0] % 2)
        ww = torch.arange(w, device=dev, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - target_hw[1] % 2)
        return torch.stack([torch.full((h, w), float(pos), device=dev), hh[:, None].expand(h, w),
                            ww[None, :].expand(h, w)], dim=-1).reshape(-1, 3)

    def _attention(self, b: Block, h: torch.Tensor, cos, sin, kv=None, mask=None):
        """One block's attention on rows h [B, N, dim] -> (output [B, N, dim], k, v).

        ``kv``: (kbuf, vbuf) [B, P + N, H, D] scratch whose first P rows hold the kept prefix; this step's keys and
        values are written behind them (no concatenation). Without it (the prefix pass) k and v are fresh tensors."""

        B, N, _ = h.shape
        H, D, eps = self.cfg.heads, self.cfg.head_dim, self.cfg.eps
        qkv = b.qkv(h.reshape(B * N, -1)).view(B, N, 3, H, D)
        q = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=h.device)
        K.rms_rope(qkv, 0, b.norm_q, cos, sin, eps, q)
        if kv is None:
            kk = torch.empty_like(q)
            K.rms_rope(qkv, 1, b.norm_k, cos, sin, eps, kk)
            vv = qkv[:, :, 2].contiguous()
            k, v = kk, vv
        else:
            kk, vv = kv
            P = kk.shape[1] - N
            K.rms_rope(qkv, 1, b.norm_k, cos, sin, eps, kk[:, P:])
            vv[:, P:].copy_(qkv[:, :, 2])
            k = v = None
        qt, kt, vt = q.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2)
        if mask is None:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
                             set_priority=True):
                o = F.scaled_dot_product_attention(qt, kt, vt)
        else:
            o = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
        return b.out(o.transpose(1, 2).reshape(B * N, H * D)).view(B, N, -1), k, v

    fused_mlp = True

    def _mlp(self, b: Block, h: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        x = h.reshape(B * N, -1)
        y = L.swiglu_mlp(b.gate_up, b.down, x) if self.fused_mlp else None
        return (y if y is not None else b.down(K.swiglu(b.gate_up(x)))).view(B, N, -1)

    def _block(self, b: Block, x, mod, cos, sin, kv=None, mask=None):
        s1, g1, s2, g2 = mod
        eps = self.cfg.eps
        a, k, v = self._attention(b, K.adaln(x, s1, eps), cos, sin, kv, mask)
        x.addcmul_(a, g1)
        x.addcmul_(self._mlp(b, K.adaln(x, s2, eps)), g2)
        return x, k, v

    # ---------------------------------------------------------------- prefix
    def _build_prefix(self, context, refs, slots, target_hw, mod0) -> Prefix:
        """Text and references through every block once (t = 0 modulation, block-causal), keys/values kept."""

        txt = self._text(context)
        B, L_txt = txt.shape[:2]
        slots_full = (list(slots) + [L_txt] * len(refs))[:len(refs)]
        bounds = [0] + slots_full + [L_txt]
        parts, ids, segs = [], [], []
        pos = length = 0
        for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            n = end - start
            if n > 0:
                parts.append(txt[:, start:end])
                ids.append(torch.arange(pos, pos + n, device=self.device, dtype=torch.float32)[:, None].expand(n, 3))
                segs.append(("text", length, length + n))
                pos, length = pos + n, length + n
            if i < len(refs):
                r = refs[i]
                h, w = r.shape[-2:]
                parts.append(F.linear(r.to(torch.bfloat16).flatten(2).transpose(1, 2), self.img_in))
                ids.append(self._image_ids(h, w, pos, target_hw))
                segs.append(("image", length, length + h * w))
                pos, length = pos + max(h, w), length + h * w
        x = torch.cat(parts, dim=1)
        P = x.shape[1]
        allow = torch.zeros((P, P), dtype=torch.bool, device=self.device)
        for kind, a, e in segs:
            if kind == "text":
                allow[a:e, :e] = torch.ones((e - a, e), dtype=torch.bool, device=self.device).tril(a)
            else:
                allow[a:e, :e] = True
        cos, sin = _rope_table(torch.cat(ids), self.cfg)
        pre = Prefix(context.clone(), [r.clone() for r in refs], tuple(slots), target_hw)
        for b in self.blocks:
            x, k, v = self._block(b, x, mod0, cos, sin, mask=allow)
            pre.kv.append((k, v))
        pre.next_pos = pos                                    # the target's first id
        return pre

    # ---------------------------------------------------------------- forward
    @torch.inference_mode()
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor,
                ref_latents: list[torch.Tensor] | None = None, image_slots: list[int] | None = None) -> torch.Tensor:
        """x [B, 64, H, W], timesteps [B] in [0, 1], context [B, L, 4096] -> velocity [B, 64, H, W] (bf16)."""

        B, C, H, W = x.shape
        refs = list(ref_latents or [])
        slots = tuple(image_slots or [])
        context = context.to(self.device, torch.bfloat16)
        temb = self._time(timesteps.to(self.device).float().reshape(-1)[:1].expand(B) if timesteps.numel() == 1
                          else timesteps.to(self.device).float())
        m = F.linear(F.silu(temb), self.mod).chunk(4, dim=-1)
        rows = lambda t, sl: t[sl].unsqueeze(1)                 # noqa: E731
        mod_t = (rows(m[0], slice(0, B)), rows(m[1], slice(0, B)).tanh(), rows(m[2], slice(0, B)),
                 rows(m[3], slice(0, B)).tanh())
        mod_0 = tuple(t.expand(B, -1, -1) for t in (rows(m[0], slice(B, B + 1)), rows(m[1], slice(B, B + 1)).tanh(),
                                                     rows(m[2], slice(B, B + 1)), rows(m[3], slice(B, B + 1)).tanh()))
        pre = next((p for p in self.prefixes if p.matches(context, refs, slots, (H, W))), None)
        if pre is None:
            if len(self.prefixes) >= self.MAX_PREFIXES:
                self.prefixes.pop(0)
            pre = self._build_prefix(context, refs, slots, (H, W), mod_0)
        else:
            self.prefixes.remove(pre)
        self.prefixes.append(pre)
        self.prefix = pre
        h = F.linear(x.to(torch.bfloat16).flatten(2).transpose(1, 2), self.img_in).contiguous()
        cos, sin = _rope_table(self._image_ids(H, W, pre.next_pos, (H, W)), self.cfg)
        P = pre.kv[0][0].shape[1]
        kbuf = torch.empty((B, P + H * W, self.cfg.heads, self.cfg.head_dim), dtype=torch.bfloat16, device=self.device)
        vbuf = torch.empty_like(kbuf)
        for b, (pk, pv) in zip(self.blocks, pre.kv):
            kbuf[:, :P].copy_(pk)
            vbuf[:, :P].copy_(pv)
            h, _, _ = self._block(b, h, mod_t, cos, sin, kv=(kbuf, vbuf))
        scale = F.linear(F.silu(temb[:B]), self.norm_out).unsqueeze(1)
        h = K.adaln(h, scale, self.cfg.eps)
        out = F.linear(h, self.proj_out)
        return out.transpose(1, 2).reshape(B, C, H, W)

    __call__ = forward
