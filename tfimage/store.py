"""Converted engines on disk: a quantized Qwen-Image 2.1 saved once, loaded in seconds from internal NVMe.

Converting a GGUF reads the whole file (5.9 GB for Q6_K; slow from an external drive) and quantizes 224 linears. The
result (~3.9 GB for NVFP4) goes to a cache keyed by the source file's path, size and mtime and the precision spec.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from . import linear as L

SMALL = ("img_in", "txt_norm", "txt_in1", "txt_in2", "t1", "t2", "mod", "norm_out", "proj_out")
FORMAT = 1


def cache_dir() -> Path:
    base = os.environ.get("TFIMAGE_CACHE") or os.path.join(os.environ.get("LOCALAPPDATA", str(Path.home())), "tfimage")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def key(source: str | Path, spec: str) -> str:
    st = os.stat(source)
    raw = f"{Path(source).resolve()}|{st.st_size}|{st.st_mtime_ns}|{spec}|{FORMAT}"
    return f"{Path(source).stem}-{spec.replace(':', '_').replace(',', '_').replace('=', '-')}-" \
           f"{hashlib.sha1(raw.encode()).hexdigest()[:10]}"


def _lin_tensors(prefix: str, lin: L.Linear) -> tuple[dict, dict]:
    if isinstance(lin, L.Nvfp4Linear):
        f = lin.lin
        return ({prefix + "words": f.words, prefix + "bs": f.bs},
                {"kind": "nvfp4", "scale": f.scale, "n": f.n, "k": f.k, "act": None if lin.dynamic else f.act,
                 "tile": lin.tile})
    if isinstance(lin, L.Fp8Linear):
        return ({prefix + "w8": lin.w8.view(torch.uint8), prefix + "ws": lin.ws.reshape(1)},
                {"kind": "fp8", "n": lin.n, "k": lin.k, "act": lin.act})
    return {prefix + "w": lin.w}, {"kind": "bf16", "n": lin.n, "k": lin.k}


def save(model, path: Path) -> None:
    tensors, meta = {}, {"format": FORMAT, "kind": model.kind, "linears": {}}
    for name in SMALL:
        tensors[name] = getattr(model, name)
    for i, b in enumerate(model.blocks):
        tensors[f"{i}.norm_q"], tensors[f"{i}.norm_k"] = b.norm_q, b.norm_k
        for proj in ("qkv", "out", "gate_up", "down"):
            t, m = _lin_tensors(f"{i}.{proj}.", getattr(b, proj))
            tensors.update(t)
            meta["linears"][f"{i}.{proj}"] = m
    tmp = path.with_suffix(".tmp")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp), metadata={"tfimage": json.dumps(meta)})
    os.replace(tmp, path)


def _lin_load(t: dict, prefix: str, m: dict, device) -> L.Linear:
    if m["kind"] == "nvfp4":
        from tensorfold.cuda.nvfp4.linear import Fp4Linear

        lin = L.Nvfp4Linear.__new__(L.Nvfp4Linear)
        lin.lin = Fp4Linear(t[prefix + "words"].to(device), t[prefix + "bs"].to(device), m["scale"], m["n"], m["k"],
                            act=m["act"] if m["act"] is not None else 1.0)
        lin.n, lin.k, lin.dynamic, lin.tile, lin.amax = m["n"], m["k"], m["act"] is None, m.get("tile", 12), 0.0
        return lin
    if m["kind"] == "fp8":
        lin = L.Fp8Linear.__new__(L.Fp8Linear)
        lin.w8 = t[prefix + "w8"].to(device).view(torch.float8_e4m3fn)
        lin.ws = t[prefix + "ws"].to(device).reshape(())
        lin.n, lin.k, lin.act, lin._a = m["n"], m["k"], m["act"], None
        return lin
    return L.Bf16Linear(t[prefix + "w"].to(device))


def load(path: Path, device="cuda"):
    from safetensors import safe_open

    from .qwen_image21 import Block, Config, QwenImage21

    with safe_open(str(path), "pt") as f:
        meta = json.loads(f.metadata()["tfimage"])
    t = load_file(str(path), device="cpu")
    model = QwenImage21.__new__(QwenImage21)
    model.cfg, model.kind, model.device, model.prefix = Config(), meta["kind"], torch.device(device), None
    model.prefixes = []
    for name in SMALL:
        setattr(model, name, t[name].to(device))
    model.blocks = []
    for i in range(model.cfg.layers):
        lins = {p: _lin_load(t, f"{i}.{p}.", meta["linears"][f"{i}.{p}"], device) for p in ("qkv", "out", "gate_up", "down")}
        model.blocks.append(Block(lins["qkv"], lins["out"], lins["gate_up"], lins["down"], t[f"{i}.norm_q"].to(device),
                                  t[f"{i}.norm_k"].to(device)))
    return model


def load_or_convert(source: str | Path, spec: str, device="cuda", log=print):
    """The converted engine for ``source`` at ``spec``: from the cache, or converted now and cached."""

    from .qwen_image21 import QwenImage21

    path = cache_dir() / (key(source, spec) + ".safetensors")
    if path.is_file():
        log(f"[tfimage] loading converted {path.name}")
        return load(path, device)
    log(f"[tfimage] converting {Path(source).name} to {spec} (once; cached at {path})")
    model = QwenImage21.from_gguf(source, spec, device)
    contexts = cache_dir() / "calib_contexts.pt"
    if any(lin.kind == "nvfp4" for _, lin in model.linears()):
        if contexts.is_file():
            from .calibrate import calibrate

            calibrate(model, torch.load(contexts), log=log)
        else:
            log(f"[tfimage] no {contexts}: NVFP4 inputs are scaled per call (a host sync each); run "
                "bench/comfy_dump_contexts.py to calibrate")
    save(model, path)
    return model
