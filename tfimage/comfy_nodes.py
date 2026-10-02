"""ComfyUI nodes: a Qwen-Image 2.1 loader whose MODEL runs the DiT on tfimage (TensorFold kernels).

The MODEL is ComfyUI's own QwenImage21 (model sampling, conditioning, latent format, sampler) built with no blocks;
its ``diffusion_model`` is replaced by the tfimage engine. Text encoding, sampling and VAE stay ComfyUI's, so a
workflow swaps one loader node and keeps everything else. The engine stays resident between prompts and keeps the
text/reference keys and values of recent prompts, so a reseed or a repeat skips the prefix.

Not supported yet (refused, never silently ignored): LoRA / weight patches, ControlNet and block patches.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

PRECISIONS = ["nvfp4", "fp8", "nvfp4:edge=2", "bf16-check"]
_ENGINE: dict = {}                      # one resident engine: {"key": (path, spec), "engine": QwenImage21}


def _engine(path: str, spec: str):
    from . import ext, store

    if _ENGINE.get("key") == (path, spec):
        return _ENGINE["engine"]
    _ENGINE.clear()
    torch.cuda.empty_cache()
    ext.use_prebuilt()
    engine = store.load_or_convert(path, spec, "cuda", log=logging.info)
    _ENGINE.update(key=(path, spec), engine=engine)
    return engine


class TFDiffusionModel(torch.nn.Module):
    """ComfyUI's view of the engine: the QwenImage21 diffusion-model call signature, no parameters of its own."""

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.dtype = torch.bfloat16
        self.transformer_blocks = torch.nn.ModuleList()      # dynamic-VRAM units: none, the engine manages itself
        self.current_patcher = None
        self._room = True                                    # make room on the first call of a sampling run

    def reset_prefix_cache(self, enabled: bool) -> None:
        # called by ComfyUI around each sampling run with the patcher; the engine keys prefixes by content instead
        patcher = self.current_patcher
        if patcher is not None and (getattr(patcher, "patches", None) or getattr(patcher, "hook_patches", None)):
            raise RuntimeError("TensorFold Qwen-Image 2.1: LoRA / weight patches are not supported by this loader yet; "
                               "use the GGUF loader for LoRA workflows")
        self._room = True

    def _make_room(self, x: torch.Tensor) -> None:
        """The engine's weights live outside ComfyUI's model list: before a run, have ComfyUI unload what it must
        (usually the text encoder) so the step's working set fits instead of spilling into shared system memory."""

        import comfy.model_management as mm

        tokens = x.shape[0] * x.shape[-2] * x.shape[-1]
        need = tokens * (6 * 4096 + 3 * 12288) * 2 + (1 << 30)
        mm.free_memory(need, x.device)
        self._room = False

    def forward(self, x, timesteps, context, ref_latents=None, image_slots=None, control=None,
                transformer_options={}, **kwargs):
        if control is not None:
            raise RuntimeError("TensorFold Qwen-Image 2.1: ControlNet is not supported by this loader yet")
        to = transformer_options or {}
        if to.get("patches_replace", {}).get("dit") or any(to.get("patches", {}).get(k) for k in
                                                              ("post_input", "single_block", "attn1_patch")):
            raise RuntimeError("TensorFold Qwen-Image 2.1: block/attention patches are not supported by this loader")
        if self._room:
            self._make_room(x)
        return self.engine(x, timesteps, context, ref_latents, image_slots).to(x.dtype)


def build_model(engine):
    """ComfyUI's QwenImage21 MODEL with the engine as its diffusion model."""

    import comfy.model_detection
    import comfy.model_management as mm
    import comfy.model_patcher

    c = engine.cfg
    unet_config = {"image_model": "qwen_image21", "in_channels": c.in_ch, "out_channels": c.in_ch, "num_layers": 0,
                   "attention_head_dim": c.head_dim, "num_attention_heads": c.heads, "context_in_dim": c.dim}
    model_config = comfy.model_detection.model_config_from_unet_config(unet_config, {})
    load_device = mm.get_torch_device()
    model_config.set_inference_dtype(torch.bfloat16, None, device=load_device)
    model = model_config.get_model({}, "", device=torch.device("cpu"))
    model.diffusion_model = TFDiffusionModel(engine)
    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=mm.unet_offload_device())


def _model_files() -> list[str]:
    import folder_paths

    names = set(folder_paths.get_filename_list("diffusion_models"))
    try:
        names |= set(folder_paths.get_filename_list("unet_gguf"))
    except Exception:  # noqa: BLE001 - ComfyUI-GGUF not installed: its folder name is unknown
        pass
    return sorted(n for n in names if n.endswith(".gguf"))


def _full_path(name: str) -> str:
    import folder_paths

    for folder in ("unet_gguf", "diffusion_models"):
        try:
            p = folder_paths.get_full_path(folder, name)
        except Exception:  # noqa: BLE001
            p = None
        if p:
            return p
    raise FileNotFoundError(name)


class TFQwenImage21Loader:
    """Load a Qwen-Image 2.1 GGUF into the TensorFold image engine (converted once, cached on internal disk)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unet_name": (_model_files(),), "precision": (PRECISIONS, {"default": "nvfp4"})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "TensorFold"

    def load(self, unet_name: str, precision: str):
        spec = "bf16" if precision == "bf16-check" else precision
        engine = _engine(_full_path(unet_name), spec)
        logging.info(f"[tfimage] Qwen-Image 2.1 engine ready: {spec}, {engine.nbytes() / 2**30:.2f} GiB of linears")
        return (build_model(engine),)


NODE_CLASS_MAPPINGS = {"TFQwenImage21Loader": TFQwenImage21Loader}
NODE_DISPLAY_NAME_MAPPINGS = {"TFQwenImage21Loader": "TensorFold Qwen-Image 2.1 Loader"}
