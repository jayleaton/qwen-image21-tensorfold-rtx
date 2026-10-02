r"""Per-component profile of a Qwen-Image 2.1 ComfyUI setup, driving ComfyUI's own code as a library (portable python).

Run with %COMFYUI_PORTABLE%\python_embeded\python.exe -B bench/comfy_profile.py --unet ... --w 1024 --h 1024
Writes runs/<tag>/profile.json and a kernel table (top CUDA kernels for one DiT step).
"""
import argparse, asyncio, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import paths  # noqa: E402
COMFY = paths.comfy()
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="prof_q6k_1024"); ap.add_argument("--unet", default="qwen-image-2.1-Q6_K.gguf")
ap.add_argument("--clip", default="qwen3vl_8b_int8_convrot.safetensors"); ap.add_argument("--w", type=int, default=1024)
ap.add_argument("--h", type=int, default=1024); ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--extra", nargs="*", default=[])
a = ap.parse_args()
run = ROOT / "runs" / a.tag; run.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(COMFY)); sys.path.insert(0, str(ROOT / "bench")); os.chdir(COMFY)
sys.argv = ["comfy", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-GGUF", "--database-url", "sqlite:///:memory:",
            "--output-directory", str(run / "images"), "--temp-directory", str(run / "temp"), *a.extra]
import comfy.options; comfy.options.enable_args_parsing()
import torch, nodes, comfy.model_management as mm
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
N = nodes.NODE_CLASS_MAPPINGS
torch.inference_mode().__enter__()
from bench_prompt import PROMPT  # noqa: E402  (sibling module)


def call(name, **kw):
    cls = N[name]
    if hasattr(cls, "execute") and hasattr(cls, "define_schema"):
        return tuple(cls.execute(**kw).result)
    obj = cls(); return getattr(obj, cls.FUNCTION)(**kw)


def sync(): torch.cuda.synchronize()
T = {}
t = time.perf_counter(); clip = call("CLIPLoader", clip_name=a.clip, type="qwen_image", device="default")[0]; T["te_load_s"] = time.perf_counter() - t
t = time.perf_counter(); pos, neg = call("TextEncodeQwenImage21", clip=clip, prompt=PROMPT, negative_prompt="", resolution=1024)[:2]; sync(); T["te_first_encode_s"] = time.perf_counter() - t
t = time.perf_counter(); call("TextEncodeQwenImage21", clip=clip, prompt=PROMPT + " ", negative_prompt="", resolution=1024); sync(); T["te_warm_encode_s"] = time.perf_counter() - t
T["text_tokens"] = int(pos[0][0].shape[1])
loader = "UnetLoaderGGUF" if a.unet.endswith(".gguf") else "UNETLoader"
kw = {"unet_name": a.unet} if loader == "UnetLoaderGGUF" else {"unet_name": a.unet, "weight_dtype": "default"}
t = time.perf_counter(); model = call(loader, **kw)[0]; T["dit_load_s"] = time.perf_counter() - t
model = call("QwenImage21Cache", model=model, device="auto", dtype="default")[0]
vae = call("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors")[0]
lat = call("EmptyLatentImage", width=a.w, height=a.h, batch_size=1)[0]

steps_t = []
def ks(steps, seed=830609515524582):
    stamps = [time.perf_counter()]
    import comfy.utils
    orig = comfy.utils.ProgressBar.update_absolute
    def hook(self, value, total=None, preview=None):
        sync(); stamps.append(time.perf_counter()); return orig(self, value, total, preview)
    comfy.utils.ProgressBar.update_absolute = hook
    try:
        out = nodes.common_ksampler(model, seed, steps, 1.0, "euler", "simple", pos, neg, lat, denoise=1.0)[0]
    finally:
        comfy.utils.ProgressBar.update_absolute = orig
    sync(); stamps.append(time.perf_counter())
    return out, [b - a_ for a_, b in zip(stamps, stamps[1:])]

t = time.perf_counter(); samples, d = ks(a.steps); T["sample_first_s"] = time.perf_counter() - t; T["first_step_times"] = d
t = time.perf_counter(); samples, d = ks(a.steps); T["sample_warm_s"] = time.perf_counter() - t; T["warm_step_times"] = d
t = time.perf_counter(); img = call("VAEDecode", samples=samples, vae=vae)[0]; sync(); T["vae_first_s"] = time.perf_counter() - t
t = time.perf_counter(); img = call("VAEDecode", samples=samples, vae=vae)[0]; sync(); T["vae_warm_s"] = time.perf_counter() - t
T["latent_shape"] = list(samples["samples"].shape)
T["vram_peak_alloc_mib"] = torch.cuda.max_memory_allocated() // 2**20
T["vram_peak_reserved_mib"] = torch.cuda.max_memory_reserved() // 2**20
from PIL import Image; import numpy as np
(run / "images").mkdir(exist_ok=True)
Image.fromarray((img[0].detach().float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).save(run / "images" / "comfy.png")

# kernel breakdown for 2 steps
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    ks(2)
evs = prof.key_averages()
rows = sorted([(e.key, e.device_time_total / 1e3 / 2, e.count // 2) for e in evs if e.device_time_total > 0], key=lambda r: -r[1])
tot = sum(r[1] for r in rows if not r[0].startswith(("aten::", "cuda", "Memcpy"))) or 1
kern = [r for r in rows if not r[0].startswith("aten::")]
T["kernels_ms_per_step"] = [(k[:120], round(ms, 2), n) for k, ms, n in kern[:40]]
T["kernel_total_ms_per_step"] = round(sum(r[1] for r in kern), 1)
(run / "profile.json").write_text(json.dumps(T, indent=1))
print(json.dumps({k: v for k, v in T.items() if k != "kernels_ms_per_step"}, indent=1))
for k, ms, n in T["kernels_ms_per_step"][:30]: print(f"{ms:8.2f} ms  x{n:<5d} {k}")
