r"""Ground truth for tfimage: one ComfyUI Qwen-Image 2.1 DiT forward on known inputs (portable python, -B).

Saves runs/ref/forward.pt: context, x, t, out (t2i) and x, ref, out_ref (one reference latent, edit-style).
"""
import asyncio, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import paths  # noqa: E402
COMFY = paths.comfy()
sys.path.insert(0, str(COMFY)); sys.path.insert(0, str(ROOT / "bench")); os.chdir(COMFY)
out_dir = ROOT / "runs" / "ref"; out_dir.mkdir(parents=True, exist_ok=True)
sys.argv = ["comfy", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-GGUF", "--database-url", "sqlite:///:memory:",
            "--temp-directory", str(out_dir / "temp")]
import comfy.options; comfy.options.enable_args_parsing()
import torch, nodes, comfy.model_management as mm
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
N = nodes.NODE_CLASS_MAPPINGS
from bench_prompt import PROMPT
torch.inference_mode().__enter__()
clip = N["CLIPLoader"]().load_clip(clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default")[0]
cond = N["TextEncodeQwenImage21"].execute(clip=clip, prompt=PROMPT, negative_prompt="", resolution=1024).result[0]
context = cond[0][0].to(torch.bfloat16)
del clip; mm.unload_all_models(); mm.soft_empty_cache()
model = N["UnetLoaderGGUF"]().load_unet(unet_name="qwen-image-2.1-Q6_K.gguf")[0]
mm.load_models_gpu([model])
dm = model.model.diffusion_model
g = torch.Generator().manual_seed(0)
x = torch.randn(1, 64, 64, 64, generator=g).to("cuda", torch.bfloat16)
ref = torch.randn(1, 64, 48, 40, generator=g).to("cuda", torch.bfloat16)
t = torch.tensor([0.6], device="cuda")
ctx = context.cuda()
out = dm(x, t, ctx, transformer_options={})
slot = 5
out_ref = dm(x, t, ctx, ref_latents=[ref], image_slots=[slot], transformer_options={})
torch.save({"context": ctx.cpu(), "x": x.cpu(), "t": t.cpu(), "out": out.float().cpu(), "ref": ref.cpu(), "slot": slot,
            "out_ref": out_ref.float().cpu()}, out_dir / "forward.pt")
print("saved", out.shape, out.float().abs().mean().item(), out_ref.float().abs().mean().item(), "ctx", tuple(ctx.shape))
