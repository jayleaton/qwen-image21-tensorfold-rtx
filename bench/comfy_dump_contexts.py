r"""Text-encoder outputs (ComfyUI, the int8_convrot Qwen3-VL text encoder) for calibration prompts -> runs/ref/contexts.pt."""
import asyncio, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import paths  # noqa: E402
COMFY = paths.comfy()
sys.path.insert(0, str(COMFY)); os.chdir(COMFY)
ref = ROOT / "runs" / "ref"
ref.mkdir(parents=True, exist_ok=True)
sys.argv = ["comfy", "--disable-all-custom-nodes", "--database-url", "sqlite:///:memory:"]
for name in ("temp", "user", "input", "output"):
    (ref / name).mkdir(exist_ok=True)
    sys.argv += [f"--{name}-directory", str(ref / name)]
import comfy.options; comfy.options.enable_args_parsing()
import torch, nodes
asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
N = nodes.NODE_CLASS_MAPPINGS
torch.inference_mode().__enter__()
clip = N["CLIPLoader"]().load_clip(clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default")[0]
prompts = json.load(open(ROOT / "bench" / "calib_prompts.json", encoding="utf-8"))
ctx = [N["TextEncodeQwenImage21"].execute(clip=clip, prompt=p, negative_prompt="", resolution=1024).result[0][0][0].to(torch.bfloat16).cpu() for p in prompts]
torch.save(ctx, ROOT / "runs" / "ref" / "contexts.pt"); print("saved", [tuple(c.shape) for c in ctx])
