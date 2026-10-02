"""Quality gate: images of a candidate run vs the baseline run, same prompts and seeds (job00..jobNN).

Metrics per image: LPIPS (alex, lower is closer), DINOv2-S cosine of CLS features (higher is closer), PSNR.
Writes runs/<cand>/quality.json and a contact sheet runs/quality_<cands>.png (baseline first column).
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("cands", nargs="+"); ap.add_argument("--thumb", type=int, default=320)
a = ap.parse_args()

def jobs(tag):
    out = {}
    for f in sorted(glob.glob(str(ROOT / "runs" / tag / "images" / "job*_*.png"))):
        out[Path(f).name.split("_")[0]] = f
    return out

def tens(path, size=None):
    im = Image.open(path).convert("RGB")
    if size: im = im.resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()

import lpips
lp = lpips.LPIPS(net="alex", verbose=False).cuda()
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).cuda().eval()
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

base = jobs(a.base)
rows, summary = [], {}
for cand in a.cands:
    cj = jobs(cand); res = {}
    for j, bp in base.items():
        if j not in cj: continue
        x, y = tens(bp), tens(cj[j])
        with torch.no_grad():
            l = lp(x * 2 - 1, y * 2 - 1).item()
            fx = dino((tens(bp, 448) - mean) / std); fy = dino((tens(cj[j], 448) - mean) / std)
            d = torch.nn.functional.cosine_similarity(fx, fy).item()
        mse = ((x - y) ** 2).mean().item(); psnr = 10 * np.log10(1 / max(mse, 1e-10))
        res[j] = {"lpips": round(l, 4), "dino": round(d, 4), "psnr": round(psnr, 2)}
    agg = {k: round(float(np.mean([r[k] for r in res.values()])), 4) for k in ("lpips", "dino", "psnr")}
    agg["worst_lpips"] = max(r["lpips"] for r in res.values()); agg["worst_dino"] = min(r["dino"] for r in res.values())
    summary[cand] = agg
    (ROOT / "runs" / cand / "quality.json").write_text(json.dumps({"vs": a.base, "mean": agg, "images": res}, indent=1))
    print(cand, json.dumps(agg), flush=True)
    for j, r in res.items(): print("   ", j, r)

T = a.thumb; cols = [a.base] + a.cands
sheet = Image.new("RGB", (T * len(cols), (T + 18) * len(base) + 18), "white"); dr = ImageDraw.Draw(sheet)
for c, tag in enumerate(cols): dr.text((c * T + 4, 2), tag, fill="black")
for r, j in enumerate(base):
    for c, tag in enumerate(cols):
        f = jobs(tag).get(j)
        if not f: continue
        sheet.paste(Image.open(f).convert("RGB").resize((T, T)), (c * T, 18 + r * (T + 18)))
        if c: dr.text((c * T + 4, 18 + r * (T + 18) + T), f"lpips {summary[tag] and json.load(open(ROOT / 'runs' / tag / 'quality.json'))['images'][j]['lpips']}", fill="black")
out = ROOT / "runs" / f"quality_{'_'.join(a.cands)}.png"; sheet.save(out); print("sheet", out)
