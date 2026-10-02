"""Baseline: drive a ComfyUI portable install headless (only ComfyUI-GGUF loaded, all dirs redirected here).

Per run: wall seconds, sampler s/it, per-node timings from the log, VRAM peak (device-wide, minus idle).
Usage: python bench/comfy_baseline.py --tag q6k_1024 --width 1024 --height 1024 [--unet ...] [--repeat 3]
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
PORTABLE = paths.portable()
PORT = 8199
PROMPT = ("A greyscale fashion editorial portrait of a woman in profile wearing oversized round black sunglasses and a "
          "structured high-necked dress with a bold op-art camouflage pattern, against a lime green pop-art collage "
          "background with zebra-striped semicircles and halftone dots, razor-sharp studio key light")


def graph(a) -> dict:
    if a.tf:
        unet = {"class_type": "TFQwenImage21Loader", "inputs": {"unet_name": a.unet, "precision": a.tf}}
    elif a.unet.endswith(".gguf"):
        unet = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": a.unet}}
    else:
        unet = {"class_type": "UNETLoader", "inputs": {"unet_name": a.unet, "weight_dtype": "default"}}
    g = {
        "1": unet,
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": a.clip, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
        "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": a.prompt, "negative_prompt": "", "resolution": 1024, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": a.width, "height": a.height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"seed": a.seed, "steps": a.steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                                                   "denoise": 1.0, "model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": a.tag, "images": ["8", 0]}},
    }
    return g


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.stop = False
    def run(self):
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
                             stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            if self.stop: break
            try: self.peak = max(self.peak, int(line.strip()))
            except ValueError: pass
        p.kill()


def vram_now() -> int:
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())


def call(path, data=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run_once(g) -> float:
    t0 = time.perf_counter()
    pid = call("/prompt", {"prompt": g})["prompt_id"]
    while True:
        time.sleep(0.1)
        h = call(f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                raise RuntimeError(json.dumps(st)[:2000])
            return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--width", type=int, default=1024); ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=25); ap.add_argument("--seed", type=int, default=830609515524582)
    ap.add_argument("--unet", default="qwen-image-2.1-Q6_K.gguf"); ap.add_argument("--clip", default="qwen3vl_8b_int8_convrot.safetensors")
    ap.add_argument("--prompt", default=PROMPT); ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--tf", default="", help="run the DiT on the TensorFold node at this precision (nvfp4, fp8, ...)")
    ap.add_argument("--jobs", default="", help="JSON file: [[prompt, seed], ...] rendered once each after the timing runs")
    a = ap.parse_args()
    run_dir = ROOT / "runs" / a.tag; run_dir.mkdir(parents=True, exist_ok=True)
    for d in ("user", "temp", "input"): (run_dir / d).mkdir(exist_ok=True)
    idle = vram_now()
    log = open(run_dir / "comfy.log", "w", encoding="utf-8")
    # lets ComfyUI load this repo's node without touching its own custom_nodes folder
    extra = run_dir / "extra_paths.yaml"
    extra.write_text(f"tfimage:\n  base_path: {(ROOT / 'comfyui').as_posix()}\n  custom_nodes: .\n", encoding="utf-8")
    cmd = [str(PORTABLE / "python_embeded" / "python.exe"), "-s", str(PORTABLE / "ComfyUI" / "main.py"), "--windows-standalone-build",
           "--port", str(PORT), "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-GGUF", "ComfyUI-TensorFold-Image",
           "--extra-model-paths-config", str(extra), "--disable-auto-launch", "--database-url", "sqlite:///:memory:",
           "--user-directory", str(run_dir / "user"), "--output-directory", str(run_dir / "images"),
           "--temp-directory", str(run_dir / "temp"), "--input-directory", str(run_dir / "input"), *a.extra]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    t_launch = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(PORTABLE))
    try:
        while True:
            try: call("/system_stats"); break
            except Exception:
                if proc.poll() is not None: raise SystemExit("ComfyUI exited; see " + str(run_dir / "comfy.log"))
                time.sleep(0.5)
        t_server = time.perf_counter() - t_launch
        vs = VramSampler(); vs.start()
        times = []
        for i in range(a.repeat):
            g = graph(a)
            g["7"]["inputs"]["seed"] = a.seed  # fixed seed; re-encode skipped by ComfyUI cache after run 1
            if i > 0:  # force re-execution of sampler+decode (not the loaders/TE) by perturbing a no-op input
                g["9"]["inputs"]["filename_prefix"] = f"{a.tag}_r{i}"
                g["7"]["inputs"]["denoise"] = 1.0 - 1e-9 * i
            times.append(run_once(g))
            print(f"run {i}: {times[-1]:.2f} s", flush=True)
        job_times = []
        if a.jobs:
            for j, (prompt, seed) in enumerate(json.load(open(a.jobs, encoding="utf-8"))):
                a2 = argparse.Namespace(**{**vars(a), "prompt": prompt, "seed": seed})
                g = graph(a2)
                g["9"]["inputs"]["filename_prefix"] = f"job{j:02d}"
                job_times.append(run_once(g))
                print(f"job {j}: {job_times[-1]:.2f} s", flush=True)
        vs.stop = True; time.sleep(0.3)
    finally:
        proc.terminate(); proc.wait(timeout=30); log.close()
    text = (run_dir / "comfy.log").read_text(encoding="utf-8", errors="replace")
    engine_ran = "[tfimage]" in text and "gguf qtypes" not in text
    if a.tf and not engine_ran:
        raise SystemExit(f"--tf {a.tf}: the log shows no tfimage engine (or a GGUF load); results are not the engine's")
    its = re.findall(r"(\d+)/(\d+) \[([0-9:]+)<[^,]*,\s+([0-9.]+)(s/it|it/s)\]", text)
    rate = [(float(r[3]) if r[4] == "s/it" else 1 / float(r[3])) for r in its if r[0] == r[1]]
    res = {"tag": a.tag, "w": a.width, "h": a.height, "steps": a.steps, "unet": a.unet, "clip": a.clip,
           "server_start_s": round(t_server, 2), "runs_s": [round(t, 2) for t in times], "sampler_s_per_it": rate,
           "job_s": [round(t, 2) for t in job_times], "tf": a.tf, "engine_ran": engine_ran, "vram_idle_mib": idle, "vram_peak_mib": vs.peak, "vram_delta_mib": vs.peak - idle,
           "executed": re.findall(r"Prompt executed in ([0-9.]+) seconds", text)}
    (run_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
