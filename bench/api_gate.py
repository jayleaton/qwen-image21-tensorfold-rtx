"""Render the fixed quality gate through the actual OpenAI API, not backend nodes."""
import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8189/v1")
ap.add_argument("--tag", default="staging_api")
a = ap.parse_args()
out = ROOT / "runs" / a.tag
(out / "images").mkdir(parents=True, exist_ok=True)
times = []
deadline = time.monotonic() + 900
while True:
    try:
        with urllib.request.urlopen(a.url + "/models", timeout=5):
            break
    except OSError:
        if time.monotonic() > deadline:
            raise TimeoutError("API did not become ready")
        time.sleep(1)
for i, (prompt, seed) in enumerate(json.loads((ROOT / "bench/jobs_gate.json").read_text())):
    payload = dict(model="qwen-image-2.1", prompt=prompt, seed=seed, steps=25,
                   size="1024x1024", response_format="b64_json", n=1)
    started = time.perf_counter()
    req = urllib.request.Request(a.url + "/images/generations", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    times.append(elapsed)
    (out / "images" / f"job{i:02d}_api.png").write_bytes(base64.b64decode(result["data"][0]["b64_json"]))
    print(f"job {i}: {elapsed:.2f} s", flush=True)
(out / "result.json").write_text(json.dumps({"url": a.url, "job_s": times,
    "mean_s": sum(times) / len(times), "steps": 25, "size": "1024x1024"}, indent=2))
