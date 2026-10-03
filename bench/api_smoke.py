"""Real gateway smoke tests. Run after the tray dropdown cycle, with image active."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://dev-box.prawn-penny.ts.net:18890/v1")
ap.add_argument("--phase", choices=["swap", "images"], default="swap")
a = ap.parse_args()
out = ROOT / "runs/staging_smoke"
out.mkdir(parents=True, exist_ok=True)
results = []


def vram():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())


def request(name, path, body, content_type="application/json", expected=200):
    started = time.perf_counter()
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    payload = out / "request.tmp"
    payload.write_bytes(body)
    try:
        response = subprocess.run(["curl.exe" if os.name == "nt" else "curl", "-sS", "--max-time", "1800",
            "-H", "Content-Type: " + content_type, "-H", "Authorization: Bearer local", "--data-binary", "@" + str(payload),
            "-w", "\n%{http_code}", a.url + path], capture_output=True)
        if response.returncode:
            raise RuntimeError(response.stderr.decode(errors="replace"))
        raw, status = response.stdout.rsplit(b"\n", 1)
        status = int(status)
    finally:
        payload.unlink(missing_ok=True)
    result = dict(name=name, status=status, elapsed_s=time.perf_counter()-started, bytes=len(raw), vram_mib=vram())
    results.append(result)
    print(json.dumps(result), flush=True)
    (out / f"{a.phase}-results.json").write_text(json.dumps(results, indent=2))
    if status != expected:
        raise RuntimeError(raw.decode(errors="replace")[:4000])
    if path.endswith("completions") and b"data: " in raw:
        if b"[DONE]" not in raw:
            raise RuntimeError("Streaming response lacks [DONE]")
        (out / "chat-stream.txt").write_bytes(raw)
        return raw
    data = json.loads(raw)
    if "data" in data and data["data"] and "b64_json" in data["data"][0]:
        if name == "four-images" and len(data["data"]) != 4:
            raise RuntimeError("n=4 did not return four images")
        for i, item in enumerate(data["data"]):
            (out / f"{name}-{i}.png").write_bytes(base64.b64decode(item["b64_json"]))
    else:
        (out / f"{name}.json").write_text(json.dumps(data, indent=2))
    return data


chat = dict(model="qwen3.8-27b", messages=[dict(role="user", content="Reply with exactly: staging works")], max_tokens=48, temperature=0)
image = dict(model="qwen-image-2.1", prompt="A red fox in fresh snow at dawn, telephoto, soft backlight", n=1,
             size="1024x1024", response_format="b64_json", seed=42, steps=25)
if a.phase == "swap":
    request("chat-cold-swap", "/chat/completions", chat)
    request("chat-warm", "/chat/completions", chat)
    request("chat-stream", "/chat/completions", dict(chat, stream=True))
    request("image-cold-swap", "/images/generations", image)
    request("image-warm", "/images/generations", image)
else:
    request("four-images", "/images/generations", dict(image, n=4, steps=1, size="256x256"))
    request("invalid-n", "/images/generations", dict(image, n=5), expected=400)
    request("invalid-size", "/images/generations", dict(image, size="1025x1024"), expected=400)
    request("unknown-model", "/images/generations", dict(image, model="staging-no-such-model"), expected=404)
    png = (out / "image-warm-0.png").read_bytes()
    boundary = "staging-edit-boundary"
    form = bytearray()
    for key, value in dict(model="qwen-image-2.1", prompt="Make the fox blue and add a full moon at night", seed=42, steps=25,
                           size="1024x1024", response_format="b64_json").items():
        form.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    form.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="image[]"; filename="fox.png"\r\nContent-Type: image/png\r\n\r\n'.encode())
    form.extend(png)
    form.extend(f"\r\n--{boundary}--\r\n".encode())
    request("reference-edit", "/images/edits", bytes(form), f"multipart/form-data; boundary={boundary}")
