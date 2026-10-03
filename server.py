"""OpenAI image API with an owned, isolated TensorFold/ComfyUI backend.

Only the public API is exposed. The compatibility backend is loopback-only,
uses no user's state, and dies with this process (including abrupt Windows exit).
"""
from __future__ import annotations

import argparse
import base64
import email.parser
import email.policy
import io
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # Windows embedded Python's ._pth omits the script directory.
MODEL = "qwen-image-2.1"
LOCK = threading.Lock()
MAX_BODY = 64 * 1024 * 1024


def validate(fields):
    if fields.get("model", MODEL) != MODEL:
        raise ValueError("Unknown model; use qwen-image-2.1")
    prompt = fields.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
        raise ValueError("prompt must be nonempty text of at most 32000 characters")
    if fields.get("response_format", "b64_json") != "b64_json":
        raise ValueError("Only response_format=b64_json is supported")
    def integer(name, default, low, high):
        value = fields.get(name, default)
        if isinstance(value, bool) or isinstance(value, float):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value
    n = integer("n", 1, 1, 4)
    steps = integer("steps", 25, 1, 100)
    seed = integer("seed", secrets.randbits(63), 0, 2**64 - 1)
    size = fields.get("size", "auto")
    if size == "auto":
        w, h = 1024, 1024
    else:
        try:
            w, h = map(int, size.lower().split("x"))
        except (ValueError, AttributeError):
            raise ValueError("size must be auto or WxH") from None
        if not (256 <= w <= 2048 and 256 <= h <= 2048 and w % 16 == h % 16 == 0 and w*h <= 2097152):
            raise ValueError("size dimensions must be multiples of 16 in 256..2048, at most 2097152 pixels")
    return dict(prompt=prompt, n=n, steps=steps, seed=seed, width=w, height=h)


class Backend:
    def __init__(self, args):
        self.args = args
        self.url = f"http://127.0.0.1:{args.backend_port}"
        self.directory = Path(args.work_dir).resolve()
        for name in ("user", "input", "output", "temp"):
            (self.directory / name).mkdir(parents=True, exist_ok=True)
        self.job = None
        self.proc = None

    def call(self, path, data=None):
        req = urllib.request.Request(self.url + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}

    def start(self):
        portable = Path(self.args.comfyui)
        extra = self.directory / "extra_paths.yaml"
        extra.write_text(f"tfimage:\n  base_path: {(ROOT / 'comfyui').as_posix()}\n  custom_nodes: .\n", encoding="utf-8")
        cmd = [str(portable / "python_embeded/python.exe"), "-s", str(portable / "ComfyUI/main.py"),
            "--windows-standalone-build", "--listen", "127.0.0.1", "--port", str(self.args.backend_port),
            "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-GGUF", "ComfyUI-TensorFold-Image",
            "--extra-model-paths-config", str(extra), "--disable-auto-launch", "--database-url", "sqlite:///:memory:"]
        for option, directory in (("user", "user"), ("input", "input"), ("output", "output"), ("temp", "temp")):
            cmd += [f"--{option}-directory", str(self.directory / directory)]
        self.log = open(self.directory / "backend.log", "a", encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        if os.name == "nt":
            # Nest inside the tray's job. Closing either job kills the backend.
            from server_job import WindowsJob
            self.job = WindowsJob()
        self.proc = subprocess.Popen(cmd, cwd=portable, env=env, stdout=self.log, stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NO_WINDOW | 0x4) if os.name == "nt" else 0)
        if self.job:
            self.job.assign(self.proc)
        deadline = time.monotonic() + 900
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Backend exited ({self.proc.returncode}); see {self.directory / 'backend.log'}")
            try:
                self.call("/system_stats")
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Backend startup timed out")
                time.sleep(.5)
        # Load every model before publishing readiness. Keep loaders cached;
        # ComfyUI moves the encoder between CPU/GPU to fit the 16 GB card.
        self.render(dict(prompt="A red apple", n=1, seed=1, steps=1, width=256, height=256), [])

    def graph(self, fields, references, prefix):
        g = {
            "1": {"class_type": "TFQwenImage21Loader", "inputs": {"unet_name": self.args.unet, "precision": self.args.precision}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": self.args.clip, "type": "qwen_image", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
            "4": {"class_type": "QwenImage21Cache", "inputs": {"device": "auto", "dtype": "default", "model": ["1", 0]}},
            "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": fields["prompt"], "negative_prompt": "", "resolution": 1024, "clip": ["2", 0]}},
            "6": {"class_type": "EmptyLatentImage", "inputs": {"width": fields["width"], "height": fields["height"], "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {"seed": fields["seed"], "steps": fields["steps"], "cfg": 1., "sampler_name": "euler", "scheduler": "simple", "denoise": 1., "model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["6", 0]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
            "10": {"class_type": "TFImageAPILatent", "inputs": {"latent": ["6", 0]}},
        }
        g["7"]["inputs"]["latent_image"] = ["10", 0]
        if references:
            g["5"]["inputs"]["vae"] = ["3", 0]
            for i, name in enumerate(references, 1):
                node = str(10+i)
                g[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
                g["5"]["inputs"][f"images.image_{i}"] = [node, 0]
        return g

    def render(self, fields, uploads):
        from PIL import Image
        prefix = "api_" + secrets.token_hex(12)
        references, results, outputs = [], [], []
        try:
            for i, raw in enumerate(uploads):
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width * im.height > 16777216:
                        raise ValueError("Reference image exceeds 16 megapixels")
                    name = f"{prefix}_ref{i}.png"
                    im.convert("RGBA").save(self.directory / "input" / name)
                    references.append(name)
            for i in range(fields["n"]):
                f = dict(fields, seed=(fields["seed"] + i) % 2**64)
                graph = self.graph(f, references, prefix + f"_{i}")
                pid = self.call("/prompt", {"prompt": graph})["prompt_id"]
                deadline = time.monotonic() + 1800
                while True:
                    if self.proc.poll() is not None:
                        raise RuntimeError("Backend exited while rendering")
                    history = self.call(f"/history/{pid}").get(pid)
                    if history:
                        if history["status"].get("status_str") != "success":
                            raise RuntimeError(json.dumps(history["status"])[:4000])
                        for item in history["outputs"]["9"]["images"]:
                            path = (self.directory / "output" / item["subfolder"] / item["filename"]).resolve()
                            if not path.is_relative_to(self.directory / "output"):
                                raise RuntimeError("Invalid backend output path")
                            outputs.append(path)
                            results.append({"b64_json": base64.b64encode(path.read_bytes()).decode()})
                        self.call("/history", {"delete": [pid]})
                        break
                    if time.monotonic() > deadline:
                        self.call("/interrupt", {})
                        raise TimeoutError("Image job exceeded 30 minutes")
                    time.sleep(.1)
            return {"created": int(time.time()), "data": results}
        finally:
            for path in outputs:
                path.unlink(missing_ok=True)
            for name in references:
                (self.directory / "input" / name).unlink(missing_ok=True)
            for path in (self.directory / "output").glob(prefix + "*.png"):
                path.unlink(missing_ok=True)

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.job:
            self.job.close()
        if hasattr(self, "log"):
            self.log.close()


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def error(self, status, message, kind="invalid_request_error"):
        self.reply(status, {"error": {"message": message, "type": kind, "param": None, "code": None}})

    def authorized(self):
        key = os.environ.get("QWEN_IMAGE_API_KEY")
        if key and not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key):
            self.error(401, "Invalid or missing API key", "authentication_error")
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        if self.path == "/v1/models":
            if self.server.backend.proc.poll() is not None:
                self.error(503, "Image backend is unavailable", "server_error")
            else:
                self.reply(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "local"}]})
        else:
            self.error(404, "Unknown endpoint")

    def do_POST(self):
        if not self.authorized():
            return
        if self.path not in ("/v1/images/generations", "/v1/images/edits"):
            self.error(404, "Unknown endpoint")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.error(413, "Request must be 1..64 MiB")
                return
            self.connection.settimeout(60)
            raw = self.rfile.read(length)
            uploads = []
            if self.path.endswith("/edits"):
                content_type = self.headers.get("Content-Type", "")
                if not content_type.startswith("multipart/form-data"):
                    raise ValueError("edits requires multipart/form-data")
                msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(
                    b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + raw)
                fields = {}
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    data = part.get_payload(decode=True)
                    if name in ("image", "image[]"):
                        uploads.append(data)
                    elif name == "mask":
                        raise ValueError("Masks are not supported; describe the edit in prompt")
                    elif name:
                        fields[name] = data.decode("utf-8")
                if not 1 <= len(uploads) <= 16:
                    raise ValueError("edits requires 1..16 image or image[] files")
            else:
                fields = json.loads(raw)
                if not isinstance(fields, dict):
                    raise ValueError("JSON body must be an object")
            fields = validate(fields)
            with LOCK:
                result = self.server.backend.render(fields, uploads)
            self.reply(200, result)
        except (ValueError, TypeError, UnicodeError, UnidentifiedImageError) as exc:
            self.error(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.log_error("render failed: %s", exc)
            self.error(500, str(exc), "server_error")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--backend-port", type=int, default=8198)
    ap.add_argument("--comfyui", default=os.environ.get("COMFYUI_PORTABLE"), required=not os.environ.get("COMFYUI_PORTABLE"))
    ap.add_argument("--work-dir", default=str(ROOT / "runs/server"))
    ap.add_argument("--precision", default="nvfp4", choices=["nvfp4", "fp8"])
    ap.add_argument("--unet", default="qwen-image-2.1-Q6_K.gguf")
    ap.add_argument("--clip", default="qwen3vl_8b_int8_convrot.safetensors")
    args = ap.parse_args()
    backend = Backend(args)
    server = None
    try:
        backend.start()
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        server.backend = backend
        def stop(signum, frame):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        print(f"READY {MODEL} http://{args.host}:{args.port}/v1 (TensorFold {args.precision}, headless compatibility backend)", flush=True)
        server.serve_forever()
    finally:
        if server:
            server.server_close()
        backend.close()


if __name__ == "__main__":
    main()
