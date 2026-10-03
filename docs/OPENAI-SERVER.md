# OpenAI image server

`serve.cmd` exposes `GET /v1/models`, JSON `POST /v1/images/generations`, and multipart
`POST /v1/images/edits` on `0.0.0.0:8189`. Set `COMFYUI_PORTABLE` first. The portable
Python already contains Pillow and the GPU dependencies; no API framework is needed.

```bat
set COMFYUI_PORTABLE=E:\AI\ComfyUI_windows_portable
serve.cmd
```

The current backend is the **headless compatibility fallback**, not a ComfyUI-free
implementation. TensorFold runs the DiT in NVFP4. ComfyUI loads the local Qwen3-VL
int8_convrot encoder and Qwen-Image 2.1 bf16 VAE, and supplies euler/simple sampling
with CFG 1. It listens only on loopback port 8198, has an in-memory database, and
redirects all writable directories to `runs/server`. The user's ComfyUI installation
is read only. The API creates and warms all model objects before publishing readiness.
Loaders remain cached; the encoder moves between CPU and GPU to fit 16 GB VRAM.
Sampling is invalidated with a passthrough latent node, preserving exact seeds and
schedules. Requests serialize on one GPU lock. Each requested image uses seed + index.

`--host`, `--port`, `--backend-port`, `--work-dir`, `--precision nvfp4|fp8`, `--unet`,
and `--clip` override defaults. Cold starts allow up to 900 seconds; each render
allows 1800 seconds. Only one API instance may use a given backend port/work folder.
On Windows the backend starts suspended, enters a kill-on-close job, then resumes.
It also inherits the tray's job: stopping or killing the tray kills the API and backend.
Ctrl+C stops a standalone API cleanly. Killing its PID closes its job and frees GPU
allocations too. Input/output PNGs and prompt history are removed after each request.

## Requests

Fields: `model` (default `qwen-image-2.1`), nonempty `prompt`, `n` (1..4), `size`
(`auto`, default 1024x1024, or WxH with multiples of 16 in 256..2048 and at most
2,097,152 pixels), `response_format` (`b64_json` only), `seed` (unsigned 64-bit),
and `steps` (1..100, default 25). PNG bytes are returned in OpenAI's
`{"created": timestamp, "data": [{"b64_json": "..."}]}` shape.

Edits accept 1..16 `image` or `image[]` files and the same fields. Reference images
are resized by the Qwen 2.1 encoder, sent into its vision path, VAE-encoded, and
attached as reference latents. The requested size controls the output canvas;
`auto` uses 1024x1024. Masks are refused. Bodies are limited to 64 MiB and references
to 16 megapixels. Images are decoded locally, not forwarded as user file paths.

Errors use `{"error":{"message":"...","type":"...","param":null,"code":null}}`.
Validation errors return 400, unknown endpoints 404, body limits 413, backend errors
500, and dead-backend readiness 503. The API is unauthenticated by default: keep its
firewall access restricted to the private tailnet. Set `QWEN_IMAGE_API_KEY` to require
a bearer key; missing/incorrect keys return 401. The tray treats 401 and 403 as an
alive authenticated upstream; the gateway enforces its own Tailscale identity check,
drops caller authorization, and supplies its configured upstream key.

## Quality verification

Generate calibration contexts and calibrate once as documented in AGENTS.md, then:

```bat
"%COMFYUI_PORTABLE%\python_embeded\python.exe" -B bench\comfy_baseline.py --tag staging_base --repeat 3 --jobs bench\jobs_gate.json
rem Stop the baseline first; run the API via your staging tray.
"%COMFYUI_PORTABLE%\python_embeded\python.exe" -B bench\api_gate.py --url http://dev-box.prawn-penny.ts.net:18890/v1 --tag staging_api
tools\env.cmd python bench\quality.py staging_base staging_api
```

## Work remaining for ComfyUI-free execution

1. Port the `qwen3vl_8b_int8_convrot` checkpoint loader and quantized linear/embedding
   operations (including rotations/scales) to standalone encoder code. These local
   weights are not a Hugging Face `from_pretrained` directory.
2. Reproduce Qwen 2.1 chat templates, token/image slot assembly, vision preprocessing,
   hidden-state selection and prefix layout; compare dumped embeddings for all gate
   prompts and multiple-reference edits against `TextEncodeQwenImage21`.
3. Load the 2.1 VAE checkpoint in standalone code with its channel/layout mapping,
   latent scaling and encode/decode precision. Verify identical reference latents
   and decoded images, including alpha handling and non-square sizes.
4. Port the exact FLUX/Qwen timestep shift, simple sigma schedule, seeded noise,
   latent packing and Euler update. Validate bf16-forward parity before quantization.
5. Replace the backend adapter only after the actual API gate (generation and edit)
   passes. NVFP4's existing DINO quality gap remains a separate quantization issue;
   removing ComfyUI alone will not resolve it.
