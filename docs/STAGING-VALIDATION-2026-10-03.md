# home-desk staging validation, 2026-10-03

RTX 5070 Ti, 16 GB, driver 617.14, Windows 11. Tests used the staging hub
on dev-box:18420 and gateway on dev-box:18890 exclusively after correcting
the isolated-settings incident below. Production is not approved by this report.

The delivered server uses the headless compatibility fallback described in
[OPENAI-SERVER.md](OPENAI-SERVER.md). It runs TensorFold NVFP4, Qwen3-VL
int8_convrot and Qwen-Image 2.1 bf16 VAE with Euler/simple, CFG 1, 25 steps.
It is not yet ComfyUI-free. All measurements are wall-clock on this machine;
VRAM is device-wide and includes desktop/other applications, not just the model.

| Test | Result | Measurement / evidence |
| --- | --- | --- |
| Native tray build | Pass | Rust/MSVC release build; 183 Rust tests, 354 JS tests, type-check and canvas build passed |
| Machines catalog/dropdown | Pass | home-desk exposed image and LM Studio models; Switching then Running observed |
| None to image | Pass | 20.283 s; 14,736 MiB device VRAM |
| Image to Qwen3.8 27B | Pass | 85.516 s; 15,530 MiB; load command itself 81.93 s |
| LM Studio to None | Pass | 3.997 s; Nothing running; no loaded LMS models; 3,959 MiB |
| Gateway image to chat, cold/warm | Pass | HTTP 200, 29.107 / 2.236 s; 15,523 / 15,543 MiB |
| Chat streaming | Pass | HTTP 200, 2.274 s, SSE completed with [DONE] |
| Gateway chat to image, cold/warm | Pass | HTTP 200, 25.482 / 7.153 s, 1024 square; 13,819 / 13,392 MiB |
| Canvas Local image generation | Pass | dev-box chat, --ar 16:9 --seed 42; 1360x768, 7.9 s; browser decoded image |
| Canvas Edit plus new prompt | Pass (transport) | 1 reference, 1360x768, 13.8 s; both result images decoded in chat |
| Multipart image[] API edit | Pass (transport) | HTTP 200, 23.332 s, 1024 square; weak visual prompt adherence observed |
| n=1..4 | Pass | decoded/count checked; n=2 1.972 s, n=3 1.993 s at 256 square/25 steps; n=4 0.858 s at 1 step |
| Simultaneous requests | Pass | two 256 square/25-step jobs serialized; completions 0.672 and 1.232 s, total 1.237 s |
| API errors | Pass | n=5 and invalid size 400; unknown gateway model 404; invalid multipart image 400; OpenAI error objects |
| Bad catalog command | Pass | UI Failed: Starting ... failed: The system cannot find the file specified. (os error 2) |
| Abrupt staging tray kill | Pass | tray PID 320944, API 284984, backend 12532 all gone; port 8189 closed; VRAM 13,760 to 3,079 MiB |
| Authenticated standalone API | Pass | actual missing/wrong bearer key 401, correct key /models 200; owned backend cleaned up |
| Authenticated readiness | Pass | Rust live TCP test exercised upstream 401/403, matching/empty/wrong model IDs and 500 |
| Gateway denial from non-owner identity | Not tested | no separate unauthorized tailnet identity available; do not infer this from readiness tests |
| imageGen skill exact curl generation example | Pass | staging gateway, fox prompt, seed 42, 1360x768, HTTP 200 in 7.867 s |

Initial swap timings under GPU contention were chat cold 177.66 s, chat warm
82.93 s, stream 12.35 s and image cold 57.63 s. An existing idle user ComfyUI
process was stopped only after confirming its exact command line. The clean
repeat above used the same catalog settings. Other cold starts ranged to
61.356 s while compilation and memory loading competed for resources.

## Actual API quality gate

Six existing gate prompts, 1024x1024, seed/steps matched to Q6_K ComfyUI baseline.
API times: 9.185, 9.529, 9.195, 8.927, 9.208, 9.064 s; mean **9.185 s/image**.
Baseline warmed repeated prompt: 36.41 and 34.54 s; first image 106.43 s.
Baseline gate prompt times: 35.58, 40.02, 52.37, 39.34, 39.99, 38.54 s.

| Prompt | LPIPS (lower better) | DINO (higher better) | PSNR |
| --- | --- | --- | --- |
| 00 | 0.1966 | 0.9711 | 14.65 |
| 01 | 0.2935 | 0.8433 | 17.48 |
| 02 | 0.3841 | 0.7740 | 13.44 |
| 03 | 0.2428 | 0.9699 | 18.82 |
| 04 | 0.0872 | 0.9622 | 24.59 |
| 05 | 0.1618 | 0.9420 | 20.48 |
| Mean | **0.2277 PASS (<=0.25)** | **0.9104 FAIL (>=0.95)** | 18.2433 |

This quality failure blocks production approval. Same prompt/seed repeated through
the API produced pixel-identical output (PNG metadata/file hashes differ).
Generation and edit artifacts were deleted after validation; numeric evidence
is retained here. Converted/calibrated weights remain in LOCALAPPDATA/tfimage
because the deployed server needs them.

## Issues fixed and deployment notes

- ComfyUI's empty history-delete response was mistakenly parsed as JSON, failing
  cold startup after warmup. Empty successful bodies now return an empty object.
- Windows embedded Python excludes the script folder from sys.path; the server
  explicitly adds the repository root.
- Cached sampler nodes could skip repeated API requests; a latent passthrough
  invalidates sampling while keeping loader caches and exact noise/schedule.
- Backend startup occurs suspended inside a kill-on-close Windows job, preventing
  descendants from escaping both standalone API and staging tray lifecycle.
- LM Studio's /models also lists downloaded models: use the distinct loaded alias
  qwen3.8-27b and require it for readiness. ELECTRON_RUN_AS_NODE must be absent
  when starting LM Studio/tray from this harness. Catalog loads with 4096 context
  and 0.65 GPU offload to fit this card.
- An initial staging settings file contained a UTF-8 BOM. The old tray silently
  rejected it and fell back to production defaults, briefly connecting to the
  production hub and reporting a device profile. It was immediately stopped.
  The installed production tray/settings were never changed. This violates the
  requested production boundary; no claim of zero production contact is made.
  Agent Storage now accepts BOM and fails closed for invalid/missing isolated
  settings, with focused regression tests.

The final staging tray uses AGENT_STORAGE_TRAY_HOME under LOCALAPPDATA,
staging-only hub, all requested remote/clipboard/drop flags false, metrics true,
and no launch-at-login. Its catalog owns the running image server. The server
binds 0.0.0.0:8189 and its backend binds loopback:8198; existing firewall access
is Private only, with no Public rule added. LM Studio serves :1234 with no chat
model loaded while image is active. The bad-command entry was removed.

To stop: choose None on home-desk in the staging Machines tab, then quit the
staging tray. Alternatively stop only its recorded PID; its job removes the API
and backend. Do not stop the installed production tray by process name.
