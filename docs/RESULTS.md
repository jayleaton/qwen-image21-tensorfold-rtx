# Results: baseline, profile, levers, milestone 1, plan

Measured 2 Oct 2026 on one desktop: RTX 5070 Ti 16 GB (Blackwell, sm_120), NVIDIA driver 617.14, Windows 11, through
ComfyUI 0.37.0 portable (torch 2.10.0+cu130) with the harness in `bench/`. Settings throughout: 25 steps, euler,
simple, CFG 1, Qwen3-VL 8B int8_convrot text encoder, bf16 VAE, QwenImage21Cache on.

## 1. Baseline (ComfyUI + ComfyUI-GGUF, Q6_K)

`bench/comfy_baseline.py`: a headless ComfyUI with only ComfyUI-GGUF loaded and every directory redirected into
`runs/`; s/image is wall time from queueing to the saved PNG.

| Setting | s / image (warm) | Sampler | VRAM peak (device) |
| --- | ---: | --- | --- |
| 1024x1024 | **33.0** | 1.22-1.39 s/it | 15.8 GB (14.2 GB over idle) |
| 992x1216 | **35.7** | | 15.1 GB |
| 480x608 | **16.1** | | |
| 1024x1024, 1 step (prompt cached) | 2.4 | | |

Cold first image (nothing in the OS file cache): 104 s on the test machine, most of it reading ~16 GB of weights from
an external drive; cold time is a property of the disk, not the engine.

## 2. Profile (one warm 1024x1024 step under ComfyUI)

Qwen-Image 2.1's DiT: 32 single-stream blocks, dim 4096, 32 heads x 128, SwiGLU 12288, one modulation shared by
every block; ~7.1B parameters. A 1024x1024 image is 4,096 latent tokens (VAE /16, 64 channels, no patchify) plus
the prompt's ~25-70 text tokens. A step is 2 x 7.1B x 4,160 = 57 TFLOP of linears plus 8.9 TFLOP of attention:
**compute bound**, unlike LLM decode, so the format's tensor-core rate matters more than bytes read.

`bench/comfy_profile.py` (ComfyUI's own code as a library, torch.profiler):

| Component | ComfyUI (Q6_K GGUF) |
| --- | --- |
| Whole step | 1.07 s (library) / 1.25-1.39 s (server) |
| Q6_K -> bf16 decode (eager torch bit ops) | ~65% of GPU kernel time, partly overlapped on a side stream |
| bf16 GEMMs (cutlass) | ~0.6 s |
| Attention (cuDNN flash) | 0.10 s |
| Text encoder | 4.2 s first prompt, 0.26 s warm |
| VAE decode | ~1 s |

Tensor-core rates measured on the DiT's GEMM shapes (`tools/gemm_ceiling.py`, `tools/gemm_tf.py`; M = 4,160 rows):

| Projection (K x N) | cuBLAS bf16 | cuBLASLt FP8 | **TensorFold NVFP4 W4A4** |
| --- | ---: | ---: | ---: |
| qkv (4096 x 12288) | 95 TF/s | 161 | **422** |
| out (4096 x 4096) | 80 | 145 | **365** |
| gate+up (4096 x 24576) | 79 | 157 | **519** |
| down (12288 x 4096) | 81 | 202 | **379** |

## 3. Levers (measured unless marked)

| Lever | Effect on a 1024x1024 step |
| --- | --- |
| NVFP4 W4A4 linears (TensorFold prompt GEMMs, `gemm_ws` bulk-copy tiles) | linears 0.6 s -> 0.124 s, no per-step weight decode |
| FP8 W8A8 linears (cuBLASLt, tensor-wise scales) | 0.67 s per step in ComfyUI (1.9x) |
| Fused adaLN, RMSNorm+RoPE (writes K straight into the attention buffer), SwiGLU (Triton) | ~400 ms of fp32 temporaries -> ~13 ms |
| TensorFold `mlp_prompt`: gate/up GEMM with a SwiGLU epilogue that writes FP4 rows for down | no [4160, 24576] bf16 round trip |
| cuDNN attention forced first (`sdpa_kernel(..., set_priority=True)`) | 293 ms (PyTorch picked mem-efficient) -> 101 ms |
| Static input scales from a calibration run (4x margin) | no host sync per linear: 0.47 -> 0.30 s together with cuDNN |
| Prefix K/V kept per prompt (LRU of 4) | text / reference tokens run once per prompt, reused across seeds |
| Converted engine cached on disk | load ~2 s instead of decoding and quantizing the GGUF |
| CUDA graphs | not done: ~640 launches a step, no syncs, GPU-bound, so launches are hidden (expected < 3%) |
| VAE tiling | not needed at <= 2 MP on 16 GB |
| Step reduction (distilled / lightning LoRA, CFG skipping, TeaCache) | no Qwen-Image 2.1 distillation was available; CFG 1 already skips the negative pass |

Existing alternatives on Windows: ComfyUI already runs FP8 / int8 tensor-core matmuls through comfy-kitchen, and
Comfy-Org publishes `qwen_image_2.1_int8_convrot`; that route is likely FP8-class (~2x) with no code (not measured
here). Nunchaku (SVDQuant W4A4) has no Qwen-Image 2.1 model. TensorRT needs an ONNX export of a model with a
block-causal prefix and custom RoPE.

## 4. Milestone 1: Qwen-Image 2.1 end to end as a ComfyUI node

| Size (warm, same ComfyUI) | GGUF Q6_K | **NVFP4** | Speedup | FP8 |
| --- | ---: | ---: | ---: | ---: |
| 1024x1024 | 33.0 s | **8.0-8.4 s** | **4.0x** | 16.7-17.8 s |
| 992x1216 | 35.7 s | **9.4 s** | **3.8x** | |
| 480x608 | 16.1 s | **1.9 s** | **8.4x** | |

Engine step at 1024x1024 (`bench/engine_step.py --profile`): **0.29-0.30 s**; kernel time: attention 101 ms, NVFP4
GEMMs 124 ms, everything else ~25 ms. NVFP4 linears 3.66 GiB; engine peak 4.3 GiB. The small size gains most because
ComfyUI's per-step GGUF decode does not shrink with the image.

Correctness: the bf16 path equals ComfyUI's forward on the same weights to bf16 noise (relative L2 0.005, cosine
0.99996), text-to-image and with a reference latent. NVFP4 single-step error vs that forward: 7.3% relative L2
(FP8: 2.0%); spread evenly over projections and blocks (no single layer dominates).

Quality gate: see the README's [Quality](../README.md#quality) section (LPIPS pass, DINO fail for both FP8 and NVFP4,
driven by two layout-sensitive prompts).

## 5. Plan

| Milestone | Work | Expected (from the profile) |
| --- | --- | --- |
| M1 (done) | NVFP4 DiT in ComfyUI, text-to-image and edit forward | 3.8-4.0x at 1 MP, 8.4x at 0.3 MP |
| M2 fidelity | per-layer error budget; FP8 for the most sensitive blocks (`nvfp4:edge=N`); SVDQuant-style low-rank residual (rank 32, ~8% GEMM cost); a 20+ prompt gate; edit workflow timed end to end | DINO >= 0.95 at >= 3.5x |
| M3 attention | INT8 / FP8 attention (SageAttention-class) on sm_120, behind the same gate | 101 -> ~50 ms: ~0.25 s a step, ~4.6x |
| M4 more DiTs | Qwen-Image 2512 (dual-stream MMDiT), FLUX.2-dev, Z-Image Turbo, FLUX.1 Kontext: same linears and kernels, one model module each | 3-4x on each DiT |
| M5 LoRA / ControlNet | merge LoRA into bf16 weights then quantize (cached per LoRA set); ControlNet as a second engine | removes today's refusals |
| M6 standalone | an OpenAI-images-style server on the same engine | optional |

VRAM budget on 16 GB: NVFP4 DiT 3.7 GB resident + ~0.6 GB step working set at 1 MP (~2.4 GB at 2048x2048) + the int8
text encoder 9.4 GB (ComfyUI evicts it before sampling: the node asks for room on the first step) + VAE 0.7 GB.
