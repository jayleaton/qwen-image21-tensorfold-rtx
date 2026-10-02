# TensorFold 0.6.1 on native Windows (RTX 5070 Ti, sm_120)

Measured 2 Oct 2026 on one desktop: Windows 11, NVIDIA driver 617.14 (CUDA UMD 13.4), Visual Studio Community 2026
(MSVC 14.51.36231), Python 3.13, torch 2.10.0+cu130, triton-windows 3.6.0.post26.

## Result

All nine of TensorFold's core CUDA extensions build for `sm_120a` under MSVC and nvcc, and TensorFold's GPU kernel
tests pass on the 5070 Ti:

| Extension | Build |
| --- | --- |
| tensorfold_nvfp4_ck_v6 (NVFP4/FP8 checkpoint math: quantizers, lane + prompt GEMMs, gemm_ws bulk-copy) | OK, 36 s |
| tensorfold_nvfp4_v3 (W4A16/W8A16 lane matmuls) | OK, 34 s |
| tensorfold_nvfp4_prompt_v1 | OK, 22 s |
| tensorfold_qmm_v5 (4-bit lane matmul, grouped sm_12x kernel, prefill8) | OK, 25 s |
| tensorfold_exl3_linear_v3 | OK, 29 s |
| tensorfold_prefill_attention_v1 | OK, 33 s |
| tensorfold_gdn_v2 | OK, 15 s |
| tensorfold_experts_v7 | OK, 40 s |
| tensorfold_exl3_experts_v1 | OK, 26 s |

`pytest tests/cuda/{nvfp4_checkpoint,nvfp4_linear,nvfp4_lane_tiles,qmm,qmm_group,exl3_linear,prefill_attention,
attention,gdn,build,group_kernel_smem}`: **214 passed, 2 skipped, 1 failed**. The failure is the test harness, not a
kernel: `tests/cuda/test_build.py::test_a_lock_left_by_a_killed_build_...` calls `select.select()` on a subprocess
pipe, which Windows only supports for sockets (WinError 10038).

Not run: the OpenAI server end to end with a model. Every CUDA family's checkpoints exceed or crowd this card's
16 GB (27B NVFP4 ~15 GB of weights alone); an EXL3 ~3 bpw 27B (~10.5 GB) is the only candidate and was not
downloaded for this pass.

## What it took (and what TensorFold could fix)

1. **nvcc 13.0 crashes on MSVC 14.51 (VS 2026).** Even a trivial kernel: `cudafe++ died with status 0xC0000005`.
   NVIDIA's pip `nvidia-cuda-nvcc==13.4.*` (+ `nvidia-cuda-crt`, `nvidia-nvvm`, `nvidia-cuda-runtime` 13.4) with
   CCCL **13.0.85** headers works against torch's cu130 runtime. CCCL 13.3 stops at `#error` unless cl.exe gets
   `/Zc:preprocessor` (torch's extension flags use the traditional preprocessor), so keep CCCL at 13.0 or add that
   flag. Worth a line in the Windows runbook: VS 2026 needs CUDA >= 13.4.
2. **`build.pip_toolkit` does not set up the pip toolkit on Windows.** On `os.name == "nt"` it only adds DLL
   folders: it never sets `CUDA_HOME` or puts `nvcc` on PATH (the Linux branch does), and it looks for
   `cudart.lib` in `<cu13>/lib` while NVIDIA's Windows wheel puts it in `<cu13>/lib/x64`. Here `CUDA_HOME` was set by
   hand (`tools/env.cmd`, `scripts/setup.cmd`).
3. **ATen's CUDA headers need the math libraries' headers.** `ATen/cuda/CUDAContextLight.h` includes
   `cusparse.h` (and cuBLAS/cuSOLVER): the pip route needs `nvidia-cusparse`, `nvidia-cublas`, `nvidia-cusolver`,
   `nvidia-curand`, `nvidia-cufft` too. The RTX "pip alone" recipe should list them for Windows.
4. **MSVC environment.** Builds need `vcvars64.bat` (cl.exe, link.exe, the Windows SDK); `vswhere.exe` is expected
   on PATH by vcvars (`C:\Program Files (x86)\Microsoft Visual Studio\Installer`).
5. **Prebuilt modules load in another Python with the same torch.** The `.pyd` files built in this repo's venv
   import into ComfyUI's portable Python (same torch 2.10+cu130, cp313) with no compiler there
   (`tfimage/ext.py`).


## Reuse for image generation

| TensorFold piece | Used by tfimage | Notes |
| --- | --- | --- |
| `nvfp4.checkpoint.quant4` (bf16 rows -> NVFP4, per-16 e4m3 scales) | yes: weights and activations | also produces the `gemm_ws` tiled layout |
| `nvfp4.checkpoint.prompt` (FP4 x FP4 block-scaled MMA, bulk-copy warp-specialised tiles) | yes: every DiT projection | 365-519 TF/s at 4,160 rows vs cuBLAS bf16 80-95 |
| `nvfp4.linear.Fp4Linear.from_checkpoint` | yes | ModelOpt NVFP4 weight layout |
| `nvfp4.checkpoint.mlp_prompt` (gate/up GEMM with SwiGLU epilogue -> FP4 rows) | not yet | next fusion |
| `exl3.linear`, `qmm` lane matmuls (W4A16 / 1-128 rows) | no | decode-shaped; a DiT step is compute bound |
| `prefill_attention` | no | causal LLM prefill layout; DiT attention is bidirectional over the target |
| server, scheduler, families, sampling | no | LLM-specific |

New for images: the DiT family itself (single-stream MMDiT blocks, shared modulation, 3-axis RoPE, block-causal
text/reference prefix), flow-matching sampling glue, a GGUF reader (TensorFold reads no GGUF), activation-scale
calibration, the ComfyUI integration.
