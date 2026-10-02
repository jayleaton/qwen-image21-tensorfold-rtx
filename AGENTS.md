# AGENTS.md: setting this up on a Windows PC with an RTX 50-series GPU

Instructions for AI coding agents (and people) bringing up the TensorFold Qwen-Image 2.1 loader in an existing
ComfyUI. Follow the steps in order; run each check and stop on a failure. Ask the user before anything that changes
state outside this repo (installing Visual Studio, linking into their ComfyUI, deleting files).
Never modify the user's ComfyUI install: the benchmarks run it headless with every directory redirected into `runs/`
and an in-memory database.

## 1. Check the machine

| Check | Command | Expect |
| --- | --- | --- |
| GPU | `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader` | compute capability `12.0` (RTX 50-series); 617.x is known good |
| ComfyUI portable | `"%COMFYUI_PORTABLE%\python_embeded\python.exe" -c "import torch;print(torch.__version__)"` | `2.10.0+cu130` (the prebuilt kernels must match torch and Python 3.13) |
| ComfyUI-GGUF | `dir "%COMFYUI_PORTABLE%\ComfyUI\custom_nodes\ComfyUI-GGUF"` | present (the baseline and `.gguf` file lists use it) |
| Models | `diffusion_models\*.gguf` (Qwen-Image 2.1), `text_encoders\qwen3vl_8b_int8_convrot.safetensors`, `vae\qwen_image_2.1_vae_bf16.safetensors` | present |
| Visual Studio | `"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath` | a path (VS 2022 or 2026 with the C++ x64 tools) |
| uv, git | `uv --version`, `git --version` | both present |
| Disk | free space on the drive holding `%LOCALAPPDATA%` (or `TFIMAGE_CACHE`) | >= 6 GB, preferably internal NVMe |

`COMFYUI_PORTABLE` is the folder holding `python_embeded\` and `ComfyUI\`. If the GGUF is elsewhere, set
`QWEN_IMAGE21_GGUF` to its full path.

## 2. Build

```bat
git submodule update --init
scripts\setup.cmd
```

Expect `tensorfold.cuda.nvfp4.checkpoint._ext OK` and `tensorfold.cuda.kernels.qmm._ext OK`, then `setup done`.
If nvcc fails with `cudafe++ died with status 0xC0000005`, the CUDA 13.0 compiler is being used with VS 2026: the
setup pins 13.4; check `.venv\Lib\site-packages\nvidia\cu13\bin\nvcc --version`.

Checks (each prints its numbers; none touch ComfyUI):

```bat
tools\env.cmd python tools\check_kernels.py           :: adaln / rms_rope / swiglu vs torch: max |d| <= 0.0078
tools\env.cmd python tools\check_gguf_dequant.py "%QWEN_IMAGE21_GGUF%"   :: max abs diff 0.0 for every type
```

## 3. Convert and calibrate (once per GGUF)

```bat
"%COMFYUI_PORTABLE%\python_embeded\python.exe" -B bench\comfy_dump_contexts.py   :: 8 prompts through the text encoder
tools\env.cmd python bench\calibrate_run.py                                       :: ~10 s convert + ~60 s calibrate
tools\env.cmd python bench\engine_step.py                                         :: step s at 1024x1024: ~0.3 on a 5070 Ti
```

The converted engine is written to `%LOCALAPPDATA%\tfimage` (or `TFIMAGE_CACHE`), ~4.2 GB for NVFP4.

## 4. Measure against the user's ComfyUI (optional, recommended)

```bat
tools\env.cmd python bench\comfy_baseline.py --tag base --repeat 3 --jobs bench\jobs_gate.json
tools\env.cmd python bench\comfy_baseline.py --tag nvfp4 --tf nvfp4 --repeat 3 --jobs bench\jobs_gate.json
tools\env.cmd python bench\quality.py base nvfp4
```

The `--tf` run fails on purpose if ComfyUI's log shows a GGUF load or no `[tfimage]` engine (a run that silently
measured the baseline twice). Look at the contact sheet `runs\quality_nvfp4.png` with the user.

## 5. Install the node (ask first)

Either a junction (ComfyUI then loads it like any custom node):

```bat
mklink /J "%COMFYUI_PORTABLE%\ComfyUI\custom_nodes\ComfyUI-TensorFold-Image" "%CD%\comfyui\ComfyUI-TensorFold-Image"
```

or an `extra_model_paths.yaml` entry with `custom_nodes:` pointing at this repo's `comfyui\` folder. Restart ComfyUI,
then in the workflow replace `Unet Loader (GGUF)` with **TensorFold Qwen-Image 2.1 Loader**, same GGUF, precision
`nvfp4`. The node finds this repo through its own location (or `TFIMAGE_REPO`).

Known refusals (an error, by design): LoRA, ControlNet, block / attention patches.
