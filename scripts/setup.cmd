@echo off
rem One-time setup: .venv with torch 2.10+cu130 and NVIDIA's pip CUDA 13.4 toolkit, then TensorFold's kernels built
rem for this GPU into .build\torch_ext (the ComfyUI node loads them from there; ComfyUI itself needs no compiler).
setlocal
cd /d "%~dp0.."
where uv >nul 2>nul || (echo uv is required: https://docs.astral.sh/uv/ & exit /b 1)
if not exist vendor\TensorFold\pyproject.toml git submodule update --init || exit /b 1
if not exist .venv\Scripts\python.exe (uv venv --python 3.13 .venv || exit /b 1)
set "PY=.venv\Scripts\python.exe"
uv pip install --python %PY% torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130 || exit /b 1
rem CUDA 13.4 nvcc (13.0 crashes under VS 2026) with CCCL 13.0 headers (13.3 needs /Zc:preprocessor); the math-library
rem wheels supply headers ATen includes
uv pip install --python %PY% "nvidia-cuda-nvcc==13.4.*" "nvidia-cuda-crt==13.4.*" "nvidia-nvvm==13.4.*" ^
    "nvidia-cuda-runtime==13.4.*" "nvidia-cuda-cccl==13.0.85" "nvidia-cusparse<13" "nvidia-cublas<14" "nvidia-cusolver<13" ^
    "nvidia-curand<11" "nvidia-cufft<13" ninja "triton-windows<3.7" pillow pytest lpips || exit /b 1
uv pip install --python %PY% -e . || exit /b 1
rem TensorFold stays an unmodified submodule: put its src on the venv's path instead of installing it
%PY% -c "import site,pathlib;pathlib.Path(site.getsitepackages()[-1],'tensorfold_vendor.pth').write_text(str(pathlib.Path('vendor/TensorFold/src').resolve()))" || exit /b 1
call tools\env.cmd python tools\tf_build_probe.py nvfp4.checkpoint qmm || exit /b 1
echo setup done
