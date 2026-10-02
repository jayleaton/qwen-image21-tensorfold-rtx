@echo off
rem Run a command with MSVC x64 (found with vswhere), NVIDIA's pip CUDA 13.4 toolkit in .venv, and the venv on PATH.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (echo Visual Studio with the C++ x64 tools was not found & exit /b 1)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul
set "ROOT=%~dp0.."
set "CUDA_HOME=%ROOT%\.venv\Lib\site-packages\nvidia\cu13"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%ROOT%\.venv\Scripts;%CUDA_HOME%\bin;%CUDA_HOME%\bin\x86_64;%CUDA_HOME%\nvvm\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0a"
set "NVCC_APPEND_FLAGS=-allow-unsupported-compiler"
set "TORCH_EXTENSIONS_DIR=%ROOT%\.build\torch_ext"
set "TRITON_CACHE_DIR=%ROOT%\.build\triton"
%*
