@echo off
setlocal
cd /d "%~dp0"
if not defined COMFYUI_PORTABLE (echo Set COMFYUI_PORTABLE to your portable installation & exit /b 1)
"%COMFYUI_PORTABLE%\python_embeded\python.exe" -B server.py %*
