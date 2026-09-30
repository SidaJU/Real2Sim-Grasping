@echo off
rem ------------------------------------------------------------------
rem Run any nerfstudio/gsplat python script with the EXACT environment that
rem fix_gsplat.bat used to build the CUDA extension.
rem
rem Why this wrapper exists:
rem   torch's JIT cache key includes the compile flags. If TORCH_CUDA_ARCH_LIST
rem   differs from the one used at build time (or is unset), the cache MISSES and
rem   torch tries to recompile with ninja -> fails when cl.exe is not on PATH.
rem   Same failure mode as docs/Lesson02 section 4 (③ JIT cache miss).
rem
rem Usage:  cmd /c scripts\run_nerfstudio.bat 22_render_gs_depth.py
rem ------------------------------------------------------------------

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul

set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\Users\theozhu\miniconda3\envs\nerfstudio\Scripts;%PATH%

rem MUST match fix_gsplat.bat, otherwise the JIT cache misses
set TORCH_CUDA_ARCH_LIST=12.0
set PYTHONIOENCODING=utf-8
set MAX_JOBS=4

cd /d "%~dp0"

echo --- env ---
where cl.exe
where ninja

C:\Users\theozhu\miniconda3\envs\nerfstudio\python.exe %1
echo exit code: %ERRORLEVEL%
