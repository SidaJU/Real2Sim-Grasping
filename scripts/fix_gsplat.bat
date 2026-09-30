@echo off
rem ------------------------------------------------------------------
rem Pre-build gsplat's CUDA extension once, so ns-train / render scripts
rem hit the JIT cache instead of compiling (and failing) mid-run.
rem
rem Recreated from docs/Lesson02_gsplat编译与JIT缓存.md  (section 6.2)
rem Run:  cmd /c scripts\fix_gsplat.bat
rem ------------------------------------------------------------------

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul

rem nvcc + ninja must be on PATH, otherwise `ninja -j 4` fails
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\Users\theozhu\miniconda3\envs\nerfstudio\Scripts;%PATH%

rem RTX 5060 = sm_120
set TORCH_CUDA_ARCH_LIST=12.0

rem without this, torch cannot decode ninja's Chinese error output and the
rem real compile error gets swallowed (Lesson02 section 2.2)
set PYTHONIOENCODING=utf-8
set MAX_JOBS=4

echo --- toolchain ---
where cl.exe
where nvcc.exe
where ninja

echo --- building gsplat CUDA extension (1-3 min on first run) ---
C:\Users\theozhu\miniconda3\envs\nerfstudio\python.exe -c "import gsplat.cuda._backend as b; print('BUILD OK:', b._C)"

echo exit code: %ERRORLEVEL%
