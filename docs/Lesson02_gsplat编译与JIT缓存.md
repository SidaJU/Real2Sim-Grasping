# Lesson 02 · gsplat JIT 编译：训练「卡死」的真凶

> 环境：Windows 11 / RTX 5060 (sm_120) / CUDA 12.8 / torch 2.7.1+cu128 / Python 3.10.20 / gsplat 1.4.0 / nerfstudio 1.1.5
> 本篇记录一次**典型的误判**：现象是「训练卡住」，真因是 CUDA 扩展在后台编译，编译还失败了。

---

## 1. 一句话总结

`ns-train` 第一次调用高斯光栅化时，gsplat 会 **JIT 编译 CUDA 扩展**（耗时 1-3 分钟，期间 GPU 利用率 0%）。
编译器（`cl.exe`）和构建工具（`ninja`）没进 PATH → 编译失败 → 报错信息还被中文编码问题二次掩盖。
**解法：配好 MSVC + ninja 环境，单独预编译一次并缓存。**

---

## 2. 现象

### 2.1 看起来是「卡死」

启动日志停在最后一行不动：

```
[16:34:28] Caching / undistorting train images     full_images_datamanager.py:230
```

查 GPU：

```
nvidia-smi → 0 %, 3347 MiB
```

**GPU 0% 但显存占着、进程活着** = 没在计算，而是在**阻塞等待**（真实情况是 CPU 在跑 nvcc 编译）。

### 2.2 真正的报错

```
ImportError: cannot import name 'csrc' from 'gsplat'
   ↓ 触发 JIT 编译
subprocess.CalledProcessError: Command '['ninja', '-v', '-j', '10']'
   returned non-zero exit status 1.
   ↓ 读取错误输出时
UnicodeDecodeError: 'cp1' codec can't decode bytes in position 0--1
```

最后那个 `UnicodeDecodeError` 是**雪上加霜**：torch 想打印 ninja 的中文错误输出，解码失败，把真正的编译错误**吞掉了**——所以日志里看不到失败原因。

---

## 3. 误判过程（值得记住的教训）

| 轮次 | 我的判断 | 结果 |
|---|---|---|
| 1 | 「viewer (viser) 在 Windows 上阻塞了训练」→ 把 `--vis viewer` 改成 `--vis tensorboard` | ❌ 还是卡在同一行 |
| 2 | 看到 `Trainer.train_iteration: 67.7964` 这条 profiling 输出 | ✅ 意识到 67 秒不是等待，是**编译** |
| 3 | 定位到 `gsplat\cuda\_backend.py` 的 `_C = load_extension(...)` | ✅ 确认是 JIT 编译失败 |

**教训**：`Trainer.train_iteration: 67.8 秒` 这种异常数字是强线索——正常训练一步只有 ~11 ms。

---

## 4. 根因链条（四个环节同时坏）

| 环节 | 问题 | 证据 |
|---|---|---|
| ① MSVC | 旧脚本调用 `...\2022\Professional\VC\...\vcvarsall.bat`，**该版本已不存在** → `cl.exe` 没进 PATH | `Test-Path` 返回 False |
| ② ninja | `ninja.exe` 在 `envs\nerfstudio\Scripts\`，**不在 PATH** | `where ninja` 找不到 |
| ③ JIT 缓存 | 已有缓存是 **MSVC 14.44** 编的，当前是 **14.38** → 环境不匹配，缓存不命中，被迫重编 | 缓存目录 `py310_cu128` 存在但仍重编 |
| ④ 编码 | 中文 Windows + `chcp 65001`，torch 解码 ninja 输出失败 → 真错误被吞 | `UnicodeDecodeError` |

**环环相扣**：只要 ①②任一缺失，`ninja -j 10` 必然失败；而 ④ 让你看不到失败原因。

---

## 5. 诊断方法

```powershell
# 1. 编译器/工具链是否在 PATH（关键三步）
where cl.exe
where nvcc.exe
where ninja

# 2. gsplat 是否有编译产物
python -c "import gsplat,os; d=os.path.dirname(gsplat.__file__); \
  print([f for f in os.listdir(d) if f.endswith(('.pyd','.so'))])"
# 空列表 = 没编译，只能靠 JIT

# 3. 找真实路径
Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
Get-ChildItem "<env>\Scripts" -Filter ninja.exe -Recurse
```

诊断结果（本机）：

```
cl.exe   → C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64
nvcc.exe → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
ninja    → C:\Users\theozhu\miniconda3\envs\nerfstudio\Scripts\ninja.exe
MSVC 可用版本：14.38.33130 / 14.44.35207
```

---

## 6. 解决方案

### 6.1 修训练脚本 `run_train_ycb.bat`

```bat
rem 修正 vcvars 路径（Professional → BuildTools），并去掉写死的版本号
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 > nul

rem ninja 所在目录必须进 PATH
set PATH=%CUDA_HOME%\bin;%USERPROFILE%\miniconda3\envs\nerfstudio\Scripts;%PATH%
set MAX_JOBS=4
```

### 6.2 单独预编译一次：`fix_gsplat.bat`

**核心思路**：不要等训练时才发现编译失败——**先单独编译并缓存**，之后 `ns-train` 直接命中缓存，不再有 60 秒停顿。

```bat
call "...BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 > nul
set PATH=%CUDA_HOME%\bin;%USERPROFILE%\miniconda3\envs\nerfstudio\Scripts;%PATH%
set TORCH_CUDA_ARCH_LIST=12.0
set PYTHONIOENCODING=utf-8          # 防止 UnicodeDecodeError 吞掉错误信息
set MAX_JOBS=4

where cl.exe && where nvcc.exe && where ninja
python -c "import gsplat.cuda._backend as b; print('BUILD OK:', b._C)"
```

### 6.3 结果

```
BUILD OK: <module 'gsplat_cuda' from
  'C:\Users\theozhu\AppData\Local\torch_extensions\torch_extensions\Cache\py310_cu128\gsplat_cuda\gsplat_cuda.pyd'>
exit code: 0
```

之后训练不再停顿，直接开始刷进度条。

---

## 7. 背景知识：为什么会有 JIT

gsplat 是带 CUDA 源码的扩展包，有两种分发方式：

| 方式 | 说明 | 本环境 |
|---|---|---|
| **预编译 wheel** | pip 装下来就带 `.pyd` | ❌ gsplat 1.4.0 在 Windows 无官方 wheel |
| **JIT 编译** | 首次 import 时用 ninja 现编，结果缓存到 `torch_extensions` | ✅ 我们走这条路 |

`_backend.py` 的逻辑：

```python
try:
    from gsplat import csrc as _C      # 预编译版
except ImportError:
    _C = load_extension(...)           # 退回 JIT 编译
```

**缓存命中条件**：编译器版本、CUDA 版本、Python 版本、编译 flags 全都一致。所以**换 MSVC 版本会导致缓存失效重编**——这正是我们遇到的情况。

---

## 8. 可复用经验

1. **GPU 0% + 显存占用 + 进程存活 = 在等 CPU（很可能是编译），不是死锁。**
2. **异常的单步耗时是强线索**：67 秒 vs 正常 11 毫秒。
3. **`where cl.exe / nvcc / ninja` 应该成为 Windows 上跑 CUDA 扩展前的标准三步检查。**
4. **预编译优于训练时编译**：单独跑一次，把失败暴露在可控环境里。
5. **错误被 `UnicodeDecodeError` 吞掉时**：设 `PYTHONIOENCODING=utf-8` 通常能让真实错误显形。
6. **JIT 缓存会因工具链版本变化失效**——升级/更换 VS 后要预期一次重新编译。

---

## 附录：同一阶段踩到的参数名坑

| 错误写法 | 正确写法 | 说明 |
|---|---|---|
| `--eval-mode none` | `--eval-mode {fraction,filename,interval,all}` | 本版本没有 `none`；用 `interval` + `--eval-interval 10` |
| `--orientation none` | `--orientation-method none` | 完整名是 `orientation-method` |

**快速查参数**：`ns-train splatfacto nerfstudio-data --help`
