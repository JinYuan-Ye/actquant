# ActQuant π₀.₅ / LIBERO 部署报告

日期：2026-10-19（Asia/Shanghai）  
工作区：`/workspace/actquant`  
状态：**环境部署完成；官方源码编译与无模型冒烟验证通过。尚未执行真实权重量化或策略评估。**

## 1. 官方来源和部署前检查

官网 [actquant.github.io](https://actquant.github.io/) 的 Code 链接确认官方仓库为 [arashakb/ActQuant](https://github.com/arashakb/ActQuant)。论文 [arXiv:2605.24011](https://arxiv.org/pdf/2605.24011) 本次读取版本为 v3；附录 F 描述使用 L40S 进行 LIBERO 评估和量化，完整量化约两小时，本次未运行。

完整安装前分析见 [PREFLIGHT_AND_PLAN.md](PREFLIGHT_AND_PLAN.md)。检查和仓库阅读在依赖安装前完成。工作区起初为空，与之前 SSH 访问的是同一主机 `acf42a8d6bf7`，因此本次直接在工作区执行，无需再次 SSH。

| 项目 | 初始现场 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS，x86_64 |
| GPU | NVIDIA L40S，46068 MiB，计算能力 8.9，初始空闲 |
| 驱动 | 595.91.07 |
| 系统 CUDA Toolkit | 12.4.131 |
| 预装 Python / torch | 3.11.10 / 2.4.1+cu124，CUDA 实测可用 |
| 预装 torchvision / torchaudio / NumPy | 0.19.1+cu124 / 2.4.1+cu124 / 1.26.3 |
| Git / uv | 2.34.1 / 0.13.1 |
| 初始磁盘 / 内存 | 约 150 GiB 磁盘可用，约 494 GiB 内存可用 |
| 网络 | 官网、GitHub、PyPI、PyTorch wheel、HuggingFace、arXiv 的 HTTPS 检查均为 200；实际源码及依赖下载成功 |
| 缺失组件 | CMake、Ninja、Conda、Rust、EGL loader、NVIDIA EGL 图形驱动 |

`nvidia-smi` 显示的 CUDA 13.2 是驱动支持上限，不是系统 nvcc 或 PyTorch runtime 版本。

## 2. 固定的源码版本

| 仓库 | 提交 |
|---|---|
| ActQuant | `b64791125070652fe6b554e244fe809c79ef5246` |
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LIBERO（OpenPI 指定子模块） | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| LeRobot（OpenPI uv.lock） | `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |

已阅读 ActQuant README、顶层 requirements、官方安装脚本、相关 CMake、π₀.₅ 服务端/评估脚本及 HSIC/Fisher 入口；也阅读了 OpenPI README、pyproject、uv.lock、LIBERO README 和两份 LIBERO requirements。

ActQuant、OpenPI、LIBERO 的**已跟踪源码均无修改**，Git diff 已核对。`src/ActQuant/build_openpi` 是指向工作区构建目录的新增符号链接，满足官方导出器寻找 libggml 的路径约定；供应商源码从官方附带 zip 解包。

## 3. 独立环境与版本

| 环境 | Python | 主要依赖 |
|---|---|---|
| `envs/openpi-server` | 3.11.10 | torch 2.7.1+cu126；torchvision 0.22.1；transformers 4.53.2；NumPy 1.26.4；JAX/JAXlib 0.5.3；safetensors 0.5.3；sentencepiece 0.2.0；websockets 15.0.1；pybind11 2.13.6 |
| `envs/openpi-libero` | 3.8.20 | torch 1.11.0+cu113；torchvision 0.12.0+cu113；torchaudio 0.11.0+cu113；NumPy 1.22.4；robosuite 1.4.1；MuJoCo 3.2.3；BDDL 1.0.1；numba 0.53.1；websockets 13.1 |
| `envs/build-tools` | 3.11.10 | CMake 3.31.6；Ninja 1.11.1.4 |

服务端遵循 OpenPI 的官方 `uv sync` 流程，使用 `--frozen --no-dev`，保留官方 uv.lock；重跑时 `--inexact` 保留另装的 pybind11。客户端从官方两份 requirements 解析，并用 `client-constraints.txt` 固定全部已解析的非 editable 依赖。两个环境分别安装自己的 openpi-client。

完整版本和源码/产物哈希见：

- [evidence/manifest.json](evidence/manifest.json)
- [evidence/openpi-server-freeze.txt](evidence/openpi-server-freeze.txt)
- [evidence/openpi-libero-freeze.txt](evidence/openpi-libero-freeze.txt)
- [evidence/build-tools-freeze.txt](evidence/build-tools-freeze.txt)
- [client-constraints.txt](client-constraints.txt)
- [evidence/reference-sha256.txt](evidence/reference-sha256.txt)

预装环境的安装前、后 `pip freeze` **完全一致**。未安装全局 Python 包，未升级系统 CUDA/驱动或使用 apt install。uv 缓存、下载的 Python 3.8 和渲染库均位于项目目录。

## 4. 冲突、偏差与处理依据

| 问题 | 原因与处理 |
|---|---|
| 官方服务端 Python 版本不一致 | README 使用 3.11，辅助安装脚本写 3.12。依据 OpenPI `.python-version` 与官方 PyTorch 安装流程选择 3.11。 |
| 服务端与 LIBERO 栈不兼容 | torch 2.7.1/transformers 4.53.2 与旧客户端版本不可混用，使用两个隔离 venv。uv 管理 Python 3.8 替代安装全局 Conda。 |
| 预装 torch 不符合 OpenPI 锁定版本 | 保留预装 2.4.1，在新环境安装锁定的 2.7.1，不对原环境升级。 |
| OpenPI 需要 Transformers 补丁 | 按其 README 使用仓库自带 `transformers_replace`。只修改新环境中的第三方库；原始文件备份至 `evidence/transformers-original`，文件哈希见 `transformers-patch-manifest.json`；`UV_LINK_MODE=copy` 防止污染 uv 缓存。未改 ActQuant 算法。 |
| 官方测试 CUDA 12.6，现场 12.4 | 使用已有 nvcc 12.4、`CMAKE_CUDA_ARCHITECTURES=89` 编译成功。服务端 torch 使用自身 cu126 runtime；系统工具链不变。验证了原生 CUDA 后端发现 L40S，但无权重推理精度/性能验证。 |
| egl-probe 编译时找不到 CMake | 首次客户端安装失败；安装独立 CMake 后把工具环境加入安装过程 PATH，重试成功。依赖版本无需变更。 |
| 缺少 NVIDIA EGL vendor | 仅解包 Ubuntu EGL/Mesa 到 `.local/render`。第一次使用系统 DRI 时发现 `.2` / `.4` 构建版本冲突；补齐相同 `.4` 版本的 DRI/glapi 库解决。 |
| EGL 默认设备不可用 | 设备 0 为不可初始化的硬件设备，设备 1 为可用 Mesa 软件设备。部署脚本探测并记录设备号，以 `MUJOCO_EGL_DEVICE_ID` 选择；客户端解除 CUDA 可见设备变量，服务端单独使用 GPU 0。实测渲染器为 llvmpipe（LLVM 15.0.7）。 |
| editable LIBERO 导入失败 | 官方 README 本来就要求将 LIBERO 仓库加入 PYTHONPATH；已在客户端激活脚本中设置。 |
| 官方评估 launcher 的路径与 CLI 不匹配 | 原脚本含 `/path/to`、Conda 路径、多卡 `--args.ports`，当前上游客户端为单端口 `--args.port`。新增单卡 launcher，调用原始 server/client，保留原始官方文件。 |
| 导出器寻找 build_openpi 路径 | 新增链接到同一 `build/pi05`，避免混用另一套 libggml；实际库发现已验证。 |
| 帮助命令退出码 | 官方 llama-quantize 帮助、quantize-vision 无参数 usage 均返回 1；验证脚本按源码约定检查退出码和帮助文本，未修改二进制行为。 |

CMake/Ninja 在独立工具环境提供；只构建 π₀.₅/SentencePiece 和量化目标，无需安装 OFT 或 Rust。`LLAMA_CURL=OFF` 避免模型 URL 下载功能与 curl 开发依赖，不影响本地量化算法。上游存在编译 warning，构建最终成功，未为消除 warning 改动官方代码。

第一次服务端安装期间，部署脚本被编辑导致脚本尾部解析错误；依赖本身已经安装完成。该操作问题已修正，后续完成语法检查、完整安装脚本成功重跑和短时验证，原始失败日志保留。

## 5. 构建与验证结果

构建目录：`build/pi05`，以下目标均存在：

- `bin/pi05`：官方推理 CLI。
- `bin/pi05.so`：Python 3.11 绑定。
- `bin/llama-quantize`：ActQuant HSIC/AMF 使用的量化工具。
- `bin/quantize-vision`：官方视觉量化工具。
- 同目录 `libggml*.so` / `libllama.so`，避免混合不同构建的共享库。

已通过：

1. 两个 Python 环境的 `uv pip check`。
2. torch CUDA 可用、32×32 小矩阵 GPU/CPU 结果比对。
3. `PI0Pytorch` 导入与 π₀.₅ 配置创建，未实例化完整模型。
4. `pi05.so` 导入及官方导出器 GGML 动态库发现。
5. 原生 GGML 枚举 `CUDA0: NVIDIA L40S`。
6. 官方服务端 NumPy 消息编码与 OpenPI 客户端解码兼容检查（非真实 WebSocket 模型推理）。
7. HSIC、Fisher、校准数据准备、GGUF 导出/合并、位宽分配入口的 `--help`。
8. LIBERO 四个目标 suite 均有 10 个任务；读取仓库内置 50 个初始状态；首个 spatial 任务重置和 64×64 RGB 渲染成功。
9. 单卡 launcher 的 Bash 语法和无参数帮助；未用真实模型运行。
10. 完整 `install.sh all` 重跑成功，后续构建重跑显示无待构建任务；预装环境和官方跟踪源码比对通过。

**未验证项**：真实 checkpoint 加载、HSIC/AMF 计算、量化后的模型精度、策略动作正确性、闭环成功率、吞吐与延迟。未下载模型或轨迹数据，因此不声称复现论文结果。软件 EGL 已可用，NVIDIA 硬件 EGL 未部署；正式高吞吐评估如需硬件渲染，需要 TAPGPU 容器提供匹配驱动的 NVIDIA 图形库。

## 6. 使用和复跑

在此 Ubuntu/CUDA 主机上复跑完整安装（仅依赖、编译与短时验证）：

```bash
cd /workspace/actquant
bash deployment/install.sh all
```

分阶段入口：`sources`、`tools`、`server`、`client`、`build`。完整流程还包括官方补丁、局部渲染库、配置、验证和版本清单；建议使用 `all`。源码提交不匹配时会拒绝替换已有 checkout，不执行删除文件的命令。

激活校准/导出/服务端环境：

```bash
source /workspace/actquant/deployment/activate_server.sh
python /workspace/actquant/src/ActQuant/tools/hsic/compute_hsic_pi05.py --help
```

激活 LIBERO 客户端：

```bash
source /workspace/actquant/deployment/activate_libero.sh
```

只重做环境检查：

```bash
bash /workspace/actquant/deployment/verify.sh
```

两个激活脚本默认设置 HuggingFace/Transformers/Datasets 离线模式。若未来授权下载，需在单独下载步骤显式解除相应变量；安装脚本本身不下载权重或数据。

后续评估准备好本地 `EVAL_DIR` 后，其目录需包含：

```text
pi05.gguf
tokenizer.model
norm_stats.json
```

未来明确开始评估时可手动调用以下命令；本次未执行：

```bash
bash deployment/run_eval_single_gpu.sh --run /absolute/EVAL_DIR libero_spatial 1
```

`1` 表示每个任务 1 次，一整个 suite 仍有 10 个任务；该命令属于实际实验，不是安装验证。策略服务仅监听本机 127.0.0.1，默认 GPU 0/端口 8000；结果写入新建时间戳目录。

从头量化还需用户提供原始 checkpoint、tokenizer、归一化统计和校准数据。Fisher/HSIC 接受本地 `--checkpoint` 与 `--calib-dir`，单卡参数使用 `--num-gpus 1`；官方 `run_hsic_quant_pi05.sh` 保留其占位路径，本次不直接执行。校准导出器读取本地 RLDS，包含纯 Python TFRecord parser，因此没有额外安装 TensorFlow。

## 7. 文件、日志和存储

- 安装脚本：[install.sh](install.sh)
- 校验脚本：[verify.sh](verify.sh)
- 单卡入口：[run_eval_single_gpu.sh](run_eval_single_gpu.sh)
- 所有安装步骤使用 `set -x` 和时间戳日志：[logs/](logs/)
- 完整复跑记录：[logs/replay-console.log](logs/replay-console.log)
- 最终服务端检查：[logs/server-smoke-final.log](logs/server-smoke-final.log)
- 最终原生后端检查：[logs/native-smoke-final.log](logs/native-smoke-final.log)
- LIBERO 渲染记录：[logs/libero-smoke-selected-device.log](logs/libero-smoke-selected-device.log)
- 首次源码获取命令：[SOURCE_COMMANDS.md](SOURCE_COMMANDS.md)
- 部署后约剩余 124 GiB；新环境约 12.4 GiB，缓存约 13 GiB，源码约 1.1 GiB，构建约 319 MiB。缓存保留，未清理或删除已有文件；精确快照见 manifest。

所有改动均位于项目内。LIBERO 仓库下载的 meshes、BDDL 和初始状态是仿真资源；未下载 demonstrations/RLDS 轨迹数据集或任何模型权重。本次也未进行量化、训练或长时间 rollout。


## 2026-10-19 LeRobot checkpoint 兼容性参考环境

用户选择在 README 指定的 LeRobot checkpoint 上继续进行带输入兼容适配的方法复现。新增 envs/lerobot-reference；安装脚本 deployment/install_lerobot_reference.sh；依赖锁 deployment/lerobot-reference.lock。官方源码单独置于 src/lerobot-v044，固定提交 8fff0fde7c79f23a93d845d1a50e985de01f8b8a。

该环境使用官方 pi extra 对应 Transformers 分支的固定提交，包版本4.53.3；OpenPI原环境保留4.53.2补丁。LeRobot rerun-sdk要求NumPy2，因此新环境固定2.2.6，原环境保留1.26.4。安装日志包含最初NumPy约束失败及修正后的安装。新环境PyTorch为2.7.1+cu126，L40S可见，官方模型/processor导入及补丁检查通过。

reproduction/pi05_lerobot/ 提供独立输入适配、用户显式下载和单样本参考前向入口。已完成CPU单元检查（2项）及配置解析/语法检查；实际tokenizer、完整权重、模型前向、跨后端对齐与闭环评估尚未验证。未下载模型/数据集、运行量化或修改官方仓库源码。详情见该目录README及deployment/PI05_CHECKPOINT_COMPATIBILITY.md。
