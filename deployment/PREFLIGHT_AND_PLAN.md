# ActQuant 部署前检查与方案

检查时间：2026-10-19（Asia/Shanghai）。检查阶段已完成，随后才开始依赖安装。

## 现场
- 工作区 `/workspace/actquant` 最初为空，不是 Git 仓库；主机 `acf42a8d6bf7`，与前次 SSH 主机一致。
- Ubuntu 22.04.5 / x86_64，Python 3.11.10，Git 2.34.1，uv 0.13.1。
- 预装 torch 2.4.1+cu124 / torchvision 0.19.1+cu124 / torchaudio 2.4.1+cu124 / NumPy 1.26.3，CUDA 实测可用。
- NVIDIA L40S 46068 MiB，计算能力 8.9；驱动 595.91.07。nvidia-smi 的 CUDA 13.2 是驱动支持上限，nvcc Toolkit 实际为 12.4.131，torch runtime 为 12.4，三者不是同一版本概念。
- 初始磁盘约 150 GiB 可用，内存可用约 494 GiB。GPU 初始空闲。
- GitHub / 项目官网 / PyPI / PyTorch wheel / HuggingFace / arXiv HTTPS 均返回 200。
- 缺少 CMake、Ninja、Conda、Rust、EGL loader 和 NVIDIA EGL 图形驱动；GCC/G++/Make 和 CUDA 编译器可用。
- 未发现适用 AGENTS.md。

## 官方来源与固定版本
- 官网 https://actquant.github.io/ 的 Code 链接直接指向 https://github.com/arashakb/ActQuant 。
- 论文 https://arxiv.org/pdf/2605.24011 ，浏览版本 v3（2026-08-03），附录 F 明确使用 L40S 完成 LIBERO rollout 和量化；完整量化约两小时，本部署不执行。
- ActQuant: b64791125070652fe6b554e244fe809c79ef5246。
- OpenPI: 215abfb217dbac7d5f1273282331b9b1866c0479。
- OpenPI 指定 LIBERO 子模块: f78abd68ee283de9f9be3c8f7e2a9ad60246e95c。
- LeRobot 依赖由 OpenPI uv.lock 固定为 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5。

已读 ActQuant README、requirements.txt、tools/pi0.5/01_setup_env.sh、CMakeLists.txt、serve_policy.py、run_libero_eval.sh、HSIC/Fisher 模型加载入口；已读 OpenPI README、pyproject.toml、uv.lock、LIBERO README/requirements 和 LIBERO requirements/setup.py。

## 冲突分析与解决方案（安装前）
1. ActQuant README 服务端 Python 3.11，与辅助脚本的 3.12 不一致。量化实际依赖 OpenPI 的 PI0Pytorch，因此选 3.11，按 OpenPI uv.lock 完整安装运行依赖，不采用辅助脚本中未固定版本的 `pip install torch`。
2. LIBERO 锁定 Python 3.8 / torch 1.11.0+cu113 / NumPy 1.22.4；OpenPI 锁定 torch 2.7.1 / transformers 4.53.2 / NumPy 1.26.4。必须两个 venv 隔离。uv 提供 Python 3.8，无需安装全局 Conda；这是对官方环境管理工具的适配，依赖流程仍遵循官方。
3. OpenPI 要求覆盖 Transformers 若干文件以支持 AdaRMS 等。仅在新建服务端环境应用上游自带补丁，采用 UV_LINK_MODE=copy 避免污染缓存或其他环境，并保存原始文件备份与哈希。这不是改动 ActQuant 量化算法。
4. 官方 CUDA 12.6 为 Pi0.5 测试版本，现场为 12.4。先用现有工具链针对 L40S sm_89 编译，仅在实际编译不兼容时考虑项目内隔离工具链；不改系统 CUDA/驱动。PyTorch wheel 自带 runtime 与编译器版本可不同。
5. 官方多卡 launcher 带 /path/to 占位和 `--args.ports`，当前上游客户端仅支持 `--args.port`。保留官方文件，另建单卡 launcher 对接原始 server/client，默认只校验参数，运行需要明确调用。
6. CMake/Ninja/pybind11 安装至独立工具环境。Pi0.5 默认 SentencePiece，构建指定目标可避免无关 OFT/Rust 编译。禁用 LLAMA_CURL 避免缺 curl 开发头和自动模型 URL 下载。
7. EGL loader 与 NVIDIA EGL vendor 缺失。先检查可在项目目录解包的 Ubuntu Mesa/EGL 依赖；软件 EGL 如可用需明确标注 CPU 渲染，不声称 NVIDIA 硬件 EGL 已验证。
8. ActQuant 顶层 requirements.txt 为 llama.cpp 通用转换工具合集，并非 Pi0.5 完整环境，不盲目叠加以免改变 OpenPI 锁定依赖。

## 边界
只下载代码、Python 解释器、软件依赖和编译资源；LIBERO 仓库内置仿真 meshes/初始状态属于源码资源。禁用 LFS smudge，不调用权重、轨迹数据集或校准数据下载脚本，不运行量化或完整评估，不删除已有文件。最终做 import、GPU 小张量、CLI help 和短暂仿真渲染检查，并记录未验证项。
