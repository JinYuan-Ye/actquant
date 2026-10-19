# LeRobot π₀.₅ 参考与输入兼容适配

最新全量任务（2026-10-19）：用户已授权60轨迹全部帧校准与量化。四套源数据104文件已校验，导出11,777帧，数据审计通过，全帧HSIC已在GPU运行；后续自动执行分配、Fisher和量化。进度与命令说明见 [FULL_CALIBRATION_RUN.md](FULL_CALIBRATION_RUN.md)，当前事实以 `reproduction/results/pi05-full-01/pipeline.json` 为准。以下四帧模型是已完成的历史教学版，不能当作本轮全量成品。

当前阶段（2026-10-19）：**第一份完整LLM混合位宽量化模型已生成并完成同后端十步采样**，见 [量化模型与实测](QUANTIZED_MODEL_DEMO.md)。126个目标全部量化，模型6.71GB→3.33GB；四帧校准、单观测配对运行，任务质量未验收。此前跨后端严格NUMERICAL_MISMATCH完整保留，见 [C++诊断](CPP_FORWARD_DIAGNOSTICS.md)，未通过改阈值转成PASS。其他流程见 [量化衔接](QUANTIZATION_BRIDGE.md)、[HSIC](HSIC_DEMO.md)、[分配](ALLOCATION_DEMO.md)、[Fisher](FISHER_DEMO.md) 和 [方法导读](METHOD_WALKTHROUGH.md)。正式校准和LIBERO闭环尚未完成。

历史对照过程见 [OpenPI 单样本对照](OPENPI_COMPARISON.md)；-02已解决MISMATCH，无需重跑。新方法入口是run_calibration.py，4样本HSIC及Fisher累计均已通过，正式规模校准未执行。

## 固定来源与环境

早期逐条操作说明见 [第一份真实 LIBERO 样本](REAL_SAMPLE_GUIDE.md)。用户已完成下载、native提取和真实前向。指南中的命令保留供复查，无需重跑。当前对照结果见上方链接。

- LeRobot v0.4.4：8fff0fde7c79f23a93d845d1a50e985de01f8b8a。
- 官方 pi extra 指定的 Transformers fix/lerobot_openpi 分支固定为 dcddb970176382c0fcf4521b0c0e6fc15894dfe0，包版本 4.53.3。
- PyTorch 2.7.1+cu126、torchvision 0.22.1+cu126、NumPy 2.2.6、SciPy 1.14.1。
- checkpoint：lerobot/pi05_libero_finetuned_v044，revision 7152f13e34c6c115d808aea0e5d08aca74eac14b。
- tokenizer：google/paligemma-3b-pt-224，revision 35e4f46485b4d07967e7e9935bc3786aad50687c。
- 全量依赖：../../deployment/lerobot-reference.lock；实际安装：../../deployment/evidence/lerobot-reference-freeze.txt。

复装环境：`bash deployment/install_lerobot_reference.sh`。执行目录均为 `/workspace/actquant`。

依赖冲突处理：OpenPI 的 Transformers 4.53.2 本地补丁与 LeRobot 官方 pi 分支分开；LeRobot 的 rerun-sdk 要求 NumPy>=2，因此不能复用 OpenPI 的 NumPy 1.26.4。两个独立解释器解决版本冲突，没有升级系统 Python 或原 OpenPI 环境。

## 已完成的参考验证流程

```bash
cd /workspace/actquant
source deployment/activate_lerobot_reference.sh
python reproduction/pi05_lerobot/download_assets.py --tokenizer
python reproduction/pi05_lerobot/reference.py --mode preprocess --synthetic --output reproduction/results/pi05-preprocess-01
```

请逐条执行，前一条成功再继续。tokenizer 来自受限的 Google 仓库；如果返回 401/403，在 Hugging Face 上取得该仓库访问权限并在终端执行 `HF_HUB_OFFLINE=0 hf auth login` 后重试。无需把 token 发给助手。此时尚不需要模型权重。

预处理报告应包含 normalization、prompt_format、tokenizer_exact_match 均为 PASS，action_shape=[1,50,7]，tokens_shape=[1,200]。合成样本只验证数据接口，不是校准集或成功率测试。

然后由你显式启动约 6.96 GiB 的权重下载和一次前向：

```bash
python reproduction/pi05_lerobot/download_assets.py --weights
python reproduction/pi05_lerobot/reference.py --mode forward --synthetic --output reproduction/results/pi05-forward-01
```

`--weights` 同时补齐小型 processor 配置/统计；不下载数据集。所有下载固定 revision 并核对 HF 文件摘要；若已有文件内容不同会拒绝覆盖。下载清单保存在 reproduction/logs/assets-*.json。

前向报告应包含 strict_load=PASS、forward=PASS、loss_shape=[1,50,32] 和有限的 loss_7d/loss_32d。固定噪声种子42，时间步0.5，B=1；关闭编译和梯度检查点，不执行 backward、优化、量化或完整去噪采样。首次单样本前向的实际耗时尚未测量。

输出目录必须是新目录；重跑请改成 -02 等新名称，脚本不覆盖已有结果。输出包含 report.json 和 inputs_and_reference.npz，供后续比较，不能把合成样本输出用于正式 Fisher/HSIC。

有真实 ActQuant 校准数据后，把 `--synthetic` 替换成 `--calib-dir <目录> --index 0`。校准数据必须含至少50步动作；默认不会把不足的长度静默补齐。相机0/1映射为 image/image2；还需检查数据源的相机方向和状态含义。

## 适配边界

compat.py 使用官方 config、注册的 preprocessor、图像预处理和模型。配置覆盖仅禁用编译/梯度检查点并选择设备。状态与动作 MEAN_STD 使用 checkpoint 自带统计；状态文本处理保持官方行为。没有重写 Fisher、HSIC、位宽分配或量化内核。

load_strict 使用官方键名映射后执行 strict=True；仅为模型中确实共享同一 Parameter 的缺失别名补键。这样避免上游 from_pretrained 捕获加载错误后返回未正确加载模型的情况。

参考前向报告两种损失：官方 LeRobot policy 取真实7维动作；ActQuant 原 Fisher 对模型补齐后的32维全平均。当前只是并列报告，没有擅自修改 Fisher 的目标。正式接入前须明确选择并记录；不能将两者混称同一实验。

openpi_inputs_from_reference 只提供后续比较所需的图像/mask/token/动作契约，不代表两套后端数值已经对齐。尚需真实样本的 tokenizer、图像、固定噪声前向及权重布局检查。现有 C++ 推理路径也未验证与该 checkpoint 的状态文本一致性，因此尚不能宣布闭环复现完成。

## 已做检查

- 官方模型和 processor 导入、CUDA 可见性及官方 Transformers 补丁检查通过，没有构建完整模型。
- test_compat.py 三个 CPU 检查通过：用 stub tokenizer 检查归一化/状态文本，检测人为破坏的动作尺度；缺失权重在建模前报错。此项不替代实际 tokenizer/权重验证。
- 配置解析得到 horizon50、内部动作32、token预算200、MEAN_STD。
- 新脚本 Python/Bash 语法检查通过。

可重复检查：

```bash
source deployment/activate_lerobot_reference.sh
python deployment/verify_lerobot_reference.py
python reproduction/pi05_lerobot/test_compat.py
```

### tokenizer 下载返回 401/403

浏览器登录 https://huggingface.co/google/paligemma-3b-pt-224 并接受访问条款；服务器使用同一账号的 Read token 登录。激活脚本默认离线，所以登录和身份验证须单次开启网络：

```bash
HF_HUB_OFFLINE=0 hf auth login
HF_HUB_OFFLINE=0 hf auth whoami
python reproduction/pi05_lerobot/download_assets.py --tokenizer
```

交互粘贴 token，Git credential 问题可选 n。若用 fine-grained token，还需允许读取已获账号授权的公开受限仓库。账号权限与 token 权限均满足后才能下载。下载脚本已提供简洁提示，401/403不表示Python/CUDA环境错误。

### 预处理 token 对照检查修正

真实 tokenizer 暴露了最初对照检查漏传 padding_side 的问题：Gemma tokenizer 默认左补齐，而 checkpoint 的 tokenizer_processor 显式指定右补齐。reference.py 现通过 validate_tokenizer 从 checkpoint 读取 max_length、padding、padding_side、truncation 和 task_key，再逐元素比较 token IDs 与 mask。CPU 回归使用本地真实 tokenizer，验证右补齐通过、错误左补齐被拒绝；不修改官方预处理或算法。原 stub tokenizer 测试不能覆盖该默认参数差异。
