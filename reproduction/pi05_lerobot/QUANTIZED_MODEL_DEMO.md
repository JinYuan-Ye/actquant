# 第一份完整 LLM 混合位宽量化模型

2026-10-19，用户明确要求开始量化，已完成全部126个PaliGemma目标矩阵的真实量化、统一GGUF合并和同后端固定噪声十步采样。结果为 `reproduction/results/pi05-quantized-demo-01/model/pi05.gguf`。

状态为 **ARTIFACT_AND_PAIRED_EXECUTION_PASS**：文件结构和运行检查通过；任务质量状态为 **NOT_EVALUATED**。本轮校准只用同一轨迹四帧，没有下载完整60轨迹，不代表论文规模结果或LIBERO成功率。

## 实际产物

| 项目 | 结果 |
|---|---:|
| 低位LLM目标矩阵 | 126 / 126 |
| 日志确认per-weight Fisher | 126 / 126 |
| 实际参数加权block bpw | 2.347089947 |
| 目标矩阵编码数据 | 581,435,392字节 |
| 未量化统一GGUF | 6,714,075,456字节 |
| 当前统一GGUF | 3,331,893,600字节 |
| 文件缩小比例 | 50.3745% |

126个矩阵格式：IQ2_XS 67、IQ2_S 21、Q2_K 4、IQ3_XXS 3、IQ3_S 2、IQ4_XS 4、Q4_K 25。实际GGUF格式与分配表逐项一致，编码大小重算的平均bpw也一致。

统一文件保留814个tensor，其中126个替换为量化权重，其余688个与基础模型字节一致。视觉编码器、embedding、action expert、projector、norm和归一化统计保留基础精度。2.347 bpw仅指126个目标，不能当作整份统一模型的平均位宽。

模型目录还包括tokenizer.model、norm_stats.json、RoPE频率快照、runtime_contract.json、量化及验证清单。模型SHA256：`25d52c5e7a54608b604b15322ad09fa0b0502209bd988b45c313f54f8528b41e`。

## 这次怎样使用论文方法

前一轮HSIC决定每个矩阵的格式，四样本Fisher提供每个权重元素的重要性。quantize_llm.py为全部126项显式生成锚定、转义后的--tensor-type参数，并传入完整fisher_llm.gguf，调用官方llama-quantize。没有改统计公式或量化内核。

独立LLM的token embedding保持F16，output保持原精度；合并时保留统一模型的embedding和norm。merge_gguf_compat.py复用官方权重合并，只修复元数据复制。37个基础元数据字段检查保留，包含完整的三项图像mean/std数组。

完整量化约162.71秒，合并与文件审计约13.87秒。没有覆盖此前任何结果。

## 同后端验证结果

此前C++与PyTorch的严格数值比较仍保留NUMERICAL_MISMATCH，atol/rtol没有改变。用户要求推进教学量化后，采用相同C++实现、输入、源RoPE约定和初始噪声，对比量化前后变化。它测量量化影响，不要求有损权重与未量化权重逐元素相等。

基线在 pi05-quant-baseline-01，量化模型在 pi05-quantized-forward-01。两者使用同一诊断源代码、GGML_PREC_F32请求、禁用TF32、匹配源BF16舍入RoPE频率、两路已验证浮点图像、带state的151个文本token以及663个有效prefix。全部十步ODE完成，速度、轨迹和最终动作均有限。

| 单个真实观测上的测量 | RMSE |
|---|---:|
| 固定time=0.5的32维速度 | 0.106166 |
| 十步后32维归一化动作 | 0.059747 |
| 十步后真实7维归一化动作 | 0.127351 |
| 十步后真实7维反归一化动作 | 0.028695 |

7维反归一化动作最大差约0.175827、平均绝对差约0.015781。这些是控制器命令空间的数值变化，不是米、弧度或成功率；没有设定任务质量验收门槛。不要把有限执行通过解读为动作质量通过。

ODE使用固定高斯初始noise、10步Euler更新；时间正弦嵌入按double公式计算后转float。量化前后完全相同。原高层serve_policy/Pi05Pipeline的状态prompt和mask问题没有通过本轮被宣称已修好；目前已验证的是外部固定输入运行入口，闭环接口仍需对接和验证。

## 可复现命令与说明

以下入口已准备，首轮操作已经完成，现有目录拒绝覆盖。新实验应换新输出目录。

```bash
source deployment/activate_server.sh
```

激活隔离环境与本地GGUF库，默认离线。

```bash
python reproduction/pi05_lerobot/quantize_llm.py \
  --output reproduction/results/pi05-quantized-demo-02
```

这是复现实验示例，默认读取已有四样本Fisher和HSIC分配，使用8个CPU线程，最多运行600秒。脚本自动展开126项格式覆盖，保持embedding为F16；--dry-run只打印命令，不创建结果或运行量化。本次可复用入口的dry-run命令已与首轮实际执行命令逐项核对一致。

```bash
python reproduction/pi05_lerobot/finish_llm_quantization.py \
  --run-dir reproduction/results/pi05-quantized-demo-02
```

在上一步成功后，检查126项实际格式、Fisher尺寸和实际bpw，合并完整模型至该目录的model/pi05.gguf，再检查所有tensor来源和元数据。预期artifact-audit.json为ARTIFACT_PASS。

```bash
python reproduction/pi05_lerobot/run_cpp_probe.py \
  --reference reproduction/results/pi05-cpp-reference-01 \
  --output reproduction/results/pi05-quantized-forward-02 \
  --model reproduction/results/pi05-quantized-demo-02/model/pi05.gguf \
  --mode matched_rope --ode-steps 10
```

运行新模型的固定输入和十步采样。执行状态看execution_returncode=0以及有限输出；脚本还保留与PyTorch参考的旧严格比较，有损模型会产生NUMERICAL_MISMATCH/退出码2，不能靠改容差将其变成精确一致。

```bash
python reproduction/pi05_lerobot/measure_quantization_effect.py \
  --baseline reproduction/results/pi05-quant-baseline-01 \
  --quantized reproduction/results/pi05-quantized-forward-02 \
  --model-dir reproduction/results/pi05-quantized-demo-02/model \
  --output reproduction/results/pi05-quantized-demo-02/paired-validation.json
```

检查量化前后运行源代码、设置和输入相同，再测量速度、十步轨迹、最终归一化及反归一化动作变化。预期PAIRED_EXECUTION_PASS且quality_acceptance仍为NOT_EVALUATED。

## 证据与后续

首轮证据集中在 pi05-quantized-demo-01：run.json记录完整量化命令，quantize.log确认126项per-weight，artifact-audit.json核验模型结构，paired-validation.json/npz记录动作对照，completion.json及model/validation_manifest.json记录校验和与实际完成范围。

后续正式复现需要准备完整校准集、重算HSIC/Fisher，再量化并进行LIBERO评估。当前教学版已经完成实际量化并可在已验证的固定输入入口运行；原跨后端误差报告完整保留，不再作为教学量化文件生成的阻塞条件。
