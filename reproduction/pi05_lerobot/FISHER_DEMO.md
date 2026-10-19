# Fisher 教学实验：为每个权重元素计算重要性

2026-10-19，四样本实验已完成：`reproduction/results/pi05-fisher-demo-01/run.json` 与 `export-audit.json` 均为 PASS。沿用前两轮的四个真实样本，没有下载新数据或修改官方源码。当前结果用于学习与流程核验，不作为正式量化配置。

## 已执行的命令

```bash
source deployment/activate_server.sh
python -u reproduction/pi05_lerobot/run_calibration.py \
  --stage fisher \
  --calib-dir reproduction/data/pi05-readiness-01 \
  --num-samples 4 \
  --output reproduction/results/pi05-fisher-demo-01
```

第一条命令激活隔离的 OpenPI/ActQuant 环境，默认离线。第二条选择 Fisher 阶段，读取指定目录中的四个样本，逐样本前向/反向、累积梯度平方并导出 GGUF；`-u` 使进度及时输出。默认 seed=42、batch=1、highest、无额外图像增强，延续已审计的 LeRobot 输入兼容路线。`--output` 必须是新目录，现有结果无需重跑。

预期得到 LLM 126 个、视觉 162 个目标矩阵的重要性以及四个有限的 loss。实际全部通过，耗时约 77.66 秒，峰值 GPU 分配内存 11.8003 GiB，进程结束后显存已释放。

## 核心算法和代码

阅读 `src/ActQuant/tools/fisher-diag/compute_fisher_pi05.py:313` 的 `fisher_step`。其主要逻辑为：

```python
per_elem = model(obs, actions)  # [1,50,32]，flow-matching 平方误差
loss = per_elem.mean()
grads = torch.autograd.grad(loss, target_weights, allow_unused=True)
for name, grad in ...:
    if grad is not None:
        accumulator[name] += grad.detach().float().pow(2).cpu()
# run_calibration.py 在四个样本结束后逐元素除以 4。
```

对一个权重元素 w_i，最终统计量为：

```text
F_i = (1/N) Σ_n (∂L_n/∂w_i)²
```

本实现用动作 MSE 的经验梯度平方平均作为 Fisher 对角重要性近似。它不构造参数两两之间的完整 Fisher 矩阵，也不是严格概率模型下 Fisher 信息的直接计算。

必须区分“先平方再平均”和“先平均再平方”。例如两个样本的梯度为 +0.2 和 -0.2：前者得到 0.04，后者得到 0，会丢掉两个样本分别具有敏感性的信息。本入口固定 batch=1，因此原 `fisher_step` 的 batch 平均不会让不同样本梯度提前抵消。

`loss` 平均全部 50×32 个元素，包含真实 7 维和 25 个补齐维度，沿用原 ActQuant 统计定义。改变 loss 的尺度会按平方改变梯度平方统计；不能随意替换成只平均 7 维的 loss。噪声与时间步由上游 forward 采样，此处没有固定 time=0.5，因此数值也不应与此前固定 noise/time 的对照 loss 直接比较。

虽然函数名叫 `freeze_non_target`，它的实际实现是将所有参数的 requires_grad 设为 True，让计算图完整连通；`autograd.grad` 只请求目标参数的梯度。此过程没有 optimizer.step，不更新权重。

## 本次结果

四个样本的 loss，按原轨迹 t=0/10/20/30 顺序为：

| 样本起点 | 动作 loss，mean32 |
|---|---:|
| 0 | 0.059598677 |
| 10 | 0.054295719 |
| 20 | 0.081426352 |
| 30 | 0.113245301 |

均值为 0.077141512。这是四次校准前向的误差，不能用它宣称训练改善或任务成功。

每个目标权重矩阵都有同形状的重要性矩阵。例如 `blk.0.attn_q.weight` 的权重和重要性都是 [2048,2048]，即 4,194,304 个独立数值。导出后该矩阵的重要性均值约 7.2014e-12，最大值约 5.4195e-7。绝对值很小与 loss 平均尺度等因素有关；量化器使用的是同一矩阵内的重要性分布，不能据此判断矩阵“不重要”。

## GGUF 中存的是什么

| 文件 | 目标矩阵数 | GGUF tensor 数 | 文件大小 |
|---|---:|---:|---:|
| fisher_llm.gguf | 126 | 252 | 7.383 GiB |
| fisher_vision.gguf | 162 | 324 | 1.599 GiB |

每个矩阵包含两项：

```text
blk.0.attn_q.weight.in_sum2  # float32 [2048,2048]，已按样本取平均的重要性
blk.0.attn_q.weight.counts   # float32 [1,1]，值为 1
```

这里 counts=1，因为 in_sum2 已经是均值；样本数 4 另存于 fisher.num_samples / imatrix.chunk_count 元数据，不应把 counts 擅自改成 4。重要性文件体积取决于目标权重元素数和 FP32 格式，增加校准样本主要增加计算量，不会按样本数线性增加最终重要性文件大小。

视觉 MLP 延续官方导出 padding：up 从 [4304,1152] 补成 [4352,1280]，down 从 [1152,4304] 补成 [1152,4352]，以对应预期的视觉 GGUF 权重布局。导出审计只验证了这一文件约定；尚未用实际导出的模型权重和 C++ 完整运行验证。

## 数值下限与末层零梯度

`write_imatrix_gguf` 位于原 Fisher 文件第 606 行。它沿用官方规则：把每个矩阵的小值抬到正值均值的 1e-6 倍；全零矩阵使用 1e-20。这是为了避免部分低位量化块尺度退化造成崩溃，导出值不完全等于原始梯度平方。

末层前缀 Q 梯度全零，O/gate/up/down 没有到动作 loss 的路径，已在此前 LeRobot/OpenPI 对照中确认。四样本累计后，这五个矩阵导出为全 1e-20（FP32 表示约 9.99999968e-21），此次全量审计再次验证一致。最后一层 K/V 仍有非零重要性；没有伪造梯度修补这些结构性项。

导出审计逐块读取所有重要性元素，检查全部为有限正 FP32；检查 288 个名称和维度与 checkpoint 头一致、counts=1、样本元数据、视觉 padding 为本矩阵数值下限，以及恰好上述五项全为 1e-20。报告位于 `export-audit.json`，含逐矩阵最小值、最大值、均值和形状。这里只验证导出完整性，不扩大为所有梯度与 LeRobot 逐元素等价的证明。

## 怎样与位宽分配配合

上一轮的 `allocation.json` 决定矩阵采用哪种格式；本轮的 GGUF 为矩阵内每个权重提供重要性。Fisher 阶段会计算全部目标，不需要读取位宽分配表。

量化器使用这两种输入，按格式允许的编码近似降低重要性加权重建误差：

```text
Σ_i F_i × (w_i - reconstructed_w_i)²
```

这是目的说明，各格式的实际编码/尺度搜索仍按各自内核执行。`src/ActQuant/src/llama-quant.cpp:968` 附近按 imatrix 元素数识别 per-weight 数据，然后分发到相应量化内核。Fisher 重要性只参与离线量化，不成为推理时额外的补偿模块。

后续已完成 LeRobot checkpoint 到 GGUF 的导出映射、tied embedding、视觉 padding 与格式覆盖检查，以及23个LLM/3个视觉矩阵的有界量化测试，见 [QUANTIZATION_BRIDGE.md](QUANTIZATION_BRIDGE.md)。下一步验证完整未量化统一GGUF的C++输入/前向。完整60轨迹校准与C++闭环仍未验证；四样本产物不能冒充正式论文结果。
