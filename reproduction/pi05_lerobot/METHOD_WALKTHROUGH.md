# 从已修好的输入进入 ActQuant 方法

截至2026-10-19：已完成4样本HSIC、位宽分配、Fisher，以及全部126个LLM目标矩阵的真实混合位宽量化。完整统一GGUF为3,331,893,600字节，同后端量化前后固定噪声十步采样均成功，详见 [QUANTIZED_MODEL_DEMO.md](QUANTIZED_MODEL_DEMO.md)。算法讲解见 [HSIC_DEMO.md](HSIC_DEMO.md)、[ALLOCATION_DEMO.md](ALLOCATION_DEMO.md) 和 [FISHER_DEMO.md](FISHER_DEMO.md)。跨后端严格数值状态仍为NUMERICAL_MISMATCH，保留原判据和 [诊断记录](CPP_FORWARD_DIAGNOSTICS.md)。样本来自同一轨迹四帧，不是论文60条轨迹校准集；视觉与action expert等保持基础精度，正式规模校准及LIBERO闭环仍待后续。

## 验证了什么

- pi05-openpi-forward-02：对齐FP32矩阵乘法运行设置后，1600个逐元素loss完全一致。报告里两个平均值末位略有不同，来自CPU NumPy与GPU PyTorch归约；实际数组差值为零。
- pi05-calibration-ready-03：从原始256图像、状态、动作、任务直接构造模型输入，与保存的LeRobot参考完全一致；原始输入前向也完全一致。4帧上官方126个PaliGemma矩阵的激活采集成功；288个目标矩阵的反向接口检查通过。峰值分配显存约11.77GiB。
- pi05-gradient-reference-01：带符号的预测速度、loss以及抽查的4个参数梯度与LeRobot完全一致。分别检查了早期LLM Q、末层LLM K、末层LLM Q、早期视觉Q。不是全部参数、全部样本的梯度等价证明。
- 原始输入适配器5项CPU测试通过；此前OpenPI对照器7项CPU测试通过。
- 官方ActQuant/OpenPI仓库未修改。新增脚本位于 reproduction/pi05_lerobot。

新的分工：助手继续负责兼容性修复和必要的有界GPU诊断。正式方法实验开始前先讲清输入和代码；不自动启动长时间实验、大型下载或覆盖旧结果。

## 1. 一个样本是什么

一个样本 = t时刻的两路图像 + t时刻8维机器人状态 + 一条任务描述 + 从t开始的50步示范动作。

演示轨迹是一串这样的时刻。相邻样本的未来动作窗口可以重叠；50步动作是监督目标，不是50张当前图像，也不是batch size=50。

当前可读的数据目录为 reproduction/data/pi05-readiness-01，含4个起点：原轨迹第0、10、20、30帧。原轨迹有110帧，四个窗口均有完整50步，不补尾。

| 文件 | dtype / shape | 含义 |
|---|---|---|
| pixel_values.npy | uint8 [N,3,256,256,3] | 样本、相机、高、宽、RGB；相机0为主视角，1为腕部，2为占位零图 |
| state.npy | float32 [N,8] | 末端位置3维、轴角3维、夹爪关节位置2维 |
| gt_actions.npy | float32 [N,50,7] | 每个观测对应的50步相对控制动作，最后一维为夹爪控制 |
| prompts.json | N个字符串 | 原始任务描述，不应提前把离散状态拼进去 |
| metadata.json | JSON | 样本数、源轨迹、相机顺序、native图像标记、动作窗口与补尾策略 |

这里N=4。像素是0到255整数；状态和动作保留RLDS原值。动作是控制器缩放前的输入，不能直接解释成每步移动多少米。占位相机不会作为真实第三相机使用。

## 2. 数据如何进入模型

对应 calibration_adapter.py 的 InputAdapter.build：

```python
state = (raw_state - state_mean) / (state_std + eps)
actions = (raw_actions - action_mean) / (action_std + eps)
# 统计量读取checkpoint的processor safetensors，不从4个样本重估。
```

状态补齐到32维后，按官方分箱方式形成文本：

```text
Task: pick up the black bowl ..., State: -1 90 255 ...;\nAction:
```

tokenizer把这段文本转换成 [B,200] token IDs及mask；显式右补齐。当前真实任务有151个有效token。MEAN_STD不保证数值在[-1,1]，不能为了消掉-1/255分箱值而裁剪状态。

图像转换为float，经过官方等价缩放得到两路 [B,3,224,224]、范围[-1,1]；增加第三路全-1占位图，图像mask为[true,true,false]。原OpenPI工具在B=1时会意外挤掉NHWC batch轴，外部适配器已补回，并测试B=1/B=2一致性。

动作归一化后从[B,50,7]补零到[B,50,32]。连续state字段仍保留，但π₀.₅主要通过前缀文本获得状态；不能只传连续state而漏掉状态文本。

## 3. 先认识我们测量的动作损失

当前实现使用flow matching。令a为归一化并补齐的示范动作，epsilon为高斯噪声，t为时间步：

```python
x_t = t * epsilon + (1 - t) * a
target_velocity = epsilon - a
predicted_velocity = model(images, tokens, x_t, t)
per_element_loss = (predicted_velocity - target_velocity) ** 2
loss = per_element_loss.mean()  # 当前ActQuant Fisher原实现：平均50×32维
```

这段是概念伪代码，实际计算见 openpi/models_pytorch/pi0_pytorch.py 的 forward。校准时噪声和时间步会按上游函数采样；前面的对照固定noise/time，只是为了比较两套实现。

LeRobot公开policy.forward通常只平均真实7维；原ActQuant Fisher平均32维。本适配保留原Fisher的32维目标，并固定batch size=1。不能把7维loss偷偷替换进Fisher，也不能把batch平均梯度的平方说成逐样本梯度平方的平均。

## 4. Stage 1：为每个矩阵选择量化格式

PaliGemma有18个block，每个block考察7个矩阵：Q、K、V、O、gate、up、down，共126个。一个矩阵选一种量化格式，矩阵内部的每个权重元素不会分别选择位宽。

第一步只做前向并通过hook取激活，不算梯度：

| 数据 | 本地代码中的含义 | 当前4样本shape |
|---|---|---|
| X | 第0层输入的前缀token平均表示 | [4,2048] |
| Y | 原始50×7动作展平，再跨样本标准化 | [4,350] |
| Z_in / Z_out | 每个线性矩阵的输入/输出，沿前缀token求平均 | [4,d_in] / [4,d_out] |

注意：上游pool函数直接平均前缀token，没有按有效mask重新加权；默认pool_tokens=0。此处保持其行为。Y的跨样本标准化用于统计依赖估计，与模型动作输入的checkpoint归一化是两个步骤。

本地 compute_hsic_pi05.py 计算：

```python
F_out = -lx * HSIC(X, Z_out) + ly * HSIC(Z_out, Y)
F_in  = -lx * HSIC(X, Z_in)  + ly * HSIC(Z_in, Y)
sens  = F_out - F_in
```

HSIC衡量两组表示在样本之间的统计依赖。脚本同时输出F_out和sens；官方驱动及分配器默认使用F_out，不应把两个字段混叫同一个分数。论文式(5)对应输出表示的评分形式。当前代码默认hidden用RBF核、Y用linear核；论文正文的核描述不能代替实际命令参数，实验要记录这些选项。

随后 allocate_pi05.py 的 greedy_l2_allocate 在平均位宽预算内分配格式。代码把负分截为0，并用 noise≈2^(-2·bpw) 估计量化误差；每次优先选择单位额外bit能带来较多目标下降的矩阵升级。它优化按层汇总误差的平方和，避免只保护少数层。bpw包含格式开销，IQ2_XS等格式的实际bpw并不恰好等于整数2。

主要阅读位置：compute_hsic_pi05.py 的 collect_activations_pi05、compute_tensor_hsic_cpu/gpu；allocate_pi05.py 的 greedy_l2_allocate。

## 5. Stage 2：固定格式后，保护矩阵内部重要的权重

本地 compute_fisher_pi05.py 的 fisher_step 是核心：

```python
per_elem = model(obs, actions)
loss = per_elem.mean()
grads = torch.autograd.grad(loss, target_weights, allow_unused=True)
for name, grad in ...:
    if grad is not None:
        accumulator[name] += grad.detach().float().pow(2).cpu()
# 所有样本结束后，除以样本数。
```

每个权重元素获得一个梯度平方平均，作为动作损失敏感性的近似。不会optimizer.step，不更新模型权重。π₀.₅只有这里使用的连续动作通路，论文AMF在alpha=1时退化为action-only；OpenVLA-OFT才涉及另外的分类通路混合。

get_target_params目前选126个LLM矩阵和162个视觉矩阵；projector与action expert不属于这些量化目标，但它们仍参与前向/梯度传播。

末层前缀输出不会再被动作头消费，因此最后一个LLM block的O、gate、up、down没有到动作loss的梯度路径，Q梯度为零；K/V仍影响动作token。已在两套实现中确认。这些项按上游allow_unused逻辑保留零累积，写GGUF时采用上游数值下限处理，不通过伪造梯度来“修复”。

write_imatrix_gguf将每个矩阵的逐元素重要性写成 `<矩阵名>.in_sum2`，还有counts等元数据。这里GGUF是文件容器，不是量化算法本身。

## 6. 位宽分配与Fisher如何一起使用

Stage 1给出矩阵→量化格式，例如blk.0.attn_q.weight→IQ2_XS。Stage 2给出该矩阵每个权重的重要性。量化内核用这两类输入，在已有格式允许的表示里选尺度/编码，尽量降低重要性加权的重建误差：

```text
Σ_i importance_i × (weight_i - reconstructed_weight_i)^2
```

这描述优化目标，不意味着各类IQ/K量化格式都用同一条闭式公式。具体实现见 src/llama-quant.cpp 的per-weight imatrix分发，以及 ggml/src/ggml-quants.c 的 quantize_row_iq2_xs_impl 等函数。Fisher不是推理时额外加入的补偿项。

## 7. 当前入口与接下来的实验

run_calibration.py是单GPU调用入口，复用官方采集、HSIC计算、fisher_step和GGUF writer；只增加输入/加载适配与独立输出目录。它不依赖原来硬编码的 `/path/to` 路径，也不需要把tokenizer复制进模型目录。默认B=1、highest、无额外图像增强，完整记录这条兼容路线的选择。

讲解完成后，用户要求继续，已用当前4帧完成HSIC、预算分配和Fisher累计，见三份DEMO文档；权重导出及有限矩阵量化衔接也通过，见QUANTIZATION_BRIDGE.md。4个同轨迹样本只能用来学习和验证运行，不足以支撑可靠的矩阵排序或论文结论。下一步验证完整未量化统一GGUF的C++输入/前向。

完整规模沿用官方10/10/10/30轨迹分层采样，seed42，抽取所有帧；episode尾部按原导出器重复最后动作。export_native_calibration.py可在四套RLDS完整到位后复用原采样逻辑，只保留256原图。当前只下载了一个分片，完整数据约10GB尚未备齐，未自动下载。

主要文件导航：

- calibration_adapter.py：原始文件→模型输入，严格加载和highest设置。
- run_calibration.py：单GPU实验调用；4样本HSIC与Fisher累计已通过，正式规模校准未执行。
- src/ActQuant/tools/hsic/compute_hsic_pi05.py：hook及HSIC评分。
- src/ActQuant/tools/hsic/allocate_pi05.py：126矩阵的预算分配。
- src/ActQuant/tools/fisher-diag/compute_fisher_pi05.py：梯度平方、累计、GGUF重要性输出。
- src/ActQuant/src/llama-quant.cpp 与 ggml/src/ggml-quants.c：消费分配及重要性，执行低位量化。

论文与官方来源：https://arxiv.org/pdf/2605.24011 ，https://actquant.github.io/ 。上述运行细节以本地固定提交代码为准；这条路线是带LeRobot兼容适配的复现，不能声称原脚本无需修改即可端到端复现。
