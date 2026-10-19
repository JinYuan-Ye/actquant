# 位宽分配教学实验：把 126 个分数变成格式表

2026-10-19，结果目录为 `reproduction/results/pi05-allocation-demo-01`。本轮直接调用官方 `allocate_pi05.py`，只在 CPU 上处理已有分数，没有加载或量化模型。输入来自四个同轨迹样本，因此所有格式表仅供教学，不作为正式配置。

## 已执行命令与参数

```bash
source deployment/activate_server.sh
python src/ActQuant/tools/hsic/allocate_pi05.py \
  --scores reproduction/results/pi05-hsic-demo-01/scores.json \
  --score-key F_out \
  --target-bpw 2.3471 \
  --base-type IQ2_XS \
  --max-type Q4_K \
  --output reproduction/results/pi05-allocation-demo-01/allocation.json
```

第一条命令激活隔离环境。第二条读取前一轮 HSIC 分数，选择 `F_out` 字段，在平均 2.3471 bits/weight 的预算下，为每个目标矩阵选择 `IQ2_XS` 到 `Q4_K` 范围内的格式，再输出 JSON。预算沿用当前官方 π₀.₅ 驱动的 IQ2_XS 常量。预期输出 126 个分配项且实际 bpw 不超过预算；本轮已完成，实际为 2.347089947089947。

此命令通过独立执行器运行，先独占创建新目录，并保存标准输出与审计记录。官方 allocator 自身会覆盖 `--output` 指向的文件；复查无需重跑，重做时必须换新目录/文件名。

## 预算怎样计算

目标仅包含 PaliGemma 的 18×7 个矩阵，参数合计 1,981,808,640。平均位宽是按参数量加权的：

```text
平均 bpw = Σ(矩阵参数量 × 所选格式 bpw) / Σ(矩阵参数量)
```

它不是 126 个格式位宽的算术平均，也不是完整 π₀.₅ 模型或实际文件的平均位宽。视觉编码器、action expert、词嵌入等不在本次预算中。

候选格式及注册表中的 bpw：IQ2_XS=2.3125、IQ2_S=2.5625、Q2_K=2.625、IQ3_XXS=3.0625、IQ3_S=3.4375、IQ4_XS=4.25、Q4_K=4.5。同 bpw 的 Q3_K 按上游 family preference 去重。格式名里的数字不等于包含块级尺度等开销后的 bpw。

## greedy_l2_allocate 在做什么

对应 `src/ActQuant/tools/hsic/allocate_pi05.py:199`。先把所有矩阵设成基础格式 IQ2_XS，然后估算每层的误差：

```python
importance[name] = max(F_out[name], 0)
noise[type] = 2 ** (-2 * bpw[type])
E[layer] = sum(importance[name] * noise[type[name]] for name in layer)
J = sum(E[layer] ** 2 for layer in layers)
```

这是分配器使用的代理目标，noise 不是实际对权重做量化后测出的误差。按层求和再平方，会让当前高误差层的改善获得更大收益。

每个矩阵可以升级到候选菜单中的下一种格式，代码计算：

```text
额外 bits = 参数量 × (新 bpw - 旧 bpw)
收益 = 升级前 J - 升级后 J
优先级 = 收益 / 额外 bits
```

`push` 将候选放入 heapq；负号使最小堆优先取较高收益/bit的条目。循环检查当前格式与候选阶段、剩余预算，执行升级后更新所在层的 E，并重新加入候选。它是官方贪心近似，不提供离散预算问题的全局最优保证。

因此矩阵参数量主要通过升级成本影响选择；当前 E 定义没有再乘一次参数量。不能自行把它改成另一种加权方式。

## 用实际数字理解选择

K/V 每个矩阵有 524,288 个参数，up/gate/down 每个有 33,554,432 个，后者大 64 倍。从 IQ2_XS 升到 IQ2_S 都增加 0.25 bpw：K 需要额外 131,072 bits（16 KiB），up 需要 8,388,608 bits（1 MiB）。这是分配器的存储成本估计。

本轮初始化时收益/bit最高的候选是 `blk.4.attn_v.weight`：F_out=36.734698，先升级到 IQ2_S 的额外成本为 131,072 bits，代理目标下降约 2.342739。最终它继续升级到了 Q4_K。

默认预算下的一些最终结果：

| 矩阵 | 参数量 | F_out | 格式 | bpw |
|---|---:|---:|---|---:|
| blk.0.attn_q.weight | 4,194,304 | 2.323887 | IQ2_S | 2.5625 |
| blk.0.attn_k.weight | 524,288 | 35.733411 | Q4_K | 4.5 |
| blk.0.attn_v.weight | 524,288 | 18.590404 | Q4_K | 4.5 |
| blk.0.ffn_up.weight | 33,554,432 | 0.273356 | IQ2_XS | 2.3125 |
| blk.0.ffn_down.weight | 33,554,432 | 4.087086 | IQ2_S | 2.5625 |

126 个矩阵中 59 个从基础格式升级。完整数量：IQ2_XS 67、IQ2_S 21、Q2_K 4、IQ3_XXS 3、IQ3_S 2、IQ4_XS 4、Q4_K 25。很多小矩阵升级，也可以让参数加权的平均位宽仍然接近 2.3。

## 核验发现的预算记录差异

`run_hsic_quant_pi05.sh` 的实际 IQ2_XS 默认常量为 2.3471，其注释称匹配启发式基线；但用 allocator 的 `iq2xs_heuristic()` 和参数量重算，基线实际为 2.324570105820106。官方源码保持原样，实验记录明确这处差异，不把两个预算当成相等。

因此额外执行一次相同预算分配。命令使用上面的同一组参数，只将 `--target-bpw` 改成 `2.324570105820106`，输出改为 `allocation-matched-budget.json`。目的在于与代码生成的启发式采用完全相同的参数加权预算；预期平均位宽不超这个数，实际恰好等于它。

| 配置 | 实际 bpw | 相对全 IQ2_XS 升级的矩阵数 | 代理目标 J |
|---|---:|---:|---:|
| 全 IQ2_XS | 2.312500 | 0 | 65.804640 |
| 官方启发式 | 2.324570 | 20 | 45.729797 |
| HSIC，相同预算 | 2.324570 | 31 | 2.639241 |
| HSIC，驱动默认预算 | 2.347090 | 59 | 1.044177 |

相同预算下，HSIC表在这个评分代理目标上更低；这不是实际模型损失、量化误差或 LIBERO 成功率的对照。四个同轨迹样本的评分也不能支持论文结论。

## 保存了什么，下一步是什么

- `allocation.json`、`allocation.log`：驱动默认预算下的映射和官方输出。
- `audit.json`：输入/源码哈希、格式统计、参数量、预算和代理目标重算结果。
- `allocation-matched-budget.json`、对应 `.log`、`matched-budget-audit.json`：相同预算的补充实验。

检查通过：126个名称完整，格式均在候选菜单中，注册参数量与上一轮观察到的 d_in×d_out 一致，参数加权 bpw 在预算内，独立重算 J 与官方输出一致。官方 ActQuant/OpenPI 源码未改动。

当前获得的是“每个矩阵选什么格式”的计划。后续四样本 Fisher 教学已完成，见 [FISHER_DEMO.md](FISHER_DEMO.md)：逐样本动作损失反向，累积每个权重元素的梯度平方，得到供实际量化器使用的重要性。Fisher 的数值大小不会直接改变本轮分配表；两阶段提供不同层次的信息。
