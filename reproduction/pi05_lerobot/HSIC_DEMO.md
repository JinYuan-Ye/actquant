# 第一次 HSIC 小实验：从 4 个样本到 126 个分数

2026-10-19，`reproduction/results/pi05-hsic-demo-01` 已完成，status=PASS。采用现有同轨迹 t=0/10/20/30 四帧，未下载数据、未修改官方算法、未更新模型权重。此次只是流程与教学实验，不能据此确定正式量化配置。

## 已执行的命令

```bash
source deployment/activate_server.sh
python reproduction/pi05_lerobot/run_calibration.py \
  --stage hsic \
  --calib-dir reproduction/data/pi05-readiness-01 \
  --num-samples 4 \
  --output reproduction/results/pi05-hsic-demo-01
```

第一条命令激活 OpenPI/ActQuant 的隔离环境，配置本地包路径并默认离线。第二条命令选择 HSIC 阶段，从指定目录读取四个样本，采集激活并计算评分；`--num-samples 4` 明确限制规模，`--output` 是新建结果目录。预期是严格加载通过、生成 126 个有限分数和 PASS 报告。以上已经执行，不需要重复运行；脚本拒绝覆盖同名目录。

本轮使用默认 seed=42、batch=1、pool_tokens=0、hidden RBF 核、Y linear 核、lx=ly=1、CPU HSIC。模型前向在 GPU 上执行。模型构造后设 highest，禁用额外图像增强；这两项延续已经验证的 LeRobot 兼容路线。报告计时约 35.91 秒，峰值 GPU 分配内存 7.116 GiB。

## 看懂一次采集

一个样本包含当前观测和接下来 50 步示范动作。输入文件与归一化细节见 METHOD_WALKTHROUGH.md；这里关注前向之后留下什么。

对线性层而言，PyTorch 的权重 shape 是 `[d_out,d_in]`，运算相当于 `output = input @ weight.T`。hook 在该层执行时保存输入和输出，并沿前缀 token 轴求平均。所以每个样本、每个矩阵各留下一个输入向量和一个输出向量。

| 变量 | 本次 shape | 含义 |
|---|---|---|
| X | [4,2048] | 第 0 层输入前缀的平均表示 |
| Y | [4,350] | 50×7 示范动作展平，按列在 4 个样本间标准化 |
| 第 0 层 Q 的 Z_in / Z_out | [4,2048] / [4,2048] | Q 投影输入 / 输出 |
| 第 0 层 K 的 Z_in / Z_out | [4,2048] / [4,256] | K 投影输入 / 输出 |
| 第 0 层 up 的 Z_in / Z_out | [4,2048] / [4,16384] | MLP 升维投影输入 / 输出 |
| 第 0 层 down 的 Z_in / Z_out | [4,16384] / [4,2048] | MLP 降维投影输入 / 输出 |

这里 4 是校准样本数，不是 token 数或层数。`Y` 的 350 是真实 7 维动作 × 50 步，不包含模型输入的 25 个补零动作维度。Y 的样本间标准化也不同于 checkpoint 的动作 MEAN_STD 归一化。

## 从向量变成 HSIC

核函数把每组表示转换成样本间的相似度矩阵。本次只有 4 个样本，因此 `K_X`、`K_Z`、`K_Y` 都是 `[4,4]`；`K_Z[i,j]` 表示第 i、j 个样本在这个中间表示上的相似程度。

实际调用的估计式位于 `src/ActQuant/tools/hsic/hsic_utils.py` 的 `hsic`：

```text
H = I - 11ᵀ/N
HSIC(K,L) = trace(K H L H) / (N-1)²
```

它衡量两套样本相似关系的统计依赖。此处没有用同文件里的 `hsic_normalized_cca`，也不能把结果解释为落在 [0,1] 的相关系数、概率或任务成功率。

hidden 使用 RBF，Y 使用线性核 `Y @ Y.T`。默认 sigma=None 时，本地 RBF 函数先取样本两两**平方距离**的中位数，再按代码对它平方构造分母；复现时应读实际实现，不能凭常见带宽公式自行改写。

## 用真实数字读评分

`compute_tensor_hsic_cpu` 对每个矩阵计算：

```python
F_out = -lx * hsic_x_out + ly * hsic_y_out
F_in = -lx * hsic_x_in + ly * hsic_y_in
sens = F_out - F_in
```

本次 lx=ly=1，第 0 层 Q 矩阵实际得到：

```text
HSIC(X, Z_out) = 0.0000418312634
HSIC(Z_out, Y) = 2.323928617026
F_out          = 2.323886785762
F_in           = 12.638604693689
sens           = -10.314717907926
```

负的 sens 只表示这个组合评分在输出处低于输入处，不表示程序失败，也不能单凭这个值判定矩阵无用。默认分配器读取 `F_out`。目标形式奖励与动作目标的依赖、扣除与初始输入的依赖；这是一种评分代理，不是对任务因果贡献的直接测量。

| 矩阵 | F_out | sens |
|---|---:|---:|
| blk.0.attn_q.weight | 2.323887 | -10.314718 |
| blk.0.attn_k.weight | 35.733411 | 23.094806 |
| blk.0.ffn_up.weight | 0.273356 | -10.854739 |
| blk.0.ffn_down.weight | 4.087086 | -12.074758 |

126 个 F_out 均为正，范围约 0.0000976444～36.734698。本次最高项是 blk.4.attn_v.weight；四个相邻轨迹窗口高度相关，这个排序不能当作整个 LIBERO 的重要性排序。

## 结果文件和代码阅读顺序

- `run.json`：PASS、输入来源、精度设置、官方源码哈希、耗时和显存。
- `indices.npy`：本次读取数据集中的行号 [0,1,2,3]，对应原轨迹 t=0/10/20/30。
- `activations.npz`：X、Y 及每个矩阵的输入/输出激活。沿用上游 FP16 缓存格式；本轮评分使用内存里的 FP32 数组，不能要求从 FP16 缓存重算后逐位相等。in_acts/out_acts 为对象字典，只加载自己生成且可信的缓存。
- `scores.json`：126 个矩阵的 HSIC、F_out、F_in、sens、维度，以及核函数设置。

已额外核验：矩阵名称恰好为 18×7 个目标；所有缓存激活有限、维度与评分记录一致；126 项 F_out 和 sens 均满足上述公式。官方 ActQuant/OpenPI 源码未改动。

建议按以下顺序读代码：

1. `compute_hsic_pi05.py:89` 的 collect_activations_pi05：hook 如何收集激活。
2. `compute_hsic_pi05.py:195`：Y 如何标准化。
3. `compute_hsic_pi05.py:324` 的 compute_tensor_hsic_cpu：核矩阵与分数。
4. `allocate_pi05.py:199` 的 greedy_l2_allocate：预算下如何使用 F_out。

后续已用这个教学评分完成位宽预算分配，见 [ALLOCATION_DEMO.md](ALLOCATION_DEMO.md)，四样本 Fisher 累计也已完成，见 [FISHER_DEMO.md](FISHER_DEMO.md)。有限矩阵实际量化验证见 [QUANTIZATION_BRIDGE.md](QUANTIZATION_BRIDGE.md)。完整60轨迹校准集、整模型低位量化和C++前向/闭环仍待后续。
