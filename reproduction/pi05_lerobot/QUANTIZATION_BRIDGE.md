# 从权重导出到实际量化：第一次衔接验证

2026-10-19，结果目录 `reproduction/results/pi05-quant-bridge-01`。官方导出器已生成完整未量化统一模型与独立 LLM；随后只对 23 个真实 LLM 矩阵和 3 个视觉矩阵运行 C++ 量化。导出数值、逐权重重要性读取、格式及解码检查全部 PASS。部分张量文件是测试夹具，不能用作完整推理模型。

## 一条参数怎样走到量化器

以 PaliGemma 第 0 层 attention 输出投影为例：

| 阶段 | 名称 |
|---|---|
| LeRobot checkpoint | model.paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.o_proj.weight |
| 独立 LLM GGUF | blk.0.attn_output.weight |
| Fisher GGUF | blk.0.attn_output.weight.in_sum2 和对应 counts |
| 统一 π₀.₅ GGUF | pali.blk.0.attn_o.weight |

量化器查找 Fisher 时使用独立 LLM 的矩阵名；量化结束后再通过 merge 将名称改回统一模型约定。名称、矩阵形状和行列顺序都需要一致，否则可能找不到重要性或直接报尺寸不匹配。

上一轮 allocation.json 是“矩阵名称→格式”的表；Fisher 是“同名矩阵→逐权重重要性”的表。GGUF 负责保存这些数据，真正的尺度和编码搜索由 C++ 量化内核完成。

## 导出实际做了什么

官方文件名 pali_llm_bf16.gguf 与实际存储类型不同：当前导出器将大矩阵写成 F16，将 norm 等小向量写成 F32。原模型是 BF16/FP32 混合精度。审计比较的是“按官方 FP16 转换后的值”与导出值，不能把这个检查称为 C++ 与 PyTorch 前向等价。

本轮逐元素/位级检查了全部 288 个量化目标矩阵：LLM126、视觉162，导出形状及数值一致；另检查共享词嵌入的三份导出表示与 checkpoint 的 lm_head 权重相符，以及 36 组层归一化的转换。统一模型全部张量值也通过有限性扫描。

共享词嵌入在独立文件里写成 token_embd.weight 和 output.weight 两份相同数据，在统一文件里写成 embed.weight；多份相同数据不表示它们在文件中共享一段存储。合并默认保留统一模型的 embedding。

Gemma RMSNorm 使用 (1+w)。独立 LLM 为适配 llama 容器约定，把 norm 存成 1+w；统一 π₀.₅ 仍存原始 w，由其运行时处理。原 merge 的 norm 跳过规则正确，兼容入口沿用它，避免重复加 1。独立 LLM 使用 llama 架构容器供量化器读取，不是对 Gemma 前向语义等价的验证。

完整未量化文件：unified/pi05.gguf 为 6,714,075,456 字节；llm/pali_llm_bf16.gguf 为 6,076,576,864 字节。统一元数据已核验 horizon=50、真实 action_dim=7、state_dim=8。

## 发现并修复的衔接问题

### 1. 合并时数值数组被截断

固定版本 merge_pi05_llm.py 的元数据复制函数用 field.parts[-1] 读取数值数组。在当前 reader 下，图像均值和标准差会从 [0.5,0.5,0.5] 变成 [0.5]。它还会将 GGUF 内部头字段当作普通元数据复制，产生重复字段提示。

新增外部入口 merge_gguf_compat.py：用 field.contents() 完整读取元数据、跳过内部 GGUF 头字段，权重合并仍调用官方 merge。小型真实张量文件复现原问题，并验证兼容入口保留全部37个待核验字段和4个张量字节，norm未被重复加1。证据在 metadata-merge-probe/audit.json。官方仓库未修改。

### 2. 视觉重要性需要按工具的补齐阶段对齐

当前 Fisher writer 把视觉 up 的重要性导出成 [4352,1280]；未量化统一模型的 up 是 [4352,1152]。quantize-vision 要先用**补齐前**的形状查找重要性，再自行补齐；直接传入前者会因形状不一致而忽略它。

外部测试入口只截去重要性末尾128列已知的人工补齐区，保留全部真实权重对应的值，再交给量化器。C++ 按自己的原有规则将权重和重要性补到1280列。Q 的重要性原本就是 [1152,1152]，可直接对应输入；down 已补到 [1152,4352]，也无需裁剪。没有改变梯度平方统计或量化算法。

本轮三个视觉矩阵日志全部出现 [imatrix: per-weight]，没有 missing 或 ignoring。此测试使用 Q4_K_M（视觉工具将其解析为 Q4_K 格式）。完整视觉重要性文件仍保留原样。

## 实际量化结果

LLM测试选取全部18个V矩阵及另外5个代表矩阵，共23个。包含所有V是为了满足量化器的attention层数一致性断言，同时覆盖多种格式、MLP和末层全floor重要性。输入只有这些矩阵，文件不具备完整推理结构。

23项全部按 allocation.json 的格式输出，日志逐项出现 (per-weight)，Python官方解码器还原后形状正确、数值有限。例如：

| 矩阵 | 输出格式 | 原F16权重大小 | 量化权重大小 | 相对导出F16的普通MSE |
|---|---|---:|---:|---:|
| blk.0.attn_q.weight | IQ2_S | 8 MiB | 1.28125 MiB | 8.3588e-5 |
| blk.0.attn_k.weight | Q4_K | 1 MiB | 0.28125 MiB | 3.5001e-6 |
| blk.0.ffn_up.weight | IQ2_XS | 64 MiB | 9.25 MiB | 1.4506e-4 |
| blk.17.attn_q.weight | IQ2_XS | 8 MiB | 1.15625 MiB | 7.2851e-5 |

大小只统计该张量的编码数据，未包含GGUF头和tokenizer。MSE用于核对解码结果，不是Fisher加权目标，也没有与无Fisher量化做质量对照；不能由此推断LIBERO效果。

主测试对全部目标显式传 --tensor-type，并使用锚定且转义的矩阵名称正则。另复查了官方“只覆盖升级项”的驱动策略：在当前 IQ2_XS 固定版本及同一23项夹具上，输出格式同样完全一致。源码对IQ2_XS已禁用默认升级，因此未复现格式覆盖问题；不能扩大到其他基础预设。不要加 --pure 来试图强制自定义表：当前C++代码将逐矩阵覆盖处理放在 !pure 分支内。

## 已执行的入口与参数解释

两份导出都调用官方脚本。新目录的 checkpoint-view 仅用只读符号链接提供原模型文件和已下载 tokenizer，没有修改 models/pi05_libero。

```bash
python src/ActQuant/tools/pi0.5/export_pi05.py \
  -d reproduction/results/pi05-quant-bridge-01/checkpoint-view \
  -o reproduction/results/pi05-quant-bridge-01/unified
```

这是统一模型导出，-d 指输入视图，-o 指新输出目录；没有设置低位量化参数，得到完整未量化 pi05.gguf、tokenizer与统计量。运行时显式使用 PI05_OUTPUT_PRECISION=fp16。

```bash
python src/ActQuant/tools/pi0.5/export_pi05_llm.py \
  -d reproduction/results/pi05-quant-bridge-01/checkpoint-view \
  -o reproduction/results/pi05-quant-bridge-01/llm
```

这是独立LLM导出，沿用官方命名与norm转换。导出前已将本地 tokenizer.model 放入 llm 输出目录，以便脚本嵌入词表。预期为矩阵F16、norm F32的文件，审计已核实。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python reproduction/pi05_lerobot/check_quantization_bridge.py \
  --export-dir reproduction/results/pi05-quant-bridge-01 \
  --work-name selected-03 --cpu-threads 4
```

这条命令限制CPU线程，先核对已有导出，再生成小型张量夹具，调用本地C++量化器，核验per-weight日志、实际格式和解码值。--work-name指定新建子目录，避免覆盖旧结果。以上已经执行，不需重跑；现有 bridge-audit.json 和测试目录受到拒绝覆盖检查。

主检查耗时约50.75秒。较早两个只读扫描因CPU/FP16断言开销中止，空的 selected/selected-02 保留，未出现数据不匹配；最终 selected-03 采用四线程和FP16原始位比较通过。

完整合并使用 merge_gguf_compat.py 的 --base、--llm、--output，分别指定统一基础文件、量化LLM和新输出路径。本页记录的衔接实验只运行了小文件验证；后续完整量化和合并已经执行，见 [QUANTIZED_MODEL_DEMO.md](QUANTIZED_MODEL_DEMO.md)。

## 接下来

后续已完成统一GGUF的C++诊断，并按用户要求采用同后端量化前后配对比较，生成全部126个LLM目标量化后的完整模型，十步动作采样输出有限。产物与误差指标见 [QUANTIZED_MODEL_DEMO.md](QUANTIZED_MODEL_DEMO.md)。[跨后端严格诊断](CPP_FORWARD_DIAGNOSTICS.md)仍为NUMERICAL_MISMATCH，不能与同后端执行检查混淆。完整60轨迹校准和LIBERO闭环仍未完成，四样本实验不能作为正式论文结果。
