# 教学版主要问题与解决方法

日期：2026-10-19。以下区分已修复的衔接问题和仍未完成的质量验证；原始实验报告随包保留。

| 问题及影响 | 处理方法 | 证据/状态 |
|---|---|---|
| 原工具默认 horizon=10，当前 checkpoint 要求50；状态8维、真实动作7维与内部32维不同 | 外部适配器读取 checkpoint，按50步动作归一化并补32维；状态按官方处理后写入文本 | calibration_adapter.py；calibration-ready-03通过 |
| 原工具未沿用 checkpoint 的 MEAN_STD、状态离散文本、BOS与右padding | 使用官方 processor 统计、完整 `Task/State/Action` prompt 与200 token规则，比较原始输入路径 | LeRobot/OpenPI输入与逐元素loss完全一致 |
| OpenPI 构造器把FP32 matmul设为high，导致TF32误差 | 构造之后显式设highest，记录实际运行精度 | openpi-forward-01不一致；forward-02逐元素EXACT_MATCH；未放宽容差 |
| OpenPI forward 内部训练增强与单样本NHWC resize挤掉batch轴 | 外部适配关闭已保存输入比较的增强，并恢复batch维 | 原始真实输入、选定梯度参考通过；不改官方仓库 |
| tied embedding 权重在checkpoint只存一份 | 验证确为共享Parameter后补同值别名，不忽略任意missing key | strict_load通过 |
| 部分末层梯度为零或未使用，看似Fisher异常 | 两个实现核查拓扑，确认末层Q结构性零梯度及O/MLP不参与该目标；保留原writer的1e-20下限 | Fisher导出288项目审计通过；不把下限当成非零原始梯度 |
| 官方默认目标2.3471与注释所称启发式预算不完全一致 | 原样保留官方默认，独立重算参数加权预算，另存匹配预算分配作教学对照 | allocation/audit及matched-budget-audit |
| 原合并器把image mean/std数组截成单项，并复制GGUF内部字段 | 外部兼容复制器使用完整contents并跳过内部头，权重合并仍调用官方实现 | 完整模型37项元数据、814个tensor审计通过 |
| 视觉Fisher的人工padding与量化输入列数不一致，可能被量化器忽略 | 有界视觉夹具测试裁去人工列后交给原工具自行padding | 3个视觉矩阵per-weight通过；本轮成品没有量化完整视觉网络 |
| 混合格式可能被量化器规则覆盖，或误用--pure禁用逐项覆盖 | 为126矩阵生成锚定并转义的明确--tensor-type；不使用--pure；逐项检查最终格式和日志 | 126项Fisher与实际格式全部通过 |
| C++原server缺状态prompt，且缓存mask/位置与PyTorch不同 | 外部诊断使用验证过的浮点图像、完整prompt并压缩masked token为663有效prefix | 输入和token核对通过；尚未把原server改成可用闭环接口 |
| RoPE非持久buffer随父模块转BF16，之后转FP32不能恢复；C++却重建理想FP32频率 | 保存源实际频率，通过公开GGML RoPE输入匹配；同后端量化前后使用相同约定 | 速度最大差从约0.018降至约4e-5；跨后端严格状态仍NUMERICAL_MISMATCH |
| 要求有损量化与原模型逐元素一致会混淆量化误差和实现差异 | 保留旧严格报告，另做相同C++实现、同观测、同噪声的量化前后十步采样 | 两者执行成功且有限；7维反归一化动作RMSE0.028695；质量NOT_EVALUATED |

## 实际解决到哪里

全部126个LLM目标真实量化并合并完整模型，统一文件由6,714,075,456降至3,331,893,600字节，减少50.3745%。文件审计及单观测同后端十步执行检查通过，不等于跨后端严格数值通过或任务成功。

本轮只使用同一轨迹四帧。正式60轨迹校准、原服务接口适配及LIBERO成功率仍未完成。历史失败和严格阈值保持原样，完整过程见 `PI05_CHECKPOINT_COMPATIBILITY.md`、`CPP_FORWARD_DIAGNOSTICS.md` 和 `QUANTIZED_MODEL_DEMO.md`。

## Windows传输现状

当前执行环境是远程Linux容器，没有挂载Windows D盘。归档和SHA256清单可在服务器生成，但要写入 `D:\actquant`，需要Windows发起SSH下载，或先将该目录共享挂载到容器。尚未完成本地接收时不能将“已打包”记成“已上传”。
