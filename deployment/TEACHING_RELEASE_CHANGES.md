# 教学版交付：相对论文仓库的改动

日期：2026-10-19。交付对象：四帧校准的 π₀.₅ LLM 混合位宽量化教学版。

## 原仓库状态

ActQuant 固定提交 `b64791125070652fe6b554e244fe809c79ef5246`，OpenPI 固定提交 `215abfb217dbac7d5f1273282331b9b1866c0479`。两者已跟踪源码没有修改，因而没有官方算法补丁。ActQuant 下的 `build_openpi` 是本地构建链接，不是源码改动，不随包复制。

所有定制代码在工作区的 `reproduction/pi05_lerobot/` 和 `deployment/`。交付包保持相对目录，并包含文件级 SHA256 清单。`upstream_reference/` 内为方便阅读而附带的少量未修改上游源码，不能据此独立构建整个项目；完整仓库仍需使用上述提交。

## 新增代码与作用

| 文件（均在 reproduction/pi05_lerobot 下） | 用途 |
|---|---|
| compat.py、reference.py | LeRobot checkpoint 配置、权重别名及官方参考输入/前向 |
| prepare_real_probe.py、export_native_calibration.py | 真实观测和原生分辨率校准数据导出 |
| runtime_precision.py、openpi_compare.py | 记录并统一矩阵乘法精度，固定输入比较 |
| calibration_adapter.py | 按 checkpoint 处理图像、状态文本、动作归一化与 padding；隔离输入兼容修复 |
| check_calibration_ready.py、check_gradient_reference.py | 激活、梯度及选定梯度的官方参考检查 |
| run_calibration.py | 调用原 HSIC/Fisher 核心函数的外部入口，不改计算公式 |
| check_quantization_bridge.py | GGUF 导出、Fisher 名称/shape、真实矩阵量化与元数据审计 |
| merge_gguf_compat.py | 修复元数据数组复制，仍使用官方 tensor 合并 |
| prepare_cpp_reference.py、cpp_forward_probe.cpp | 构造导出精度参考、固定输入 C++ 诊断与十步采样 |
| compare_cpp_forward.py、run_cpp_probe.py | 保留严格比较与执行状态，生成可重复诊断 |
| quantize_llm.py | 将126项位宽分配与逐权重 Fisher 交给官方量化器 |
| finish_llm_quantization.py | 合并完整 GGUF，核对格式、元数据和未改动权重 |
| measure_quantization_effect.py | 同后端、同输入和噪声的量化前后动作误差 |
| test_*.py | 输入与兼容工具的已有测试 |

`deployment/` 随包提供环境隔离、安装、激活、构建/渲染检查和历史核查文档。依赖版本记录在 `deployment/evidence/` 的选定清单中。原 OpenPI 要求的 transformers 替换属于其官方依赖安装步骤，独立环境保留了版本记录。

## 教学文档阅读顺序

1. `reproduction/pi05_lerobot/METHOD_WALKTHROUGH.md`：数据格式与整个方法。
2. `HSIC_DEMO.md`、`ALLOCATION_DEMO.md`、`FISHER_DEMO.md`：评分、预算分配、逐权重重要性。
3. `QUANTIZATION_BRIDGE.md`：权重与重要性如何交给 C++ 量化器。
4. `QUANTIZED_MODEL_DEMO.md`：实际量化结果、验证指标和命令解释。
5. 本目录 `TEACHING_RELEASE_ISSUES.md`：主要问题与处理结果。

## 模型与交付边界

模型位于 `reproduction/results/pi05-quantized-demo-01/model/pi05.gguf`，3,331,893,600字节。126个LLM目标全部量化，目标矩阵平均2.347089947 bit/权重；其余组件保持基础精度。模型目录内 tokenizer、归一化统计和 RoPE 快照一并交付。

校准仅为同一轨迹的4帧，不是完整60轨迹校准。交付包含四帧数据、分配表、分数及小型验证记录；不包含原始大模型、基础GGUF、约9.64GB逐权重Fisher文件、完整数据集、Python环境和CUDA构建目录。仅凭交付包不能从零重新量化；原服务器保留了这些中间产物。原推理服务器尚未完成输入适配，运行约定见模型目录的 `runtime_contract.json`。本次十步诊断执行通过，LIBERO任务质量未评估。
