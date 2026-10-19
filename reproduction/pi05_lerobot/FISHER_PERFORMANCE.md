# Fisher低GPU利用率：瓶颈与等价加速

2026-10-19，用户指出Fisher很慢且显卡未充分使用，已定位并处理。

## 慢在哪里

原fisher_step对每个目标执行：

```python
accumulators[name] += gradient.detach().float().pow(2).cpu()
```

288个矩阵合计2,392,879,104元素，每帧约9.57GB FP32梯度平方从GPU传到CPU。实测累积部分4.56秒，其中平方及D2H约4.23秒，CPU加法约0.33秒。将CPU线程从56降到4仍为4.49秒，主要问题是每帧同步传输，不是显存装不下或必须计算十几个小时。

## 改了什么

`fisher_gpu_accumulation.py`保留逐样本mean32 loss、相同autograd目标、FP32平方、样本相加顺序和最终平均，把累积器留在GPU：

```python
accumulators[name] += gradient.detach().float().pow(2)
```

每200帧快照及最终导出时才搬回CPU。`GPUAccumulatorMap.pop`在导出时转CPU，最终仍调用原GGUF writer。`run_fisher_gpu.py`不修改已有run_calibration.py字节，能够复用原脚本严格绑定代码哈希的断点，恢复相同累加器和NumPy/Torch/CUDA随机状态。外部执行适配器的独立哈希和验证证据记录于runtime_precision。

包装器关闭每帧empty_cache，让CUDA分配器复用内存，并将CPU线程设为8。原论文仓库源码、统计公式、样本数、batch size=1和量化内核未改。不能用“增大batch后对平均梯度平方”替代逐样本平方再平均，两者一般不相同。

## 如何验证

- `accumulation-performance.json`：288个真实矩阵形状，CPU/GPU累积逐元素完全一致；GPU累积部分约0.085秒。
- `gpu-fisher-validation.json`：真实模型的第0、1216帧，以完全相同输入和随机状态分别调用原实现和GPU实现。两个样本的loss、全部288个累积矩阵逐元素完全一致；快照及导出转换通过。
- 真实单步：原实现4.58/4.18秒，GPU实现0.224/0.219秒。此为两个样本的小范围测量，不把它冒充整次实验加速比。
- 从原第200帧检查点恢复后，重放的201～219帧打印loss与原记录一致；实际执行已继续超过该范围。

此前“约13–14小时”是原慢路径按4秒/帧外推。切换后需要按GPU执行的实际速度重新估计，并计入快照IO、最终导出及后续量化，不能继续沿用旧估计。

## 当前入口与证据

全流水线使用 `run_full_pipeline.py --fisher-execution gpu`。恢复时保留原stage已完成结果，并传入--resume；本次从200帧断点恢复，旧CPU执行摘要保存在 `reproduction/results/pi05-full-01/fisher-cpu-execution/`，切换记录为 `fisher-execution-switch.json`。小型验证报告随源码发布。

直接验证入口 `validate_gpu_fisher.py --output 新报告路径`会加载模型、计算两个真实样本并逐项比较；GPU执行包装器只接受与执行代码哈希一致、状态PASS的验证报告。本次检查已经完成，不需要在正在运行全量任务时重复启动。
