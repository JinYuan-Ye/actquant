# 全量60轨迹量化结果

完整校准：60条轨迹，11,777帧，10/10/10/30轨迹来自spatial/object/goal/long。HSIC和Fisher均覆盖所有帧，重新分配位宽并实际量化全部126个LLM目标。使用固定LeRobot checkpoint及已验证输入适配，不复用四帧教学分数或Fisher。

| 项目 | 实际结果 |
|---|---:|
| 完整基础GGUF字节数 | 6,714,075,456 |
| 完整量化GGUF字节数 | 3,331,889,504 |
| 文件缩小比例 | 50.3746% |
| 126目标参数加权block位宽 | 2.347073413 |
| 逐权重Fisher日志确认 | 126 |
| 统一模型tensor总数 | 814 |
| 保持基础权重字节不变的tensor | 688 |
| 保留元数据字段 | 37 |

实际格式分布：`{"IQ2_S": 18, "IQ2_XS": 65, "IQ3_S": 2, "IQ3_XXS": 5, "IQ4_XS": 1, "Q2_K": 6, "Q4_K": 29}`。2.347附近的平均位宽仅指126个目标，不是整份统一模型平均位宽。视觉、embedding、action expert、projector和norm保持基础精度。

## 验证证据

完整GGUF格式/来源/元数据审计通过；量化前后使用相同C++实现、输入、源RoPE约定、初始噪声进行十步动作采样，输出均有限。单观测反归一化7维动作RMSE为0.030076933，最大绝对差0.133620282。这些是控制器输入动作的误差指标，不是米，也不是任务成功率。

跨后端严格NUMERICAL_MISMATCH与原容差保留。原高层服务接口及LIBERO闭环尚未验证，任务质量NOT_EVALUATED。相同校准规模不等于已复现论文成功率。

机器可读证据位于`reproduction/results/pi05-full-01/`：data-audit.json、hsic/run.json、allocation.json、fisher/run.json、fisher-audit.json、quantized/artifact-audit.json、paired-validation.json和pipeline.json。

## 下载完整模型

[Release](https://github.com/JinYuan-Ye/actquant/releases/tag/pi05-full-60ep-20261019-v1)包含全部模型分片、tokenizer、归一化统计、RoPE快照、运行约定及摘要清单。下载全部附件后执行：

```bash
python restore_release_model.py --directory /path/to/downloaded-assets
```

这条命令按照MODEL_SHA256.json核验每个分片，按原顺序还原pi05.gguf并验证完整摘要；不覆盖已有模型。Windows可将参数改成D:/actquant/model等实际下载目录。

完整模型SHA256：`35b2b430c9a9e94e3b105e65b155844b7c2d014096abc933a9c63680de99ca40`。

Fisher低利用率问题及数值一致性验证见[FISHER_PERFORMANCE.md](FISHER_PERFORMANCE.md)，数据与运行入口见[FULL_CALIBRATION_RUN.md](FULL_CALIBRATION_RUN.md)。
