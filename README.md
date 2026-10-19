# ActQuant π₀.₅ 全量校准复现

量化所需代码和说明已提供，全量Fisher计算进行中，模型尚未完成。使用 README 指定的 LeRobot checkpoint，按原论文仓库10/10/10/30轨迹配置，使用60条轨迹的全部11,777帧重新计算HSIC、位宽分配和Fisher，量化126个LLM目标。视觉、embedding与action expert保持基础精度。

完整模型完成并通过验证后提供于：[GitHub Release](https://github.com/JinYuan-Ye/actquant/releases/tag/pi05-full-60ep-20261019-v1)。附件按不超过1GiB分片，下载Release中的全部附件后，在本地执行：

```bash
python deployment/restore_release_model.py --directory /path/to/downloaded-assets
```

此命令按MODEL_SHA256.json逐项检查分片，拼接成pi05.gguf并核对完整SHA256；已有模型不会覆盖。参数应替换为下载附件的目录，Windows也可使用例如D:/actquant/model。

## 代码与说明

- 原论文仓库完整源码：[src/ActQuant](src/ActQuant)，上游跟踪文件保持原样。
- 新增适配与方法入口：[reproduction/pi05_lerobot](reproduction/pi05_lerobot)。
- 数据格式及算法：[方法导读](reproduction/pi05_lerobot/METHOD_WALKTHROUGH.md)、[HSIC](reproduction/pi05_lerobot/HSIC_DEMO.md)、[位宽分配](reproduction/pi05_lerobot/ALLOCATION_DEMO.md)、[Fisher](reproduction/pi05_lerobot/FISHER_DEMO.md)。
- 主要问题与解决：[问题汇总](deployment/TEACHING_RELEASE_ISSUES.md)、[兼容性核查](deployment/PI05_CHECKPOINT_COMPATIBILITY.md)。问题汇总保留了教学阶段历史；正式规模数据以本次报告为准。
- 正式运行：[FULL_CALIBRATION_RUN.md](reproduction/pi05_lerobot/FULL_CALIBRATION_RUN.md)，原始审计和结果在[reproduction/results/pi05-full-01](reproduction/results/pi05-full-01)。
- 依赖版本与构建：[deployment](deployment)、[UPSTREAM_SOURCES.json](UPSTREAM_SOURCES.json)。

克隆本仓库后初始化两个固定版本依赖：

```bash
git submodule update --init src/openpi src/lerobot-v044
```

这条命令获取已固定提交的OpenPI和LeRobot源码。后续构建按deployment文档执行；ActQuant源码已随仓库提供，安装脚本会核验其快照。模型原始权重和校准数据需要按文档单独准备，未提交到Git历史。

## 验证范围

跨后端严格数值差异仍保留NUMERICAL_MISMATCH和原容差。量化前后比较采用相同C++探针与输入/初始噪声。原高层服务器尚未完成输入适配，未进行LIBERO闭环任务质量验收。本仓库不是原论文官方仓库，也不宣称已复现论文成功率。四帧教学版文档明确标为历史教学实验，与本轮全量结果区分。
