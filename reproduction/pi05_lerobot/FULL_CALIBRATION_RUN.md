# 60轨迹全帧校准与量化运行记录

2026-10-19，用户明确要求“现在继续全量校准集量化”，授权本轮所需的完整数据下载和长时间统计计算。四帧教学版及Windows交付归档保持原样，正式规模任务使用新目录。

## 全量的具体含义

沿用固定ActQuant提交的数据采集配置：spatial/object/goal/long分别10/10/10/30条轨迹，共60条；前3组各10个任务每任务1条，long每任务3条。随机种子42，使用所选轨迹的每一帧，不设帧数截断。不是把LIBERO训练集全部轨迹用于校准。

原始数据固定为 `openvla/modified_libero_rlds` 的 `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`，四套共96个TFRecord分片和8个元数据文件，10,230,689,658字节。下载复用已有正确文件，逐个核对LFS SHA256或小文件Git blob摘要，拒绝覆盖内容不同的已有文件。

本轮104个源文件已全部下载并核验，导出数据审计通过：共11,777帧，spatial 1,216帧、object 1,388帧、goal 1,208帧、long 7,965帧。每帧的50步动作窗口和重复末动作补尾已逐项验证。后续HSIC与Fisher使用全部11,777个索引，不取子集。

## 输出和进度

- 源数据：`data/libero_probe_rlds`，目录名沿用旧探针位置，内容已扩充为四套完整源数据。
- 原生分辨率校准数据：`reproduction/data/pi05-full-60ep-01`。
- 全阶段状态：`reproduction/results/pi05-full-01/pipeline.json`。
- 下载摘要与进度：同目录 `download.json`。
- 样本与动作窗口审计：`data-audit.json`。
- HSIC：`hsic/run.json`、`hsic/scores.json`、`hsic/indices.npy`。
- 位宽表：`allocation.json`。
- Fisher：`fisher/run.json`、`fisher/fisher_llm.gguf`、`fisher/fisher_vision.gguf`。
- Fisher导出审计：`fisher-audit.json`。
- 成品（完成后才存在）：`quantized/model/pi05.gguf`。
- 同后端动作误差（完成后才存在）：`paired-validation.json`。

只有 `pipeline.json` 的status为PASS才表示整个量化和文件/配对执行检查已完成。RUNNING表示正在运行，FAIL应查看phase对应日志。即便PASS，也不表示LIBERO成功率已验收。

## 保持不变的方法

LeRobot checkpoint、官方输入约定、horizon50、B=1、完整32维均值loss、FP32 matmul highest、关闭额外图像增强与教学版一致。HSIC复用上游全帧GPU实现（hidden RBF、Y linear、lx=ly=1、默认带宽估计），126目标使用F_out分配；预算2.3471，候选IQ2_XS至Q4_K。Fisher保留原逐样本梯度平方统计和GGUF writer，外部GPU执行器仅改变累积器存放位置，已在真实模型全部288项验证逐元素一致，见 [FISHER_PERFORMANCE.md](FISHER_PERFORMANCE.md)。不更改数学公式或量化内核。所有126项显式传入分配格式，并验证per-weight Fisher被读取。

本轮仍量化126个LLM目标；视觉、embedding、action expert等保持基础精度。计算并保存视觉Fisher不等于本轮对视觉全量低位量化。

## 为全量运行增加的工程支持

`export_native_calibration.py`复用原分层采样和动作补尾，保留256图像交给已验证模型预处理。可选稀疏存储只把全零第三相机保留为文件空洞，shape/dtype/数值不变。记录所选轨迹全局索引、任务与上游脚本哈希。

`audit_full_calibration.py`检查60轨迹/40个任务分组、全部帧、两路真实相机、零占位相机、有限状态动作，并逐个核对每帧50步动作窗口及轨迹末尾重复策略。Fisher导出阶段核查288目标的checkpoint shape映射、FP32正有限值、counts和全帧元数据。

全帧HSIC激活仅保存在内存，禁用可能占满磁盘的缓存；评分前释放模型显存，仍调用同一个官方评分函数。全量运行前已用真实四样本288项Fisher验证新审计器。

Fisher每200帧保存累加器、已完成loss、样本索引、NumPy/Torch/CUDA随机状态和代码哈希，可精确恢复。原子替换只更新本次任务的恢复快照，成功导出后删除本次已过期快照，不删除旧实验。模拟中断回放测试已确认恢复结果与连续执行逐字节相同。HSIC目前没有中途恢复；若该阶段失败，需要检查原因并新建其输出目录后重跑，不能假称已完成。

## 已执行的启动入口

```bash
source deployment/activate_server.sh
python -u reproduction/pi05_lerobot/run_full_pipeline.py \
  --run-dir reproduction/results/pi05-full-01 \
  --calib-dir reproduction/data/pi05-full-60ep-01 \
  --fisher-execution gpu
```

激活命令选择隔离的OpenPI环境、离线模型加载和GGUF/C++库路径。流水线命令指定正式规模结果和数据目录，随后自动导出、审计、HSIC、分配、Fisher、量化、合并与单观测十步配对采样。最初CPU路径用--wait-for-download启动；下载完成后，该参数不再需要。现用--fisher-execution gpu从原200帧快照精确续算，跳过已完成阶段。Fisher传入的数据总帧数来自审计，不是4或其他上限。该入口已经启动，不应并行再启动第二份；文件锁也会拒绝同一任务并发。

## 仍需独立验证

跨后端严格NUMERICAL_MISMATCH记录和原容差保留。量化前后使用同一C++探针、相同输入/噪声/RoPE约定比较。原策略服务器输入接入和LIBERO闭环不属于这一流水线的已通过范围，不能以全量校准或单观测有限输出替代任务质量评估。
