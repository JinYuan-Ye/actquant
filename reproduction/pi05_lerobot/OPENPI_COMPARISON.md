# LeRobot → OpenPI 单样本前向对照

目的：用同一份权重、已保存的模型输入、噪声和时间步，检查 OpenPI 与已通过的 LeRobot 参考之间的逐元素损失差异。脚本位于官方仓库外，没有修改 Fisher、HSIC 或量化算法，也没有接入正式校准流程。

## 当前结果：最高精度对照已完全一致

用户已完成pi05-openpi-forward-02，comparison.status=EXACT_MATCH，全部1600个loss差为0。进一步原始输入适配及抽查梯度对照也通过，见 METHOD_WALKTHROUGH.md。以下保留首轮差异诊断和命令作历史记录，不需要重跑。

## 历史诊断：首轮MISMATCH与FP32运算设置

用户已完成 pi05-openpi-forward-01，实际严格加载、输入一致和forward均PASS；1590/1600元素超限，最大绝对差0.02026653289794922、平均绝对差0.0008019326020129525。旧结果保留。

源码及meta构造检查发现：OpenPI构造器无条件设置torch.set_float32_matmul_precision("high")；LeRobot在compile_model=false时保持默认highest。本地新进程记录对应cuda_matmul_allow_tf32分别为true/false，cuDNN的TF32开关均为true。两边参数都包含FP32层，所以只对齐BF16权重存储精度不够。旧报告没有记录此开关，这一解释仍需单变量GPU实验确认；证据见 reproduction/evidence/pi05-matmul-precision-audit-01.json。

当前在openpi-server环境执行：

```bash
python reproduction/pi05_lerobot/openpi_compare.py \
  --mode forward \
  --reference reproduction/results/pi05-real-forward-01 \
  --float32-matmul-precision highest \
  --output reproduction/results/pi05-openpi-forward-02
```

用途：使用同一权重、输入、噪声、时间和误差门槛，只把FP32矩阵乘法运行策略显式改成highest。参数必须在模型构造之后生效，脚本已处理；highest不会把整个BF16模型转换成FP32，也不改变Fisher算法。输出新目录-02保留旧失败结果。报告应记录runtime_precision_after_constructor的high/allow_tf32=true，以及runtime_precision_before_forward的highest/allow_tf32=false；comparison.status仍需实际运行判断。保留原atol1e-6/rtol1e-5，不自动放宽。

runtime_precision.py记录数值开关；7项CPU检查已通过。下面保留首轮操作和通用说明供复查，不需要重跑旧-01目录。

## 已完成的准备检查

- LeRobot 参考：reproduction/results/pi05-real-forward-01，loss_7d=0.025776993483304977，loss_32d=0.0767202377319336。
- 独立入口：reproduction/pi05_lerobot/openpi_compare.py。
- CPU检查：test_openpi_compare.py 六项通过，包括配置字段映射、严格前缀处理、真实共享参数别名、逐元素差异检测、输入不变/不消耗随机数、拒绝错误相机mask。
- 结构检查结果：reproduction/results/pi05-openpi-inspect-01/report.json。只读safetensors文件头，使用meta模型记录结构，不分配权重存储、不运行完整模型。812个文件张量对应813个目标状态键，唯一缺少的是与lm_head共享同一Parameter的embed_tokens别名；补齐后键名、形状、精度全部匹配。目标状态键中122个FP32、691个BF16（包含共享别名）。schema_match=PASS不等于完整权重严格加载已经通过。

## 适配范围

1. 将chunk_size映射到action_horizon=50、max_action_dim映射到action_dim=32，token长度200，禁用compile。按官方构造器保留BF16/FP32混合精度，不将整个模型强制cast成BF16。
2. 当前固定checkpoint仅去掉最外层model.前缀。没有泛化的模糊重命名或丢弃权重；缺失键只允许由模型实际共享同一Parameter的已存在别名补齐。完整用户运行仍执行strict=True。
3. 从参考NPZ直接读取官方已处理图像、mask、token、归一化动作、状态、noise和time。不会重新分词、归一化、缩放、采样噪声或采样时间步。此阶段隔离模型实现差异，尚未验证原始RLDS→OpenPI的在线输入适配。
4. **发现的新差异：OpenPI PI0Pytorch.forward硬编码调用_preprocess_observation(train=True)，即使eval()也会随机裁剪、旋转、改变颜色。** LeRobot已保存的参考未经过这条增强路径。独立SavedInputPI05子类仅重写该预处理调用，显式使用train=False，并逐张量检查输入保持相等。forward本体继续使用上游实现。这个关闭增强的选择仅用于同输入对照，没有偷偷改写正式Fisher/HSIC的实验策略；后续正式校准是否保留增强要单独说明并记录。

## 用户执行

从 /workspace/actquant 运行。前一条成功后执行后一条。

```bash
source deployment/activate_server.sh
```

用途：从lerobot-reference切换到已有的openpi-server隔离环境，让目标模型使用OpenPI对应的Transformers 4.53.2及官方替换文件。`source`在当前shell内切换Python和环境变量，不安装或升级任何依赖。正常情况下终端环境名变成openpi-server。两边依赖不同，不在同一解释器混合导入LeRobot和OpenPI；参考数据通过NPZ传递。

```bash
python reproduction/pi05_lerobot/openpi_compare.py \
  --mode forward \
  --reference reproduction/results/pi05-real-forward-01 \
  --output reproduction/results/pi05-openpi-forward-01
```

用途：读取现有参考文件，重做输入与结构检查，计算权重摘要、严格加载同一checkpoint，在GPU上执行一次OpenPI前向，然后比较全部1600个逐元素损失（1×50×32）。不做反向传播、量化、动作采样或机器人闭环。

- `--mode forward`：显式授权本次完整权重加载和前向；默认CUDA。
- `--reference`：已成功的LeRobot结果目录，包含report.json和inputs_and_reference.npz，不能传原始数据目录。
- `--output`：新结果目录；已有路径会拒绝覆盖，重跑需改为-02等编号。
- 噪声和time=0.5直接从参考读取，seed仅保留运行记录及初始化可复查性，不重新生成参考噪声。

首轮GPU前向已完成但数值对照MISMATCH；新的highest对照尚未执行。命令会先计算权重摘要及完成CPU构造/加载，再进入GPU。

## 如何判断结果

report.json应包含schema_match、reference_input_validation、strict_load、input_identity、forward均为PASS，loss_shape=[1,50,32]。这些状态分别表示结构匹配、参考包有效、实际权重严格加载、送入前向的输入不变、返回的损失形状/数值有效。

comparison.status单独报告数值一致性：

- EXACT_MATCH：所有逐元素损失数值相等。
- WITHIN_TOLERANCE：有差异，但所有元素满足abs(target-reference) ≤ atol + rtol×abs(reference)。默认atol=1e-6、rtol=1e-5；这是预先声明的诊断门槛，不是已证明适用于所有BF16计算的科学验收界限。
- MISMATCH：至少一个元素超出门槛。脚本保存结果后以非零状态退出；不能因forward=PASS就忽略此状态，也不要直接放宽阈值。下一步查看最大/平均差异、超限数量与对应索引，再定位实现/精度差异。

comparison.npz保存目标损失、参考损失及逐元素差值；报告同时保存原参考文件摘要、权重摘要、源码摘要、版本、显存、7维/32维损失。报错时已创建的输出目录也保留，重试需新编号。

即使完全匹配，也仅支持这个样本、噪声、时间步下的逐元素损失一致。当前参考只保存了平方误差，不能据此单独证明带符号的预测速度、全部模型行为或梯度等价。正式Fisher的梯度、原始数据接入、60轨迹采样、GGUF/C++和闭环评估仍待后续验证。
