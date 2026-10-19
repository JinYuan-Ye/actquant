# 第一份真实 LIBERO 样本

2026-10-19 最新：用户已完成native提取 pi05-real-probe-02 和真实前向 pi05-real-forward-01。strict_load/forward均PASS，loss_7d=0.025776993483304977、loss_32d=0.0767202377319336，峰值分配显存7.117170333862305 GiB。以下命令保留为已执行流程，不需要重跑。下一步进行ActQuant/OpenPI独立适配与数值对照。

## 目的与来源

先用一份真实观测及连续50步动作检查输入路径，再决定正式校准数据的准备方式。本次样本不是论文的60条轨迹校准集，不应送入正式 Fisher/HSIC 实验。

- 仓库：https://huggingface.co/datasets/openvla/modified_libero_rlds
- 固定 revision：6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551。
- 文件：libero_spatial_no_noops/1.0.0/libero_spatial-train.tfrecord-00000-of-00016。
- 大小：118202941 字节，约112.73 MiB；另下载两个很小的 JSON 描述文件。
- SHA256：6d6360559a4b71df3bcabd9b55a75451f04375449ca5267616cbc1c21528eac5。
- 本地目录：data/libero_probe_rlds。这里只含一个分片，不可当作完整 spatial 数据集。

prepare_real_probe.py 位于官方仓库外，复用固定本地 ActQuant 导出脚本的解析器与图像缩放函数。它只提取一个样本，要求两路真实图像及完整50步动作，拒绝缺失字段和非有限值；没有修改官方采样或量化算法。正式校准导出脚本默认读取四个任务集，不能把此部分下载直接交给它冒充完整数据。

## 用户执行：每步成功后再继续

在 /workspace/actquant、已激活的 lerobot-reference 环境中运行。

```bash
python reproduction/pi05_lerobot/prepare_real_probe.py --mode download
```

用途：下载固定版本的一个 RLDS 分片及其描述文件。RLDS 分片是包含多条演示轨迹的数据文件，模型不直接读取它。`--mode download` 仅下载和校验，不提取样本、不加载模型；下载进程单独开启网络。成功时三份文件均显示 Verified；若已有内容不同则拒绝覆盖。大小约113 MiB，不下载整个约10.23 GB仓库。

```bash
python reproduction/pi05_lerobot/prepare_real_probe.py \
  --mode extract --episode 0 --frame 0 \
  --output reproduction/data/pi05-real-probe-01
```

用途：从已下载分片的第0条轨迹、第0帧提取观测，并取该帧起连续50步真实动作。`--episode` 是本分片内的轨迹索引，不是整个数据集的全局编号；`--frame` 是轨迹内帧索引；`--output` 必须是新目录。此步骤离线运行，不加载模型，不做归一化。长度不足会报错，不重复尾部动作来凑数。

预期 pixel_values_shape=[1,3,224,224,3]、state_shape=[1,8]、gt_actions_shape=[1,50,7]，action_tail_padding=false。第三路相机是占位零图，参考适配器只取前两路，再交给官方模型构造带 mask 的占位相机。产物包括四个输入文件、metadata.json 和 cameras.png（左侧主相机、右侧腕部相机）。

```bash
python reproduction/pi05_lerobot/reference.py \
  --mode preprocess \
  --calib-dir reproduction/data/pi05-real-probe-01 --index 0 \
  --output reproduction/results/pi05-real-preprocess-01
```

用途：将真实样本交给 checkpoint 官方预处理器，核对状态/动作归一化、状态文本与 token。`--calib-dir` 替代之前的 `--synthetic`，`--index 0` 选择提取目录中的唯一一个样本；`--mode preprocess` 不加载完整模型。预期 normalization、prompt_format、tokenizer_exact_match 均为 PASS，sample_kind=raw_calibration，action_shape=[1,50,7]，tokens_shape=[1,200]。weights_present=true 只表示权重文件在磁盘上，本命令没有加载权重。

保留 metadata.json、cameras.png 和预处理 report.json 供下一步检查。已有输出目录不可重用，重新执行时改编号，并相应更改后续输入路径。真实数据提取和预处理尚未由用户实测；目前只有四个小型 CPU 检查通过，覆盖相机/时间对齐、拒绝不足50步、缺失相机/非有限动作、拒绝覆盖。

## 已核实与待核实的语义

1. 远端 features.json 标注：两路256×256 RGB JPEG；state 是8维末端状态（6维位姿、2维夹爪）；action 是7维末端动作。
2. OpenVLA 再生成源码说明，HDF5 转 RLDS 时图像已旋转180度。本脚本保留 RLDS 方向，只复用 ActQuant 的224×224 PIL双线性缩放，不额外旋转。来源：https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py 。仍需检查实际预览，并与 checkpoint 训练输入方向、缩放约定比较。
3. 本地 LeRobot LiberoEnv 将 agentview_image 映射到 image、robot0_eye_in_hand_image 映射到 image2。RLDS 主相机/腕部相机按此顺序输入。
4. 状态和动作原值原样保留；MEAN_STD 使用 models/pi05_libero 的 processor safetensors，不能拿这个单样本重估统计量。动作的单位、夹爪符号、状态旋转表示以及与 checkpoint 训练数据的一致性仍需追溯核实，不能因维度吻合就宣布语义对齐。
5. 真实样本预处理和图像检查通过后，再由用户执行真实样本固定噪声前向。参考路径成功仍不代表 ActQuant/OpenPI/C++ 的兼容性已完成。

## 当前执行：保留原始图像的真实参考前向

已读回旧样本：第0条轨迹共有110帧，选择第0帧及连续50个动作，没有尾部补齐。双相机预览可正常解码，主相机和腕部相机顺序与源码映射一致；外观本身不能证明训练像素完全一致。状态排列与 LeRobot LiberoProcessorStep 对应为位置3维、轴角3维、夹爪关节位置2维。本地 LIBERO 使用 OSC_POSE：动作前6维是控制器缩放前的相对控制输入，不能直接当米/弧度值；默认位置缩放上限0.05，旋转0.5，Panda夹爪 -1 打开、+1 关闭。此次保留数据原值，不进行控制器单位换算或符号变换。与 checkpoint 实际训练数据版本的逐样本等价尚未验证。

此前预处理 PASS 核对了归一化、prompt、token；没有核对图像缩放。官方模型 _preprocess_images 使用 PyTorch resize_with_pad_torch；旧提取路径则先用 PIL 缩至224。CPU实际对照记录于 reproduction/evidence/pi05-real-input-audit-01.json：在模型[-1,1]像素范围，主相机平均绝对差约0.00337、最大0.14554，腕部平均约0.00370、最大0.16775。这是输入差异，不是策略效果评估。已新增显式 --image-mode native，旧默认 actquant 保持可复查，官方量化源码未改动。

```bash
python reproduction/pi05_lerobot/prepare_real_probe.py \
  --mode extract --episode 0 --frame 0 --image-mode native \
  --output reproduction/data/pi05-real-probe-02
```

用途：离线从同一已校验分片重新提取相同样本。`--image-mode native` 保留解码后的256×256 RGB图像，让官方模型承担缩放；episode/frame 指定相同轨迹和起点，状态、动作、任务保持一致。新目录 -02 保留旧结果。预期 pixel_values_shape=[1,3,256,256,3]，state_shape=[1,8]，gt_actions_shape=[1,50,7]，action_tail_padding=false。此命令不加载模型、不重新下载。

```bash
python reproduction/pi05_lerobot/reference.py \
  --mode forward \
  --calib-dir reproduction/data/pi05-real-probe-02 --index 0 \
  --output reproduction/results/pi05-real-forward-01
```

用途：对这个真实样本先执行相同的预处理检查，再严格加载模型，运行一次固定噪声前向。`--mode forward` 已包括 preprocess，所以不需要再单独执行一轮预处理；`--calib-dir` 指向新的 native 输入，`--index 0` 取唯一的样本，结果写入新目录。默认使用CUDA、seed=42、time=0.5和BF16，不执行反向传播、量化或完整动作采样。

预期 strict_load/forward/normalization/prompt_format/tokenizer_exact_match 均为PASS；raw_image_shapes 为两路[1,3,256,256]，model_image_shapes 为三路[1,3,224,224]（含mask掉的占位相机），loss_shape=[1,50,32]，两个loss均有限。相同状态/任务的 valid_tokens 应仍为151。真实loss尚无实测，不预设阈值，也不要求比合成loss小。

目前5项CPU检查通过，包括native像素保持和动作/状态不变；native样本提取与真实前向已由用户执行通过。已完成的旧预处理保留为证据，其prompt里的-1/255源于归一化状态约-1.509/1.083；MEAN_STD不保证落在[-1,1]，不能擅自裁剪。

## 真实前向读回验证

助手只用CPU读取保存的NPZ确认：loss均有限且与report的两种平均值吻合，模型图像mask为[true,true,false]，动作补齐25维为零，noise=[1,50,32]，time=[0.5]。补齐25维的平均loss约0.09098435。7维平均与32维平均的大小关系不固定；输入动作补零也不表示flow-matching逐元素损失必为零。

参考结果验证了本次LeRobot输入路径可运行，尚未证明OpenPI数值等价、完整动作生成质量或闭环成功率。
