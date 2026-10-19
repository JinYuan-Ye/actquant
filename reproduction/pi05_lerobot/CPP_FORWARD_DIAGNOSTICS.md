# C++ 未量化前向：主要差异已定位，严格对照仍未通过

2026-10-19。本页记录跨后端诊断：`reproduction/results/pi05-cpp-forward-08` 状态 **NUMERICAL_MISMATCH**。构建与CUDA执行均成功，但原严格判据 atol=1e-6、rtol=1e-5 尚未全部通过，门槛保持不变。后续按用户要求已完成126个LLM矩阵量化及同后端配对十步采样，见 [QUANTIZED_MODEL_DEMO.md](QUANTIZED_MODEL_DEMO.md)。配对执行通过不改变本页严格比较结论；原服务器及LIBERO闭环尚未验证。

## 对照对象必须相同

此前 LeRobot/OpenPI EXACT_MATCH 是同一BF16/FP32计算路线之间的对照。官方GGUF导出器将大矩阵转成FP16，C++计算路径又与PyTorch不同，不能直接把原BF16数值当成所有C++诊断的唯一参考。

prepare_cpp_reference.py先复查原BF16模型loss与保存的真实参考逐元素EXACT_MATCH，再将参数按官方导出规则舍入到FP16值、用FP32计算构造诊断参考。norm/bias等一维参数保留原FP32值。它保留源模型实际的RoPE buffer数值，这一点是本轮发现的关键。

这里始终固定原样本的noise、time=0.5和noisy actions，比较带符号的预测速度 [1,50,32]。不是随机ODE采样输出，更不是闭环成功率。该FP32诊断参考与原BF16速度的最大差本身约0.053，不能把这一精度变化混称为低位量化误差。

## 输入层发现的问题

- 原 serve_policy.py 读取图像和prompt，没有读取机器人state并形成checkpoint要求的状态文本；直接使用该服务器无法证明沿用当前LeRobot输入约定。
- 原 Pi05::tokenize_prompt 会清理换行再追加换行，与当前包含 `State: ...;\nAction: ` 的完整prompt约定不同。诊断直接对保存的完整prompt调用原SentencePiece encode，151个有效token与官方参考完全一致。
- C++缓存路径让所有prefix token相互注意，只在suffix attention中屏蔽占位图像，并用物理长度编号。PyTorch则在prefix注意力中也使用pad mask，并对有效mask累加生成位置。这会让占位图像影响有效prefix，且改变后续位置。

外部诊断仅保留两路真实图像和151个有效文本token，去掉第三路masked图像与右侧padding，因此prefix从968个物理位置压缩为663个有效位置。有效token顺序和模型输入值保持一致，action位置为663～712。PyTorch内对比完整mask与压缩输入，最大速度差2.6226e-6，在原绝对/相对联合判据内通过。该验证只覆盖当前单样本。

诊断直接传入已验证的224×224、[-1,1]浮点像素，绕过C++原图缩放；这使运行时计算和图像缩放差异分离。原服务端完整预处理及ODE采样尚未对齐。

## 为什么提高矩阵乘法精度还不够

GGML默认FP16权重路径可将输入/输出也转成FP16，并选择半精度计算。另一个精度设置是cuBLAS句柄默认启用TF32。

外部cpp_forward_probe.cpp通过符号拦截调用同一GGML函数，并可给乘法节点设置公开的GGML_PREC_F32；配合NVIDIA_TF32_OVERRIDE=0隔离精度影响。没有修改官方源码、原二进制、量化内核或原GGUF权重。

这些开关显著降低视觉误差，但固定同一prefix的动作速度仍有约0.01842最大差，证明它们不是唯一来源。另做了数值不变的F32存储副本对照，专家输出与上述结果一致，也未解决该残差。

## 关键差异：RoPE 非持久 buffer 的 BF16 舍入

Gemma构造器会先生成FP32的inv_freq，随后父模块 `.to(bfloat16)` 会转换这个已注册buffer。选择性恢复FP32的逻辑只遍历参数，没有恢复buffer；之后调用model.float()也只能把舍入后的值转回FP32。

简化关系为：

```text
理想频率 = 1 / 10000^(2i/256)
源模型实际频率 = float32(bfloat16(理想频率))
旋转相位 = position × 实际频率
```

它是非持久buffer，不存入safetensors权重文件。GGUF中的参数键/值全部正确，也不能保证C++重建的频率与源模型一致。

Meta构造验证了PaliGemma与expert的inv_freq均为BF16；按固定构造公式检查，频率舍入差最大约0.00138819，在位置663相位差可达约0.92037弧度。记录见 pi05-cpp-reference-01/rope-buffer-audit.json。新参考入口会直接保存实际buffer；本轮初始参考的频率文件按已确认的CPU初始化公式补充重建。

诊断通过GGML公开RoPE frequency factors输入匹配源模型的实际频率，没有替换Transformer结构。因子按图输入名称写入，每次联合前向实际写入36个频率输入，日志可核验。因子比值的FP32运算以及两套后端的算子实现仍可能带来剩余误差，当前没有把它全部归结为正常浮点噪声。

## 实际结果

| 诊断 | 固定同一prefix的速度最大差 | 使用C++视觉/文本prefix的最大差 |
|---|---:|---:|
| 01：默认计算精度 | 0.02062845 | 0.02229452 |
| 02：请求F32 matmul | 0.01895764 | 0.01883984 |
| 03：再禁用TF32 | 0.01842037 | 0.01840669 |
| 07/08：再匹配源RoPE频率 | 0.0000399351 | 0.0000555515 |

最新08的固定prefix比较：mean abs diff约9.7699e-7，RMSE约3.0598e-6，50/1600个元素超限；这50个均在真实7维动作范围内。整条C++prefix路径有82/1600超限。文本嵌入完全一致，raw prompt的token IDs完全一致；同一C++prefix的cached与joint速度逐元素完全一致。

视觉中间表示仍未过严格判据。03/08的两路视觉RMSE约7.1942e-6、2.1916e-5，最大差约0.00061035、0.00123405。原C++时间步正弦嵌入用float运算，参考用float64计算后转float32，其最大差约8.5811e-5。为隔离速度误差，本轮专家前向传入的是同一份参考时间嵌入；时间步生成差异未混入上述速度比较。

## 可重放入口

以下是本轮已执行的诊断入口说明；结果目录已有文件，无需重跑。

```bash
source deployment/activate_server.sh
```

激活隔离环境和本地库路径，默认离线。

```bash
python reproduction/pi05_lerobot/prepare_cpp_reference.py \
  --output reproduction/results/pi05-cpp-reference-01
```

生成单样本固定输入及对应导出精度的PyTorch参考，包括像素、151个token、663个prefix、noise/time、速度和RoPE buffer。原BF16保存参考必须EXACT_MATCH。该目录拒绝覆盖；准备阶段约35.57秒，峰值GPU分配内存约16.44GiB。

```bash
python reproduction/pi05_lerobot/run_cpp_probe.py \
  --reference reproduction/results/pi05-cpp-reference-01 \
  --output reproduction/results/pi05-cpp-forward-08 \
  --mode matched_rope
```

新建输出目录，复用现有8个官方core对象编译独立探针，调用CUDA0后端，再比较阶段输出。matched_rope启用F32 matmul请求、禁用TF32并匹配源频率；upstream/f32_request/no_tf32用于隔离各项因素。所有模式均使用诊断提供的压缩prefix，不是原服务器端到端流程。

run.json分别记录build_returncode、execution_returncode和comparison_returncode。最新运行构建0、执行0、数值比较2，因此总体为NUMERICAL_MISMATCH，进程退出码2。非有限输出会直接判为执行失败。最新整条诊断入口约7.08秒。

## 文件与失败尝试

- pi05-cpp-reference-01：固定参考与buffer说明。
- pi05-cpp-forward-01/02/03：计算精度逐项诊断，comparison.json保留。
- pi05-cpp-f32-weights-01：约13.43GB的F32诊断副本，来自官方F16导出的精确升精度值，原文件未覆盖。
- pi05-cpp-forward-04：尝试图内临时weight cast时触发GGML调度器断言；失败记录保留，该路径已从最新源文件移除。
- pi05-cpp-forward-05：独立F32存储副本诊断，专家残差与03一致。
- pi05-cpp-forward-06：首版频率输入按tensor指针定位失效，速度输出非有限，属于失败诊断。旧execution.json只反映进程完成，不代表数值通过；最新入口已加入有限性保护。
- pi05-cpp-forward-07：按图输入名称正确写入频率后得到有限输出，显著改善。
- pi05-cpp-forward-08：统一可重放入口，最终严格状态NUMERICAL_MISMATCH，证据为outputs/comparison.json。

官方ActQuant/OpenPI源码未改，诊断均已结束，GPU内存已释放。

## 下一步仍需完成

继续定位视觉算子、时间嵌入和RoPE因子/归一化/规约顺序导致的剩余差异；不要自动放宽门槛或把当前指标改写成完整基线PASS。还需将状态prompt、浮点图像输入、有效token压缩及频率约定接入实际推理接口，再验证固定初始噪声的多步ODE输出。完整60轨迹、整模型低位量化和LIBERO闭环仍未完成。
