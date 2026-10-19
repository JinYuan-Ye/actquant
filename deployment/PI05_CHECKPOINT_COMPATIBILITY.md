# π₀.₅ checkpoint 与校准脚本兼容性核查

本页保留兼容性修复与四帧教学阶段的历史证据。60轨迹、11,777帧全量版本的产物、误差指标和下载入口统一见 [FULL_QUANTIZATION_RESULTS.md](../reproduction/pi05_lerobot/FULL_QUANTIZATION_RESULTS.md)，其完成判据为全量pipeline状态PASS；不要把下文历史教学规模当作全量规模。

核查日期：2026-10-19。本文保留历史核查过程。最新已完成全部126个LLM目标矩阵的真实混合位宽量化、完整GGUF合并及同后端固定噪声十步采样，模型由6.71 GB缩至3.33 GB。详见 [量化产物与验证记录](../reproduction/pi05_lerobot/QUANTIZED_MODEL_DEMO.md)。跨后端严格NUMERICAL_MISMATCH保留，未放宽阈值；本次采用同后端配对比较测量量化影响。四帧教学校准不代表论文规模结果，LIBERO闭环尚未验证。

ActQuant 本地版本：b64791125070652fe6b554e244fe809c79ef5246。
OpenPI 本地版本：215abfb217dbac7d5f1273282331b9b1866c0479。
待核查 checkpoint：lerobot/pi05_libero_finetuned_v044，已下载配置版本 7152f13e34c6c115d808aea0e5d08aca74eac14b。

## 初始核查结论（后续进展见文末）

基础环境准备完成与 checkpoint 的端到端兼容性是两项不同的验收。当前发现输入约定不一致，LeRobot 官方模型现已通过权重 strict load 和合成前向；ActQuant/OpenPI/C++ 路径仍未验证。现有信息不足以宣称能原样复现论文结果。

| 项目 | checkpoint 配置 / 上游约定 | 当前 ActQuant 校准实现 |
|---|---|---|
| 动作时间长度 | chunk_size=50 | load_model 只读取 action_horizon；缺失时取 10 |
| 动作维度 | 真实动作 7，max_action_dim=32 | action_dim 缺失默认 32；当前数值匹配，但没有做字段映射 |
| 动作归一化 | ACTION=MEAN_STD，统计张量为 7 维 | 从校准文件读动作，只截取、补齐，没有应用该统计量 |
| 状态归一化 | STATE=MEAN_STD，统计张量为 8 维 | 仅查 JSON 的 q01/q99；不读取现有 processor safetensors，未找到时 clip |
| 状态进入模型 | 配置声明 pi05_prepare_state_tokenizer_processor_step；LeRobot v0.4.4 将状态离散后加入任务文本 | 仅编码任务文本加换行；OpenPI 的 pi05 分支不经过连续 state_proj |
| 文本 token | OpenPI 官方 tokenizer 显式 add_bos=True；无状态分支单独编码换行 | sp.encode(task + 换行)，未显式添加 BOS；精确 token 对齐尚待验证 |

状态归一化差异在当前 pi05 路径中不能简单描述为“状态值错了所以改变输出”：更直接的问题是状态并未通过上述两条路径进入模型。统计文件含 q01/q99 并不表示 checkpoint 配置选择了分位数归一化。

## 本地证据位置

- src/ActQuant/README.md:190：推荐 LeRobot v044 checkpoint。
- models/pi05_libero/config.json：chunk_size、normalization_mapping、tokenizer_max_length。
- models/pi05_libero/policy_preprocessor.json：normalizer 与状态文本处理步骤。
- src/ActQuant/tools/fisher-diag/compute_fisher_pi05.py:145：build_observation_and_actions。
- 同文件 :192：文本输入；:207：动作切片与补齐；:236：统计加载；:264：模型加载。
- src/ActQuant/tools/fisher-diag/get_pi05_calib_data.py:333：直接抽取 RLDS 动作序列，没有应用 checkpoint 的 MEAN_STD 统计。
- src/ActQuant/tools/hsic/compute_hsic_pi05.py:57：复用 Fisher 的模型加载和输入构造。因此状态与文本问题也影响 HSIC 输入。HSIC :195 的 Y 标准化是统计量计算的一步，不能替代 Fisher 模型目标的归一化。
- src/openpi/src/openpi/models_pytorch/pi0_pytorch.py:238：embed_suffix，state_proj 仅在 not self.pi05 时使用。
- src/openpi/src/openpi/models/tokenizer.py:22：官方文本编码约定。
- src/openpi/src/openpi/training/config.py:744：OpenPI pi05_libero 配置为 action_horizon=10、discrete_state_input=False。

## 外部证据与边界

LeRobot v0.4.4 官方 processor：
https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/pi05/processor_pi05.py

LeRobot checkpoint 模型卡要求使用对应 pre/post processors：
https://huggingface.co/lerobot/pi05_libero_finetuned_v044

第三方 issue 质疑发布 GGUF 的原始权重来源：
https://github.com/arashakb/ActQuant/issues/2

核查时该 issue 仍 Open，页面未见作者答复。其权重比较是第三方报告，本次没有下载权重独立复验，不能将其当作作者确认。OpenPI checkpoint 只能作为候选；更换来源不能自动解决动作尺度和 tokenizer 对齐，也不能自动成为论文的严格复现。

## 后续路径

1. 保留现有 checkpoint 小文件和原仓库，暂停大型下载及正式 HSIC/Fisher 计算。
2. 若要求原样复现论文，需作者确认原始 checkpoint revision、预处理约定和对应复现版本。可以准备问题，由用户决定是否向作者提交。
3. 若选择在 README 指定的 LeRobot checkpoint 上复现方法，应先建立其官方未量化参考，再在仓库外实现可审计的输入/字段适配；量化算法保持原样。这属于带兼容性适配的复现，不能声称是原脚本无修改复现。用户已选择该路径；独立参考环境及输入适配已准备，见 reproduction/pi05_lerobot/README.md。完整权重加载和合成前向已通过，真实校准样本验证待执行。
4. 适配验收先比较同一小批样本的图像、token、mask、动作尺度及形状，再以固定噪声和时间步比较未量化前向。仅有有限 loss 或成功运行不能证明语义对齐。
5. 用户亲自执行上述模型步骤。通过小样本验证后再进入 HSIC、Fisher、量化和 LIBERO 评估。

初始核查时本地模型目录只有 config.json、policy_preprocessor.json、policy_postprocessor.json 和 normalizer safetensors；没有 model.safetensors。

## 2026-10-19 后续进展

独立 envs/lerobot-reference 已安装并通过依赖与导入检查。CPU 适配器检查通过；该环境部署阶段尚未下载完整权重/tokenizer 或运行模型，后续结果见下文。另确认 LeRobot policy.forward 对真实7维动作求平均，而当前 ActQuant Fisher 对补齐32维求平均；参考脚本同时报告两种损失，未修改官方算法。新环境命令、版本和冲突处理记录于 reproduction/pi05_lerobot/README.md 与 deployment/lerobot-reference.lock。

## 2026-10-19 合成参考前向通过

用户已执行完整权重下载及固定噪声合成样本前向，报告：reproduction/results/pi05-forward-01/report.json。strict_load=PASS，forward=PASS，loss_shape=[1,50,32]；loss_7d=1.5784870386123657，loss_32d=0.41200968623161316。seed=42、time=0.5、bfloat16，PyTorch 峰值分配显存约7.117 GiB。真实 tokenizer 的逐元素对照也通过。

该结果确立了 LeRobot 官方模型的合成样本参考，不代表 ActQuant/OpenPI 权重映射、真实样本语义、GGUF/C++ 或量化闭环兼容。下一步先准备真实校准样本并核对相机、状态和动作语义。

## 2026-10-19 真实样本预处理与图像缩放核查

用户已通过 reproduction/results/pi05-real-preprocess-01，真实样本为110帧轨迹的第0帧和连续50步动作，无尾部填充；token有效长度151。两路图像可解码，状态/动作/任务源数据已读回。

发现额外输入差异：ActQuant导出器提前使用PIL bilinear把256图像缩到224，LeRobot官方模型则使用PyTorch resize_with_pad_torch。旧预处理PASS未覆盖图像缩放。CPU实测主/腕相机在[-1,1]像素范围的平均绝对差约0.00337/0.00370，最大约0.14554/0.16775；证据 reproduction/evidence/pi05-real-input-audit-01.json。独立提取器新增 --image-mode native 保留256原图；官方仓库未改动。

下一步用户提取 pi05-real-probe-02 并执行 pi05-real-forward-01。官方参考需要保留原图并交给官方模型缩放；正式ActQuant适配也必须明确该差异。状态排列和本地控制器约定已核对，但不能据此宣称与checkpoint实际训练数据逐样本完全一致。

## 2026-10-19 native真实样本参考前向通过

用户已执行 reproduction/results/pi05-real-forward-01，使用 pi05-real-probe-02 的原始256图像，由官方LeRobot模型缩放至224。strict_load/forward及归一化、token检查均PASS；loss_shape=[1,50,32]，loss_7d=0.025776993483304977，loss_32d=0.0767202377319336，BF16、seed42、time0.5，峰值分配显存约7.117GiB。助手已CPU读回NPZ确认两个loss平均值、补零动作和相机mask[true,true,false]。

下一步：在仓库外准备LeRobot→OpenPI权重/输入适配，使用同一保存的图像、mask、token、动作、noise、time作对照，再接入校准入口。ActQuant当前loader仍默认horizon10，输入构造仍与checkpoint约定不符，不能直接用真实样本PASS作为原Fisher可运行的证据。尚未执行OpenPI目标模型前向、正式Fisher/HSIC、量化或LIBERO闭环。

## 2026-10-19 OpenPI独立对照准备完成

新增 reproduction/pi05_lerobot/openpi_compare.py，六项CPU检查通过。reproduction/results/pi05-openpi-inspect-01/report.json 记录meta结构审计：checkpoint812个张量，目标813个状态键，仅补齐真正共享Parameter的词嵌入别名；其余键名、形状、FP32/BF16精度全匹配。这不是实际权重strict load通过；完整目标加载/前向由用户下一步执行。

另发现OpenPI forward显式调用train=True的预处理，eval不会关闭随机裁剪、旋转和颜色增强。为同输入对照，仓库外子类仅将预处理设为train=False，并逐张量断言输入不变；上游forward本体、Fisher/HSIC和量化源码未改。正式校准时的增强策略尚未决定，不能把这个测试设置悄悄当作原实验。

脚本直接读取LeRobot参考NPZ的图像、token、mask、动作、noise和time，对全部1600个平方误差逐元素比较并保存差值。默认atol1e-6/rtol1e-5是诊断门槛，实际数值一致性尚未实测。详细操作与边界见 reproduction/pi05_lerobot/OPENPI_COMPARISON.md。

## 2026-10-19 OpenPI首次对照MISMATCH与运行精度差异

pi05-openpi-forward-01已完成实际strict load及前向，input_identity=PASS，但1590/1600个逐元素loss超限。最大绝对差0.02026653，平均绝对差0.00080193。模型可加载不等于数值已对齐。

发现OpenPI构造器无条件将FP32矩阵乘法设为high（允许TF32），而LeRobot在本次compile=false配置下保持highest（不允许matmul TF32）；meta构造复现了设置差异。旧报告未记录此开关，不能断言这是全部误差来源。两个环境的Gemma/SigLIP/PaliGemma实现文件忽略末尾换行后一致。新增显式 --float32-matmul-precision highest，在构造后应用并记录设置；下次用户运行pi05-openpi-forward-02验证，误差门槛不变。七项CPU检查通过，修正设置后的GPU结果仍待用户执行。

## 2026-10-19 方法入口小规模就绪

用户OpenPI前向-02已1600元素EXACT_MATCH，TF32/matmul设置对齐后最大差为0。按用户新的自动排错授权，助手已完成原始输入适配、激活与反向验证：

- reproduction/results/pi05-calibration-ready-03：原始数据→模型输入和前向均EXACT_MATCH；官方HSIC hooks在4个真实样本上采集126矩阵，X=[4,2048]、Y=[4,350]；288个Fisher目标的梯度存在/有限性检查通过，明确记录末层结构性4个unused和1个zero梯度。峰值约11.77GiB。
- reproduction/results/pi05-gradient-reference-01：LeRobot与适配OpenPI的有符号预测、loss和4个代表参数梯度逐元素完全一致；末层unused行为两边一致。
- 5项原始输入CPU回归检查通过，GGUF重要性文件格式小型往返检查通过。原始输入适配器修正了B=1时OpenPI图像工具挤掉batch轴的行为，官方仓库未改。

新增calibration_adapter.py、run_calibration.py与export_native_calibration.py。正式方法入口复用官方统计/量化函数，保留32维loss、B=1，记录highest与禁用额外图像增强这一兼容路线。正式HSIC分数、Fisher累计、位宽分配与量化均尚未运行。当前可开始小规模学习实验；60轨迹数据尚未备齐，不能宣称完整论文规模或GGUF/C++闭环已经就绪。教学见 reproduction/pi05_lerobot/METHOD_WALKTHROUGH.md。


## 2026-10-19 第一次 HSIC 教学实验完成

用户在方法讲解后要求继续，助手已执行 run_calibration.py --stage hsic，使用 reproduction/data/pi05-readiness-01 的4帧，输出 reproduction/results/pi05-hsic-demo-01。报告PASS，126个目标完整，缓存激活有限且维度正确，F_out与sens公式逐项验证通过。CPU评分、hidden RBF/Y linear、lx=ly=1、seed42、B=1；模型前向highest且无额外增强。计时约35.91秒，峰值分配显存7.116GiB，进程已结束，GPU显存释放。

第0层Q实际F_out=2.323886785762，sens=-10.314717907926；默认分配器读取F_out，不能把负sens当报错。结果缓存沿用FP16格式，本轮评分使用内存FP32，重载缓存重算不保证逐位一致。官方ActQuant/OpenPI源码未改，旧结果未覆盖。教学与已执行命令解释见 reproduction/pi05_lerobot/HSIC_DEMO.md。

下一步可用教学分数演示位宽预算分配。尚未运行分配、Fisher累计或量化；4帧同轨迹分数不能作为可靠任务排序，正式60轨迹校准及GGUF/C++闭环仍待后续。不要重新运行已完成的排错步骤，不自动启动大型下载或长时间实验。


## 2026-10-19 位宽分配教学实验完成

用户继续后，直接调用官方allocate_pi05.py，使用4样本HSIC的F_out，IQ2_XS→Q4_K候选菜单，无额外分数变换。结果 reproduction/results/pi05-allocation-demo-01。驱动默认目标2.3471，实际2.347089947089947，59/126矩阵升级；目标参数共1,981,808,640。126名称、候选格式、观察维度对应参数量、参数加权预算、独立重算代理目标均通过，见audit.json。

发现官方run_hsic_quant_pi05.sh的2.3471常量与其注释所称的启发式匹配不符：allocator的iq2xs_heuristic实际bpw为2.324570105820106。未修改官方源码。补做相同预算分配 allocation-matched-budget.json，31矩阵升级，实际bpw恰好等于基线。对应代理目标J为2.639240733，启发式45.72979730；这只是四样本评分下的代理目标，不能宣称模型质量或闭环成功率提高。默认预算J=1.044177445。

代码和实际数值讲解见 reproduction/pi05_lerobot/ALLOCATION_DEMO.md，含已执行命令说明、加权预算、单位额外bit收益、小K/V和大MLP的成本差别。仅CPU分配，未加载模型或实际量化，未覆盖旧结果。下一步可做少量样本Fisher教学实验；正式60轨迹和GGUF/C++闭环仍未完成。


## 2026-10-19 四样本 Fisher 教学实验完成

用户明确“开始”后执行 run_calibration.py --stage fisher --calib-dir reproduction/data/pi05-readiness-01 --num-samples 4 --output reproduction/results/pi05-fisher-demo-01。复用原fisher_step与writer，B=1、mean32、highest、无额外图像增强。报告PASS，四次loss为0.05959867686、0.05429571867、0.08142635226、0.11324530095，均值0.077141512185；约77.66秒，峰值11.800335GiB，GPU进程已结束并释放显存。

输出 fisher_llm.gguf：126重要性矩阵（252个GGUF tensor），7,927,256,128字节，约7.383GiB；fisher_vision.gguf：162矩阵（324个tensor），1,716,415,968字节，约1.599GiB。另有losses.npy/indices.npy/run.json。重要性文件大是因为逐权重保存FP32，与只有4样本不矛盾。

export-audit.json全量分块读取所有重要性值，288目标名称/shape由checkpoint头与原映射独立核对；FP32正且有限、counts=1、元数据4样本、GGUF维度反转、视觉MLP padding均PASS。末层Q/O/gate/up/down恰好5矩阵全为FP32 1e-20，符合既有结构性零/unused梯度及writer floor规则。此检查不等于全部梯度LeRobot数值等价证明。源码哈希也记录在审计中。未改官方算法，未覆盖旧结果，未执行optimizer或实际量化。

教学见 reproduction/pi05_lerobot/FISHER_DEMO.md，包含命令解释、先平方再平均、mean32尺度、随机noise/time、GGUF布局和floor与Fisher原值的区别。下一步应检查checkpoint→GGUF权重导出（tied embedding、视觉padding等）与量化器per-weight重要性对接，再进行有界量化测试。完整60轨迹与GGUF/C++前向/闭环仍待验证。


## 2026-10-19 GGUF导出与有界量化衔接完成

reproduction/results/pi05-quant-bridge-01 包含官方完整未量化导出：unified/pi05.gguf (6,714,075,456字节)及llm/pali_llm_bf16.gguf (6,076,576,864字节，实际矩阵F16、norm F32)。checkpoint-view只用只读链接提供原模型文件和本地tokenizer，未改模型目录。export-run.json记录命令与官方源码哈希。

check_quantization_bridge.py最终selected-03检查PASS，约50.75秒。全部288目标按官方FP16转换后位级匹配，embedding三份副本与source相符，36组norm转换正确，全部统一模型张量有限；horizon50/action7/state8正确。23个LLM真实矩阵（包含18个V以满足量化器层数断言）与3个视觉矩阵已由本地C++工具量化，per-weight日志全部出现、实际格式和解码有限性通过。只是部分张量夹具，不能加载作完整模型。官方IQ2_XS只覆盖升级项的策略在相同23项上未出现格式差异，源码确有IQ2_XS关闭默认升级的逻辑，不要泛化称驱动格式有bug。

两处衔接问题处理在官方仓库外：1)原merge数值数组copy取parts[-1]，把image mean/std 3项截成1项，且复制GGUF内部头产生重复字段；merge_gguf_compat.py改用contents并跳过内部头，调用原tensor merge，小型真实权重文件复现旧问题且修复PASS，37元数据字段/4张量字节/norm保留通过。2)视觉Fisher up为[4352,1280]而未量化输入为[4352,1152]，C++工具先匹配补齐前尺寸，会忽略直接传入的Fisher；外部选定矩阵测试裁掉128列人工padding再让C++按原规则补齐，3项全部per-weight、无missing/ignoring。完整重要性文件原样保留。不能把该selected对齐文件直接当完整162项文件使用。

前两次只读扫描因CPU/FP16扫描开销中止，空selected/selected-02及interrupted记录保留；最终四线程与位级比较通过。主证据bridge-audit.json、metadata-merge-probe/audit.json、completion.json，教学QUANTIZATION_BRIDGE.md。官方ActQuant/OpenPI源码未改，GPU显存0。下一步先验证完整未量化统一GGUF的C++输入与前向；目前正式60轨迹、整模型低位量化、C++前向/闭环尚未完成。不要将文件导出/量化成功等同于模型行为一致。


## 2026-10-19 C++固定输入前向诊断：主要差异定位，严格门槛未通过

新文档 reproduction/pi05_lerobot/CPP_FORWARD_DIAGNOSTICS.md。新增 prepare_cpp_reference.py、cpp_forward_probe.cpp、compare_cpp_forward.py、run_cpp_probe.py，全部在官方仓库外。最新 reproduction/results/pi05-cpp-forward-08：构建/执行0，严格比较返回2，总状态NUMERICAL_MISMATCH，不能称为C++基线PASS。全部GPU诊断已结束，显存0，官方源码未改。

参考 pi05-cpp-reference-01 首先确认原BF16 loss与旧保存参考EXACT_MATCH，再按导出规则将rank>=2参数舍入FP16值并用FP32计算。保留源RoPE实际buffer。固定旧noise/time=0.5。压缩第三masked图像与文本padding为663有效prefix，151文本token；PyTorch压缩与全mask速度maxdiff2.6226e-6，在原abs/rel判据内通过。原BF16与此导出精度FP32参考本身maxdiff约0.053，不能混为量化误差。准备约35.57秒、峰值16.44GiB。

原server不读取state形成prompt，Pi05::tokenize_prompt还会清理换行；原prefix cache不屏蔽占位prefix keys且按物理长度编号。诊断传已验证的浮点像素与完整prompt token并去掉masked tokens，未将原server修成可用闭环策略。SPM原始完整prompt的151 token与参考完全一致，文本embedding完全一致。

精度case01默认速度同prefixmaxdiff0.02062845。case02请求GGML_PREC_F32仍0.01895764；case03另NVIDIA_TF32_OVERRIDE=0仍0.01842037。GGML CUBLAS默认TF32已证实。case05用相同数值F32存储诊断副本（pi05-cpp-f32-weights-01/pi05.gguf，13,425,726,784字节、459矩阵升精度）专家与03相同，故非单纯F16权重类型问题。case04图内cast触发GGML调度断言，失败保留，代码已移除此路径。

关键：Gemma父模块.to(bfloat16)把非持久registered inv_freq buffer也转BF16，之后仅参数选择性恢复FP32，model.float无法恢复已舍入频率；meta验证两个rotary buffer均BF16，最大频率舍入约0.00138819、位置663相位差约0.92037rad。CPP按配置重建理想F32频率，所以只核对checkpoint keys/values不够。外部探针用公开RoPE frequency factors匹配实际源buffer，按图输入名称写入（日志每次36项）。case06首版按tensor指针写入失败产生NaN；旧execution只记录进程0但数值检查失败。最新入口有非有限输出保护。

case07及可重放case08匹配RoPE后，同prefix速度maxdiff3.9935112e-5，mean9.76986e-7，RMSE3.05976e-6，50/1600超限且都在真实7维；完整CPPprefixmaxdiff5.55515e-5、82/1600超限。cached与joint逐元素EXACT_MATCH。严格atol1e-6/rtol1e-5未放宽。视觉两路RMSE7.194e-6、2.192e-5仍未过严格判据；原CPP timestep浮点公式与Torch double->float最大差8.5811e-5，本轮速度使用同一参考timestep输入，未混入该差异。

下一步继续定位剩余视觉算子、time生成、RoPE factor有限精度/归一化/规约顺序差异，不能自动调大容差。还需把状态prompt/浮点图像/有效token压缩/实际RoPE约定接入实际推理接口并检查固定初始noise的多步ODE。正式60轨迹、完整低位量化与LIBERO闭环仍待完成。run_cpp_probe默认matched_rope是诊断选项，原高层Pi05Pipeline/serve_policy尚未验证，不要直接拿来评估。


## 2026-10-19 完整LLM量化及同后端十步采样完成

用户明确要求赶紧开始量化后，使用已有四帧HSIC分配与Fisher，通过quantize_llm.py调用官方llama-quantize，全部126个LLM目标真实量化，CPU8线程约162.71秒。没有修改官方算法、放宽跨后端阈值、下载新数据或覆盖旧结果。finish_llm_quantization.py检查126项实际格式及per-weight重要性日志，并调用兼容合并器生成完整统一GGUF；814个tensor中126替换、688与基础文件字节一致，37项元数据完整保留。

产物 reproduction/results/pi05-quantized-demo-01/model/pi05.gguf，3,331,893,600字节；基础文件6,714,075,456字节，缩小50.3745%。目标矩阵实际参数加权block bpw为2.347089947；这不是整个模型平均位宽。SHA256为25d52c5e7a54608b604b15322ad09fa0b0502209bd988b45c313f54f8528b41e。视觉、embedding、action expert、projector和norm保持基础精度。

pi05-quant-baseline-01与pi05-quantized-forward-01采用同一C++探针源码、相同浮点输入/完整状态prompt/663有效prefix/源RoPE频率/初始噪声，十步Euler采样均执行0且输出有限。measure_quantization_effect.py得到真实7维反归一化动作RMSE0.02869504449、最大绝对差0.1758265325。此为单观测配对误差，无质量阈值或成功率结论。completion.json为ARTIFACT_AND_PAIRED_EXECUTION_PASS，quality_acceptance为NOT_EVALUATED；旧跨后端NUMERICAL_MISMATCH与原阈值全部保留。不要因旧报告重新阻断已授权教学量化，也不要称跨后端数值已通过。

复现入口quantize_llm.py、finish_llm_quantization.py、measure_quantization_effect.py及十步诊断参数均在官方仓库外；量化命令dry-run与实际运行完全一致。详见 reproduction/pi05_lerobot/QUANTIZED_MODEL_DEMO.md，其中逐条解释命令。模型目录runtime_contract.json记录诊断运行约定；原serve_policy/Pi05Pipeline尚未验证，不能直接称为可用闭环策略。下一阶段为正式规模校准与实际推理接口/LIBERO评估；当前四帧教学结果不等于论文复现。
