# 全量量化完成后的GitHub发布

用户已明确授权将新增代码、说明文件和全量模型发布到 `https://github.com/JinYuan-Ye/actquant`。服务器GitHub CLI已完成账号授权，并确认该仓库具有push权限。此文档不包含任何凭据。

## 发布门槛和进度

发布程序等待 `reproduction/results/pi05-full-01/pipeline.json` 为PASS，且每个阶段均PASS。检查校准覆盖为60轨迹11,777帧、完整Fisher/量化产物审计、单观测配对执行状态以及模型SHA256。计算失败时停止发布，保留错误。

当前发布状态位于 `exports/github-publication-full-01.json`。WAITING_FOR_FULL_QUANTIZATION仅表示等待，不代表上传完成；只有PUBLISHED加远端commit和release_url才表示实际上传并核验完成。

## 仓库组织

`src/ActQuant`包含固定上游提交的840个跟踪文件及原许可证，保持原文件内容不变。快照清单 `.upstream-snapshot.json` 保存每个文件SHA256；安装脚本支持核对快照，避免把外层复现仓库的Git提交误认成ActQuant上游提交。

新增代码在 `reproduction/pi05_lerobot` 和 `deployment`。OpenPI、LeRobot以固定Git子模块提交引用，保留原工作区相对路径。根README指向数据格式、方法、工程问题、全量审计和模型附件。保留历史教学/失败报告，但明确区分其状态和全量结果。

发布目录是 `exports/github-actquant-full-01`，与正在计算的原工作区隔离。只选择代码、说明、固定版本依赖信息和小型JSON证据；原模型、数据集、恢复快照、环境、凭据和私有交接历史不进入Git。示例传输脚本的服务器地址替换为占位符。

## 大模型附件

GitHub Release单附件须小于2GiB，官方说明：https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases 。完整GGUF按不超过1GiB拆分为 `pi05.gguf.part001` 等附件；并上传tokenizer、归一化统计、RoPE频率快照、运行约定、量化及验证清单、模型来源说明、SHA256清单和跨平台还原脚本。

模型分片只是传输方式，不改变权重。`MODEL_SHA256.json`记录分片及完整模型大小和摘要。用户下载所有附件后运行 `restore_release_model.py --directory 下载目录`，逐项核验后还原原始完整GGUF，拒绝覆盖已有模型。

## 自动执行

```bash
python deployment/publish_after_full_quantization.py
```

命令使用服务器已授权的GitHub CLI，等待全量流水线完成后，重新生成可审阅目录并核验源代码和模型。随后创建提交、正常推送main（不force）、创建草稿Release、上传并检查远端附件大小与SHA256，再公开Release。文件锁拒绝重复运行；已有附件只在摘要一致时复用，不覆盖不同内容。

该入口由助手启动时，不需要用户再启动一份。若上传失败，计算结果仍保留，查看发布状态和日志后可重新执行。状态文件会记录失败原因；没有PUBLISHED就不能宣称模型已上传。

## 验证范围

全量校准与量化通过也不代表LIBERO成功率通过。跨后端严格NUMERICAL_MISMATCH及原容差保留，发布说明会公开当前验证范围。此发布不包括在GitHub上自动运行新GPU实验或新建issue。
