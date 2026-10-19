# 初始源码获取命令

以下命令在依赖安装前用于官方仓库分析；输出保存在 logs/source-clone.log、openpi-clone.log 和 libero-source.log。

```bash
mkdir -p /workspace/actquant/deployment/logs /workspace/actquant/deployment/evidence /workspace/actquant/src
python -m pip freeze > /workspace/actquant/deployment/evidence/preinstalled-pip-freeze.txt
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/arashakb/ActQuant.git /workspace/actquant/src/ActQuant
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git /workspace/actquant/src/openpi
GIT_LFS_SKIP_SMUDGE=1 git -C src/openpi submodule update --init --depth 1 third_party/libero
```

后续依赖安装/重试都通过 deployment/install.sh、install_render_libs.sh 执行，日志包含展开后的命令与每个安装版本。首次各阶段执行顺序：server → client（缺 CMake 失败）→ tools → render → client 重试成功 → render 匹配版本补齐 → server 复核 → configure → build → verify → all 完整重跑。安装脚本最终调整为 tools 先于 client。

最终 install.sh 将源码固定到报告中的提交，不再依赖当时远程分支 HEAD。不会自动执行模型下载或实验脚本。
