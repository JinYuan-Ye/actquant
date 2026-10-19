#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/deployment/evidence"
exec > >(tee "$ROOT/deployment/evidence/preflight-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
set -x
date -u
hostname
pwd
cat /etc/os-release
uname -a
df -h "$ROOT" /tmp
free -h
nvidia-smi
nvcc --version
git --version
uv --version
/usr/bin/python3 - <<'PY'
import sys, torch
print(sys.version, sys.executable)
print('torch',torch.__version__,'CUDA runtime',torch.version.cuda,'available',torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU',torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))
PY
for u in https://actquant.github.io/ https://github.com https://pypi.org/simple/ https://download.pytorch.org/whl/ https://huggingface.co https://arxiv.org; do
 curl -I -L --connect-timeout 8 --max-time 15 -s -o /dev/null -w '%{url_effective} %{http_code} %{time_total}s\n' "$u"
done
