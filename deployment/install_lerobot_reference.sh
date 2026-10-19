#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p deployment/logs deployment/evidence envs
exec > >(tee -a "deployment/logs/lerobot-reference-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
set -x
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_LINK_MODE=copy PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [[ ! -d src/lerobot-v044 ]]; then
    git clone --depth 1 --branch v0.4.4 https://github.com/huggingface/lerobot.git src/lerobot-v044
fi
[[ "$(git -C src/lerobot-v044 rev-parse HEAD)" == 8fff0fde7c79f23a93d845d1a50e985de01f8b8a ]]
[[ -x envs/lerobot-reference/bin/python ]] || uv venv --python /usr/bin/python3 envs/lerobot-reference
if [[ ! -s deployment/lerobot-reference.lock ]]; then
    uv pip compile deployment/lerobot-reference.in \
        --override deployment/lerobot-reference-overrides.txt \
        --python envs/lerobot-reference/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        --index-strategy unsafe-best-match \
        --output-file deployment/lerobot-reference.lock
fi
uv pip sync --python envs/lerobot-reference/bin/python deployment/lerobot-reference.lock \
    --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
uv pip check --python envs/lerobot-reference/bin/python
uv pip freeze --python envs/lerobot-reference/bin/python > deployment/evidence/lerobot-reference-freeze.txt
