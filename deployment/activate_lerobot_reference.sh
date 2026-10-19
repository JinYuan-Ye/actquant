# source this file from Bash. Keep this interpreter separate from openpi-server.
ACTQUANT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ACTQUANT_ROOT/envs/lerobot-reference/bin/activate"
unset PYTHONPATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$ACTQUANT_ROOT/.cache/huggingface" PYTHONNOUSERSITE=1
