# source this file from Bash.
ACTQUANT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ACTQUANT_ROOT/envs/openpi-server/bin/activate"
export PYTHONPATH="$ACTQUANT_ROOT/build/pi05/bin:$ACTQUANT_ROOT/src/ActQuant/gguf-py${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$ACTQUANT_ROOT/.cache/huggingface" XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONNOUSERSITE=1
