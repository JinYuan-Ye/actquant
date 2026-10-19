# source this file from Bash. Default rendering uses local Mesa, not NVIDIA EGL.
ACTQUANT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ACTQUANT_ROOT/envs/openpi-libero/bin/activate"
export LIBERO_CONFIG_PATH="$ACTQUANT_ROOT/.config/libero"
export LD_LIBRARY_PATH="$ACTQUANT_ROOT/.local/render/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$ACTQUANT_ROOT/.local/render/usr/share/glvnd/egl_vendor.d/50_mesa.json"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NUMBA_CACHE_DIR="$ACTQUANT_ROOT/.cache/numba"
export XDG_CACHE_HOME="$ACTQUANT_ROOT/.cache"
export LIBGL_DRIVERS_PATH="$ACTQUANT_ROOT/.local/render/usr/lib/x86_64-linux-gnu/dri"
export PYTHONPATH="$ACTQUANT_ROOT/src/openpi/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export LIBGL_ALWAYS_SOFTWARE=1
# Simulator uses software EGL; CUDA is used separately by the policy server.
unset CUDA_VISIBLE_DEVICES
if [[ -f "$ACTQUANT_ROOT/.config/libero/egl_device_id" ]]; then
 export MUJOCO_EGL_DEVICE_ID="$(cat "$ACTQUANT_ROOT/.config/libero/egl_device_id")"
fi
export PYTHONNOUSERSITE=1
