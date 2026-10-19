#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-all}"
mkdir -p "$ROOT/deployment/logs" "$ROOT/deployment/evidence" "$ROOT/envs" "$ROOT/.cache/uv" "$ROOT/.local/python"
exec > >(tee -a "$ROOT/deployment/logs/install-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
export PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_PYTHON_INSTALL_DIR="$ROOT/.local/python" UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1
export GIT_LFS_SKIP_SMUDGE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
ACT="$ROOT/src/ActQuant"
OPENPI="$ROOT/src/openpi"
SERVER="$ROOT/envs/openpi-server"
CLIENT="$ROOT/envs/openpi-libero"
TOOLS="$ROOT/envs/build-tools"
ensure_repo() {
 local url="$1" dir="$2" rev="$3"
 if [[ -f "$dir/.upstream-snapshot.json" ]]; then
   python "$ROOT/deployment/verify_source_snapshot.py" --source-dir "$dir" --revision "$rev"
   return
 fi
 if [[ ! -d "$dir" ]]; then
   git clone --no-checkout --filter=blob:none "$url" "$dir"
   git -C "$dir" checkout "$rev"
 fi
 [[ "$(git -C "$dir" rev-parse HEAD)" == "$rev" ]] || { echo "Refusing to change existing checkout: $dir"; exit 1; }
}
sources() {
 ensure_repo https://github.com/arashakb/ActQuant.git "$ACT" b64791125070652fe6b554e244fe809c79ef5246
 ensure_repo https://github.com/Physical-Intelligence/openpi.git "$OPENPI" 215abfb217dbac7d5f1273282331b9b1866c0479
 git -C "$OPENPI" submodule update --init --depth 1 third_party/libero
}
server() {
 [[ -x "$SERVER/bin/python" ]] || uv venv --python /usr/bin/python3 "$SERVER"
 UV_PROJECT_ENVIRONMENT="$SERVER" uv sync --project "$OPENPI" --frozen --no-dev --inexact --python "$SERVER/bin/python"
 uv pip install --python "$SERVER/bin/python" pybind11==2.13.6
}
client() {
 export PATH="$TOOLS/bin:$PATH"
 [[ -x "$CLIENT/bin/python" ]] || uv venv --python 3.8.20 "$CLIENT"
 uv pip install --python "$CLIENT/bin/python" \
   -c "$ROOT/deployment/client-constraints.txt" \
   -r "$OPENPI/examples/libero/requirements.txt" -r "$OPENPI/third_party/libero/requirements.txt" \
   --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match \
   -e "$OPENPI/packages/openpi-client" -e "$OPENPI/third_party/libero"
}
build_tools() {
 [[ -x "$TOOLS/bin/python" ]] || uv venv --python /usr/bin/python3 "$TOOLS"
 uv pip install --python "$TOOLS/bin/python" cmake==3.31.6 ninja==1.11.1.4
}
build() {
 python "$ROOT/deployment/prepare_vendor.py"
 "$TOOLS/bin/cmake" -S "$ACT" -B "$ROOT/build/pi05" -G Ninja \
   -DCMAKE_MAKE_PROGRAM="$TOOLS/bin/ninja" -DCMAKE_BUILD_TYPE=Release \
   -DGGML_CUDA=ON -DBUILD_PI05_PYTHON=ON -DBUILD_PYTHON=OFF \
   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89 \
   -DPython_EXECUTABLE="$SERVER/bin/python" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
   -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
 "$TOOLS/bin/cmake" --build "$ROOT/build/pi05" --target pi05 pi05_py llama-quantize quantize-vision -j "${BUILD_JOBS:-8}"
 if [[ ! -e "$ACT/build_openpi" && ! -L "$ACT/build_openpi" ]]; then
   ln -s ../../build/pi05 "$ACT/build_openpi"
 elif [[ "$(readlink -f "$ACT/build_openpi")" != "$ROOT/build/pi05" ]]; then
   echo "Refusing to overwrite existing $ACT/build_openpi"; exit 1
 fi
}
case "$PHASE" in
 sources) sources;; server) server;; client) client;; tools) build_tools;; build) build;;
 all) sources; build_tools; server; client; \
 "$SERVER/bin/python" "$ROOT/deployment/configure.py"; \
 python "$ROOT/deployment/configure_libero.py"; \
 bash "$ROOT/deployment/install_render_libs.sh"; \
 (source "$ROOT/deployment/activate_libero.sh"; unset MUJOCO_EGL_DEVICE_ID; python "$ROOT/deployment/select_egl_device.py"); \
 build; bash "$ROOT/deployment/verify.sh"; python "$ROOT/deployment/inventory.py";;
 *) echo 'Usage: install.sh {sources|server|client|tools|build|all}'; exit 2;;
esac
