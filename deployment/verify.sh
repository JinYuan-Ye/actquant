#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec > >(tee "$ROOT/deployment/logs/verify-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
set -x
(
 source "$ROOT/deployment/activate_server.sh"
 uv pip check --python "$VIRTUAL_ENV/bin/python"
 timeout 120 python "$ROOT/deployment/smoke_server.py"
 timeout 60 python "$ROOT/src/ActQuant/tools/fisher-diag/compute_fisher_pi05.py" --help
 timeout 60 python "$ROOT/src/ActQuant/tools/hsic/compute_hsic_pi05.py" --help
 timeout 60 python "$ROOT/src/ActQuant/tools/pi0.5/serve_policy.py" --help
)
(
 source "$ROOT/deployment/activate_libero.sh"
 uv pip check --python "$VIRTUAL_ENV/bin/python"
 timeout 120 python "$ROOT/deployment/smoke_libero.py"
 timeout 60 python "$ROOT/src/openpi/examples/libero/main.py" --help
)
timeout 60 "$ROOT/envs/openpi-server/bin/python" "$ROOT/deployment/smoke_native.py"
echo "ALL ENVIRONMENT CHECKS PASSED (no model inference or evaluation)"
