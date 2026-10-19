#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/deployment/logs" "$ROOT/.cache/debs" "$ROOT/.local/render"
exec > >(tee -a "$ROOT/deployment/logs/render-install-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
set -x
cd "$ROOT/.cache/debs"
# Download/extract only; no apt install, no changes to system packages.
apt-get download libegl1=1.4.0-1 libegl-mesa0=23.2.1-1ubuntu3.1~22.04.4 libgbm1=23.2.1-1ubuntu3.1~22.04.4 libwayland-client0=1.20.0-1ubuntu0.1 libwayland-server0=1.20.0-1ubuntu0.1 libgl1-mesa-dri=23.2.1-1ubuntu3.1~22.04.4 libglapi-mesa=23.2.1-1ubuntu3.1~22.04.4
for deb in ./*.deb; do
 dpkg-deb -x "$deb" "$ROOT/.local/render"
done
sha256sum ./*.deb > "$ROOT/deployment/evidence/render-deb-sha256.txt"
LD_LIBRARY_PATH="$ROOT/.local/render/usr/lib/x86_64-linux-gnu" ldd "$ROOT/.local/render/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0"
