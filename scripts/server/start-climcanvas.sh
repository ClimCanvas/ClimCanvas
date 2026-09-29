#!/usr/bin/env bash
# start-climcanvas — user launcher: looks up your port in the port table
# (ports.tsv) and starts ClimCanvas on it, so nobody has to remember a port number.
#
# Usage (each user, under their own account):
#   scripts/server/start-climcanvas.sh
#
# Environment variables (optional overrides):
#   CLIMCANVAS_PORTS_TABLE  path of the port table       (default /etc/climcanvas/ports.tsv)
#   CLIMCANVAS_APP_DIR      directory containing app.py  (default: ../.. from this script)
#   CLIMCANVAS_PYTHON       python executable            (default python3)
#   CLIMCANVAS_ALLOWED_DIRS allowed data directories (colon-separated; passed on as CC_ALLOWED_DIRS)
set -euo pipefail

TABLE="${CLIMCANVAS_PORTS_TABLE:-/etc/climcanvas/ports.tsv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${CLIMCANVAS_APP_DIR:-$(cd "$HERE/../.." && pwd)}"
PY="${CLIMCANVAS_PYTHON:-python3}"

me="$(id -un)"
[ -f "$TABLE" ] || { echo "port table not found: $TABLE (ask the administrator to run: climcanvas-ports assign $me)" >&2; exit 1; }

port="$(grep -vE '^[[:space:]]*(#|$)' "$TABLE" | awk -v u="$me" '$1==u {print $2; exit}')"
[ -n "$port" ] || { echo "no port is assigned to $me (ask the administrator to run: climcanvas-ports assign $me)" >&2; exit 1; }

echo "Starting ClimCanvas on 127.0.0.1:$port (as $me)"
echo "If you logged in without a tunnel, run this on your own machine (8501 = any free port there):"
echo "  ssh -L 8501:localhost:$port $me@<server>"
echo "Then open http://localhost:8501 in a browser on your own machine."

if [ -n "${CLIMCANVAS_ALLOWED_DIRS:-}" ]; then
    export CC_ALLOWED_DIRS="$CLIMCANVAS_ALLOWED_DIRS"
fi

exec "$PY" -m streamlit run "$APP_DIR/app.py" \
    --server.headless=true \
    --server.address=127.0.0.1 \
    --server.port="$port"
