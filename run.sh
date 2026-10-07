#!/bin/bash
# Start the Splosh server from the repository, building first when there is no release binary.
#
#   ./run.sh                  serve the model named by `model` in splosh.toml
#   ./run.sh --model uq6      any arguments go to `splosh serve`
#   ./run.sh --build          rebuild the shaders and the binary, then serve
#   ./run.sh --restart        replace the engine of the server that is already running
#   ./run.sh --takeover       start in place of the server that is already running
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
BIN="$ROOT/.build/release/splosh"
PORT=8091

build=0
restart=0
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) build=1 ;;
    --restart|--takeover) restart=1; args+=("$1") ;;
    --port) PORT="${2:-}"; args+=("$1") ;;
    *) args+=("$1") ;;
  esac
  shift
done

if [[ "$build" == 1 || ! -x "$BIN" ]]; then
  make shaders
  swift build -c release --disable-sandbox
fi

# Two servers would load the model twice, and running out of memory locks the Mac.
if [[ "$restart" == 0 ]] && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "run.sh: something is already listening on 127.0.0.1:$PORT." >&2
  echo "run.sh: --restart replaces its engine, --takeover starts in its place." >&2
  exit 1
fi

exec "$BIN" serve ${args[@]+"${args[@]}"}
