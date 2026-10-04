#!/bin/bash
# Splosh M0.10 shell preflight. The Swift doctor owns the structured report; this entry point
# repeats the non-Swift host probes and then refuses any required-through-M1 FAIL.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fail=0
pass() { printf 'PASS preflight-%s %s %s %s\n' "$1" "$2" "$3" "$4"; }
bad() { printf 'FAIL preflight-%s %s %s %s\n' "$1" "$2" "$3" "$4" >&2; fail=1; }

[[ "$(uname -m)" == arm64 ]] && pass arch arm64 arm64 'run on Apple silicon' || bad arch "$(uname -m)" arm64 'run on Apple silicon'
macos="$(sw_vers -productVersion 2>/dev/null || true)"
[[ "$macos" == 27.* ]] && pass macos "$macos" 27.x 'use macOS 27' || bad macos "$macos" 27.x 'use macOS 27'
swift_line="$(swift --version 2>/dev/null | head -1 || true)"
[[ "$swift_line" == *"Swift version: 6.4"* || "$swift_line" == *"Swift version 6.4"* ]] && pass swift "$swift_line" 'Swift 6.4' 'select Swift 6.4' || bad swift "$swift_line" 'Swift 6.4' 'select Swift 6.4'
command -v xcrun >/dev/null && command -v swift >/dev/null && pass tools available metal/metallib/metal-nm 'Xcode tools' 'select Xcode 27' || bad tools unavailable 'metal metallib metal-nm' 'select Xcode 27'
for tool in metal metallib metal-nm; do xcrun --find "$tool" >/dev/null 2>&1 && pass "tool-$tool" available available 'install/select Xcode Metal tools' || bad "tool-$tool" missing available 'install/select Xcode Metal tools'; done
mkdir -p .build/metal-module-cache .build/preflight-probe inputs artifacts
cat > .build/preflight-probe/probe.metal <<'EOF'
#include <metal_stdlib>
using namespace metal;
kernel void preflight_probe(uint3 id [[thread_position_in_grid]]) {}
EOF
if xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path="$ROOT/.build/metal-module-cache" -c .build/preflight-probe/probe.metal -o .build/preflight-probe/probe.air >/dev/null 2>&1 && xcrun -sdk macosx metallib .build/preflight-probe/probe.air -o .build/preflight-probe/probe.metallib >/dev/null 2>&1; then pass metal-module-cache compiled-in-workspace 'real Metal compile' 'use -fmodules-cache-path inside workspace'; else bad metal-module-cache compile-failed 'real Metal compile' 'use -fmodules-cache-path inside workspace'; fi
[[ -x "$ROOT/tools/fetch_inputs.py" ]] && python3 tools/fetch_inputs.py --help >/dev/null 2>&1 && pass fetch-help available 'python3 fetch_inputs.py --help' 'install Python 3' || bad fetch-help unavailable 'python3 fetch_inputs.py --help' 'install Python 3'
free="$(df -kP . | awk 'NR==2 {print $4*1024}')"; min=$((10*1024*1024*1024)); [[ "$free" -ge "$min" ]] && pass disk "$free bytes" ">=$min bytes" 'free at least 10 GiB' || bad disk "$free bytes" ">=$min bytes" 'free at least 10 GiB'
if "$ROOT/.build/out/Products/Debug/splosh" doctor >/tmp/splosh-preflight-doctor.out 2>/tmp/splosh-preflight-doctor.err; then pass doctor 'required checks passed' 'no required FAIL' 'inspect deferred checks'; else cat /tmp/splosh-preflight-doctor.out; cat /tmp/splosh-preflight-doctor.err >&2; bad doctor 'required doctor check failed' 'no required FAIL' 'run splosh doctor'; fi
exit "$fail"
