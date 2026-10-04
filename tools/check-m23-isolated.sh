#!/usr/bin/env bash
set -euo pipefail
stage="$1"; filter="$2"; expected="$3"; shift 3
root="$(pwd)/.build/tmp/$stage"
perm="Sources/SploshCore/Resources/default.metallib"
test -f "$perm"
before="$(shasum -a 256 "$perm" | awk '{print $1}')"
cleanup() { rm -rf "$root"; after="$(shasum -a 256 "$perm" | awk '{print $1}')"; echo "$filter permanent SHA-256 before=$before after=$after"; test "$before" = "$after"; test ! -e "$root"; }
trap cleanup EXIT
mkdir -p "$root" .build/metal-module-cache
rm -f "$root"/*.air "$root/milestone.metallib"
for src in "$@"; do
  xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path="$(pwd)/.build/metal-module-cache" -c "$src" -o "$root/$(basename "$src" .metal).air"
done
xcrun -sdk macosx metallib "$root"/*.air -o "$root/milestone.metallib"
./tools/check-metallib-exports "$root/milestone.metallib" $expected
SPLOSH_TEST_METALLIB="$root/milestone.metallib" ./tools/swift-test-filter "$filter"
echo "$filter cleanup path=$root"
