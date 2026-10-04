# M3.6 shipping-SDK alignment probe

## Scope and prerequisite status

This is the M3.6 probe required before M3.1. The pinned MLX q4 pack is now materialized under `inputs/mlx-q4/` and its index/shard manifest verifies; BF16 reference assets remain absent. The probe itself uses a real shipping Metal SDK and real Apple GPU tensor creation/validation; no model-weight conversion or byte-exact verification is claimed here. The probe itself uses a real shipping Metal SDK and real Apple GPU tensor creation/validation.

## Environment / SDK

- Host: Apple M5 Pro (`Apple M5 Pro`), macOS 27.0 / build 26A428.
- Swift: Apple Swift 6.4.0.33.1; target `arm64-apple-macos27.0`.
- SDK: `/Applications/Xcode-27.0.0-Beta.6.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk` (macOS 27.0).
- API: Metal `MTLTensorDescriptor` / buffer-backed `MTLTensor` (`MTLDevice.makeTensor(descriptor:attachments:)`).
- Device printed by probe: `Apple M5 Pro`.

## Owned reproducible fixture

- Source: `audit/scratch/alignmentprobe/Package.swift`
- Probe: `audit/scratch/alignmentprobe/Sources/alignmentprobe/main.swift`
- It creates a real 2-D `uint4` tensor with shape `[32, 4]`, innermost stride `1`, and tests row strides of 128 and 160 bytes. The fixture uses `.compute`, shared storage, a real buffer attachment, and prints the backing pointer modulo 128.

## Exact command and exit

From repository root:

```sh
S="$PWD/audit/scratch"
export TMPDIR="$S/tmp"
swift build --disable-sandbox \
  --package-path "$S/alignmentprobe" \
  --cache-path "$S/spmcache" \
  --config-path "$S/spmconfig" \
  --security-path "$S/spmsecurity"
# exit 0
"$S/alignmentprobe/.build/debug/alignmentprobe"
# exit 0
```

The build command emitted `BUILD_EXIT=0`; the executable emitted `RUN_EXIT=0`.

## Probe output (verbatim)

```text
device=Apple M5 Pro
sdk=macOS 27.0 Metal tensor descriptor API
stride_bytes=128 result=CREATE_OK required_size=512 required_align=1 tensor_strides=Optional(<MTLTensorExtents: ...>
    Rank = 2
    Extents = [ 1, 256 ]) pointer_mod_128=0
stride_bytes=160 result=CREATE_FAIL error=Error Domain=MTLTensorDomain Code=2 "Tensor Descriptor Validation
[tensor.strides extentAtDimensionIndex:1] (320 elements) must be aligned to 256 elements (128 byte aligned) when tensor.dataType is a 4-bit format MTLTensorDataType.
"
```

The pointer for the accepted case was 128-byte aligned (`pointer_mod_128=0`). The 160-byte row stride is rejected by the shipping SDK before dispatch, with an explicit 128-byte validation error. This is a contradictory fixture in the sense required by M3.6: the non-128 stride does not work; it is not treated as an acceptance or numerical pass.

## Selected rule

**Rows must be padded to a 128-byte row stride for 4-bit Metal tensors; pointers must be 128-byte aligned.** Evidence is the SDK's real validation result above, not a hard-coded assumption. The probe also exercised the required 32-element first dimension (`shape[0] = 32`).

This selected rule is narrower than general section/extent alignment: it applies to the q4 tensor row stride and base pointer. It does not establish unrelated packed-section alignment.

## `convert --verify` consumption status

M3.1 has not landed: `Sources/SploshQuant/Packing.swift` is still a placeholder and `Sources/SploshCLI/ConvertCommand.swift` still reports `notImplemented`. Therefore no `convert --verify` command can truthfully consume this rule yet, and no model bytes or verification result are claimed here. M3.1 must consume this recorded rule (and rerun its affected samples) rather than independently hard-code a different choice. Until then M3.1 remains blocked by the M2.7 missing packs and the unimplemented converter; this audit deliberately preserves that block.

## Earlier failed attempts retained as evidence

- `swift run ... alignmentprobe` from the repository package: exit **1**, `error: no executable product named 'alignmentprobe'` (fixture is a separate package).
- `swift run ... --target alignmentprobe`: exit **64**, `error: Unknown option '--target'` (SwiftPM run has no such option here).
- First fixture build: exit **1**, Swift API diagnostics required `tensorSizeAndAlign(descriptor:)` and throwing `makeTensor` spelling.
- Second fixture build: exit **1**, API diagnostic required `setBuffer(_:offset:for:)` (Swift 6 rename).

Those were fixture-authoring failures, not probe results; the final fixture build and run above both exit 0.
