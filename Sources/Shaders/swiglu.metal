#include <metal_stdlib>
#include "common/abi.h"

using namespace metal;

// SwiGLU activation for the dense MLP: silu(gate) * up.
// Inputs and output are bfloat16, while the activation and product are evaluated in
// float32.  This is the model's bf16 path; keeping the accumulator wide avoids an
// avoidable loss before the final bf16 store.
struct SwiGLUParams {
    uint length;
};

kernel void swiglu(device const bfloat *gate [[buffer(0)]],
                   device const bfloat *up [[buffer(1)]],
                   device bfloat *output [[buffer(2)]],
                   constant SwiGLUParams &params [[buffer(3)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid >= params.length) {
        return;
    }

    const float g = float(gate[gid]);
    const float u = float(up[gid]);
    // SiLU, not sigmoid or GELU: x * sigmoid(x).
    const float silu = g / (1.0f + exp(-g));
    output[gid] = bfloat(silu * u);
}
