#include <metal_stdlib>
#include "common/abi.h"

using namespace metal;

// Zero-centred RMSNorm. The variance and reciprocal root are deliberately computed in
// fp32; the model's scale is (1 + weight), not weight alone (rev4 §4.7).
struct RMSNormParams {
    uint length;
    float epsilon;
};

kernel void rmsnorm(device const float *input [[buffer(0)]],
                    device const float *weight [[buffer(1)]],
                    device float *output [[buffer(2)]],
                    constant RMSNormParams &params [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid != 0 || params.length == 0) {
        return;
    }

    float sum = 0.0f;
    for (uint i = 0; i < params.length; ++i) {
        const float value = input[i];
        sum += value * value;
    }
    const float inverseRMS = rsqrt(sum / float(params.length) + params.epsilon);
    for (uint i = 0; i < params.length; ++i) {
        output[i] = input[i] * inverseRMS * (1.0f + weight[i]);
    }
}
