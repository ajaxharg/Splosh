#include <metal_stdlib>

using namespace metal;

// Partial-dimension NeoX RoPE with Qwen interleaved mRoPE frequencies.
// The first rotaryDim channels rotate; the remainder pass through unchanged.
struct RopeMropeParams {
    uint headDim;
    uint rotaryDim;
    uint positionT;
    uint positionH;
    uint positionW;
    uint sectionH;
    uint sectionW;
    float ropeTheta;
    float attentionScaling;
};

kernel void rope_mrope(device const float *input [[buffer(0)]],
                       device float *output [[buffer(1)]],
                       constant RopeMropeParams &params [[buffer(2)]],
                       uint gid [[thread_position_in_grid]])
{
    if (gid >= params.headDim || params.rotaryDim == 0u || params.rotaryDim > params.headDim ||
        (params.rotaryDim & 1u) != 0u) {
        return;
    }
    if (gid >= params.rotaryDim) {
        output[gid] = input[gid];
        return;
    }

    const uint rotaryHalf = params.rotaryDim / 2u;
    const uint slot = gid < rotaryHalf ? gid : gid - rotaryHalf;
    uint position = params.positionT;
    // Literal interleaved mRoPE bounds: slice(offset, section * 3, 3).
    if (slot >= 1u && slot < params.sectionH * 3u && ((slot - 1u) % 3u) == 0u) {
        position = params.positionH;
    } else if (slot >= 2u && slot < params.sectionW * 3u && ((slot - 2u) % 3u) == 0u) {
        position = params.positionW;
    }

    const float exponent = (2.0f * float(slot)) / float(params.rotaryDim);
    const float inverseFrequency = 1.0f / pow(params.ropeTheta, exponent);
    const float angle = float(position) * inverseFrequency;
    const float c = cos(angle) * params.attentionScaling;
    const float s = sin(angle) * params.attentionScaling;
    const float x = input[gid];
    const float paired = input[gid < rotaryHalf ? gid + rotaryHalf : gid - rotaryHalf];
    // rotate_half(x) = cat(-x[half:], x[:half]).
    output[gid] = x * c + (gid < rotaryHalf ? -paired : paired) * s;
}
