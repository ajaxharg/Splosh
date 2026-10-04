#include <metal_stdlib>
using namespace metal;
struct GDNPrepareParams { uint channels; uint length; };
kernel void gdn_prepare(device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]], device float *output [[buffer(2)]], constant GDNPrepareParams &p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.channels || gid.y >= p.length) return;
    uint t = gid.y, c = gid.x;
    float x0 = t >= 3 ? input[(t-3)*p.channels+c] : 0.0f;
    float x1 = t >= 2 ? input[(t-2)*p.channels+c] : 0.0f;
    float x2 = t >= 1 ? input[(t-1)*p.channels+c] : 0.0f;
    float x3 = input[t*p.channels+c];
    float y = weights[c*4] * x0 + weights[c*4+1] * x1 + weights[c*4+2] * x2 + weights[c*4+3] * x3;
    output[t*p.channels+c] = y / (1.0f + exp(-y));
}
