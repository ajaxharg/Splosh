#include <metal_stdlib>
using namespace metal;
struct GDNDecodeParams { uint heads; uint keyDim; uint valueDim; };
kernel void gdn_decode(device const float *query [[buffer(0)]], device const float *key [[buffer(1)]], device const float *value [[buffer(2)]], device const float *beta [[buffer(3)]], device const float *decay [[buffer(4)]], device float *state [[buffer(5)]], device float *output [[buffer(6)]], constant GDNDecodeParams &p [[buffer(7)]], uint h [[thread_position_in_grid]]) {
    if (h >= p.heads) return;
    uint sb = h*p.keyDim*p.valueDim;
    float d = exp(decay[h]);
    for (uint i=0; i<p.keyDim*p.valueDim; ++i) state[sb+i] *= d;
    for (uint j=0; j<p.valueDim; ++j) {
        float predicted=0; for (uint i=0;i<p.keyDim;++i) predicted += state[sb+i*p.valueDim+j]*key[h*p.keyDim+i];
        float delta=beta[h]*(value[h*p.valueDim+j]-predicted);
        for (uint i=0;i<p.keyDim;++i) state[sb+i*p.valueDim+j] += key[h*p.keyDim+i]*delta;
    }
    for (uint j=0;j<p.valueDim;++j) { float o=0; for (uint i=0;i<p.keyDim;++i) o += state[sb+i*p.valueDim+j]*query[h*p.keyDim+i]; output[h*p.valueDim+j]=o; }
}
