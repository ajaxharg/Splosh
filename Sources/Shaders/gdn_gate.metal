#include <metal_stdlib>
using namespace metal;
struct GDNGateParams { uint length; uint dim; float epsilon; };
kernel void gdn_gate(device const float *input [[buffer(0)]], device const float *z [[buffer(1)]], device const float *weight [[buffer(2)]], device float *output [[buffer(3)]], constant GDNGateParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.length) return;
    uint base=(i/p.dim)*p.dim; float sum=0; for(uint j=0;j<p.dim;++j){float x=input[base+j];sum+=x*x;} float inv=rsqrt(sum/float(p.dim)+p.epsilon); float x=input[i]*inv*(1+weight[i%p.dim]); float g=z[i]; output[i]=x*g/(1+exp(-g));
}
