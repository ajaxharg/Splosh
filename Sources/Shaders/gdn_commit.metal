#include <metal_stdlib>
using namespace metal;
struct GDNCommitParams { uint count; };
kernel void gdn_commit(device const float *inactive [[buffer(0)]], device float *active [[buffer(1)]], constant GDNCommitParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < p.count) active[i] = inactive[i]; }
