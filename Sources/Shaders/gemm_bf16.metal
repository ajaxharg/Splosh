#include <metal_stdlib>
#include "common/abi.h"

using namespace metal;

// Row-major dense GEMM: C[M,N] = A[M,K] * B[K,N].  The operands are bf16
// storage values, promoted to float for every multiply and accumulated in fp32.
struct GemmBF16Params {
    uint rows;
    uint columns;
    uint inner;
};

kernel void gemm_bf16(device const bfloat *a [[buffer(0)]],
                      device const bfloat *b [[buffer(1)]],
                      device float *c [[buffer(2)]],
                      constant GemmBF16Params &p [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.columns || gid.y >= p.rows) return;
    float sum = 0.0f;
    const uint row = gid.y;
    const uint col = gid.x;
    for (uint k = 0; k < p.inner; ++k) {
        sum += float(a[row * p.inner + k]) * float(b[k * p.columns + col]);
    }
    c[row * p.columns + col] = sum;
}
