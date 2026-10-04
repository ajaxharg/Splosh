#include <metal_stdlib>

using namespace metal;

// MLX affine q4 GEMM.  A is [M,K] in bf16 and the packed right operand is
// logically [N,K], so C = A * transpose(dequant(Q)).  Each q4 row is padded
// independently to the supplied stride; words contain low-first nibbles.
struct GemmQ4Params {
    uint rows;
    uint columns;
    uint inner;
    uint q4RowStrideWords;
    uint orientation; // 0: rowsByK, 1: kByColumns
    uint groupsPerRow;
};

inline float q4_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

inline uint q4_word_index(uint outputColumn, uint k, constant GemmQ4Params &p) {
    return p.orientation == 0u
        ? outputColumn * p.q4RowStrideWords + k / 8u
        : k * p.q4RowStrideWords + outputColumn / 8u;
}

inline uint q4_shift(uint outputColumn, uint k, constant GemmQ4Params &p) {
    return p.orientation == 0u ? (k % 8u) * 4u : (outputColumn % 8u) * 4u;
}

kernel void gemm_q4(device const bfloat *a [[buffer(0)]],
                    device const uint *packed [[buffer(1)]],
                    device const ushort *scales [[buffer(2)]],
                    device const ushort *biases [[buffer(3)]],
                    device float *c [[buffer(4)]],
                    constant GemmQ4Params &p [[buffer(5)]],
                    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.columns || gid.y >= p.rows || p.inner == 0u ||
        p.q4RowStrideWords == 0u || p.groupsPerRow == 0u ||
        (p.orientation > 1u)) return;

    const uint row = gid.y;
    const uint outputColumn = gid.x;
    float sum = 0.0f;
    for (uint k = 0; k < p.inner; ++k) {
        const uint word = packed[q4_word_index(outputColumn, k, p)];
        const uint q = (word >> q4_shift(outputColumn, k, p)) & 0xFu;
        const uint sidecar = outputColumn * p.groupsPerRow + k / 64u;
        const float weight = q4_bf16(scales[sidecar]) * float(q) + q4_bf16(biases[sidecar]);
        sum += float(a[row * p.inner + k]) * weight;
    }
    c[row * p.columns + outputColumn] = sum;
}
