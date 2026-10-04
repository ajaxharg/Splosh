#include <metal_stdlib>

using namespace metal;

// engine_q8.metal — the lane kernels of engine.metal for 8-bit weights.
//
// The MLX 8-bit pack is the 4-bit one with a byte per code: affine, group 64, four codes per
// U32 word, low byte first, so a quant group is sixteen words where q4 has eight. Scales and
// biases are the same bf16 sidecars. The accelerator kernels for these weights are in
// engine_na.metal; these are the ones that dequantise in the shader: the embedding gather, the
// small row-major projections, and the one- and two-row steps over the tiled layout
// ([tile of 128 rows][quant group][row][64 bytes]).
//
// Each kernel has the grid, the bindings and the parameters of its q4 counterpart in
// engine.metal, named there with `q4` where this has `q8`.

#define SP_LANES 32u

inline float sp_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline float sp_silu(float x) { return x / (1.0f + exp(-x)); }

// The four weights of one word.
inline float4 sp_q8(uint word, float s, float b)
{
    return s * float4((uint4(word) >> uint4(0u, 8u, 16u, 24u)) & uint4(0xFFu)) + b;
}

struct SpEmbedParams {
    uint rows;
    uint strideWords;
    uint groupsPerRow;
    uint hidden;
};

struct SpGemmParams {
    uint rows;
    uint outDim;
    uint inner;
    uint strideWords;
    uint groupsPerRow;
    uint hasResidual;
    uint inStride;
    uint outStride;
};

// Embedding: gather a row per token and dequantise it. Thread grid: (strideWords, rows).
kernel void sp_embed_q8(device const uint *packed [[buffer(0)]],
                        device const ushort *scales [[buffer(1)]],
                        device const ushort *biases [[buffer(2)]],
                        device const uint *tokens [[buffer(3)]],
                        device float *out [[buffer(4)]],
                        constant SpEmbedParams &p [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.strideWords || gid.y >= p.rows) return;
    const uint token = tokens[gid.y];
    const uint group = token * p.groupsPerRow + (gid.x >> 4);
    *(device packed_float4 *)(out + gid.y * p.hidden + gid.x * 4u) =
        sp_q8(packed[token * p.strideWords + gid.x], sp_bf16(scales[group]), sp_bf16(biases[group]));
}

// A small row-major weight from a bf16 operand (the 48-row a and b projections under a deferred
// norm). Threadgroup grid: (outDim, rows), 32 threads per group.
kernel void sp_gemm_q8_b(device const uint *packed [[buffer(0)]],
                         device const ushort *scales [[buffer(1)]],
                         device const ushort *biases [[buffer(2)]],
                         device const bfloat *a [[buffer(3)]],
                         device float *out [[buffer(4)]],
                         constant SpGemmParams &p [[buffer(6)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_threadgroup]])
{
    const uint n = group.x, row = group.y;
    if (n >= p.outDim || row >= p.rows) return;
    const uint words = p.inner >> 2;
    const uint wordBase = n * p.strideWords;
    const uint groupBase = n * p.groupsPerRow;
    float acc = 0.0f;
    for (uint w = lane; w < words; w += SP_LANES) {
        const uint gi = groupBase + (w >> 4);
        const float4 weights = sp_q8(packed[wordBase + w], sp_bf16(scales[gi]), sp_bf16(biases[gi]));
        acc += dot(float4(bfloat4(*(device const packed_bfloat4 *)(a + row * p.inStride + (w << 2)))), weights);
    }
    const float total = simd_sum(acc);
    if (lane == 0u) out[row * p.outStride + n] = total;
}

// Row-major weights, R activation rows per pass. Threadgroup grid: (outDim, ceil(rows / R)).
#define SP_DEFINE_GEMM_Q8(NAME, R)                                                                 \
kernel void NAME(device const uint *packed [[buffer(0)]],                                          \
                 device const ushort *scales [[buffer(1)]],                                        \
                 device const ushort *biases [[buffer(2)]],                                        \
                 device const float *a [[buffer(3)]],                                              \
                 device float *out [[buffer(4)]],                                                  \
                 device const float *residual [[buffer(5)]],                                       \
                 constant SpGemmParams &p [[buffer(6)]],                                           \
                 uint3 group [[threadgroup_position_in_grid]],                                     \
                 uint lane [[thread_index_in_threadgroup]])                                        \
{                                                                                                  \
    const uint n = group.x;                                                                        \
    const uint r0 = group.y * R;                                                                   \
    if (n >= p.outDim || r0 >= p.rows) return;                                                     \
    float acc[R];                                                                                  \
    for (uint m = 0; m < R; ++m) acc[m] = 0.0f;                                                    \
    const uint words = p.inner >> 2;                                                               \
    const uint wordBase = n * p.strideWords;                                                       \
    const uint groupBase = n * p.groupsPerRow;                                                     \
    for (uint w = lane; w < words; w += SP_LANES) {                                                \
        const uint gi = groupBase + (w >> 4);                                                      \
        const float4 weights = sp_q8(packed[wordBase + w], sp_bf16(scales[gi]), sp_bf16(biases[gi])); \
        const uint k = w << 2;                                                                     \
        for (uint m = 0; m < R; ++m) {                                                             \
            acc[m] += dot(float4(*(device const packed_float4 *)(a + (r0 + m) * p.inStride + k)), weights); \
        }                                                                                          \
    }                                                                                              \
    for (uint m = 0; m < R; ++m) {                                                                 \
        const float total = simd_sum(acc[m]);                                                      \
        if (lane == 0u && r0 + m < p.rows) {                                                       \
            const uint index = (r0 + m) * p.outStride + n;                                         \
            out[index] = p.hasResidual != 0u ? total + residual[index] : total;                    \
        }                                                                                          \
    }                                                                                              \
}

SP_DEFINE_GEMM_Q8(sp_gemm_q8_r1, 1u)
SP_DEFINE_GEMM_Q8(sp_gemm_q8_r2, 2u)
SP_DEFINE_GEMM_Q8(sp_gemm_q8_r4, 4u)
SP_DEFINE_GEMM_Q8(sp_gemm_q8_r8, 8u)

// Row-major weights, an R x T tile of outputs per lane. Threadgroup grid: (outDim / T, ceil(rows / R)).
#define SP_DEFINE_GEMM_TILE_Q8(NAME, R, T)                                                         \
kernel void NAME(device const uint *packed [[buffer(0)]],                                          \
                 device const ushort *scales [[buffer(1)]],                                        \
                 device const ushort *biases [[buffer(2)]],                                        \
                 device const float *a [[buffer(3)]],                                              \
                 device float *out [[buffer(4)]],                                                  \
                 device const float *residual [[buffer(5)]],                                       \
                 constant SpGemmParams &p [[buffer(6)]],                                           \
                 uint3 group [[threadgroup_position_in_grid]],                                     \
                 uint lane [[thread_index_in_threadgroup]])                                        \
{                                                                                                  \
    const uint n0 = group.x * T;                                                                   \
    const uint r0 = group.y * R;                                                                   \
    if (n0 >= p.outDim || r0 >= p.rows) return;                                                    \
    float acc[T][R];                                                                               \
    for (uint t = 0; t < T; ++t) for (uint m = 0; m < R; ++m) acc[t][m] = 0.0f;                    \
    const uint words = p.inner >> 2;                                                               \
    for (uint w = lane; w < words; w += SP_LANES) {                                                \
        float4 weights[T];                                                                         \
        for (uint t = 0; t < T; ++t) {                                                             \
            const uint gi = (n0 + t) * p.groupsPerRow + (w >> 4);                                  \
            weights[t] = sp_q8(packed[(n0 + t) * p.strideWords + w], sp_bf16(scales[gi]), sp_bf16(biases[gi])); \
        }                                                                                          \
        const uint k = w << 2;                                                                     \
        for (uint m = 0; m < R; ++m) {                                                             \
            const float4 activations = float4(*(device const packed_float4 *)(a + (r0 + m) * p.inStride + k)); \
            for (uint t = 0; t < T; ++t) acc[t][m] += dot(activations, weights[t]);                \
        }                                                                                          \
    }                                                                                              \
    for (uint t = 0; t < T; ++t) {                                                                 \
        for (uint m = 0; m < R; ++m) {                                                             \
            const float total = simd_sum(acc[t][m]);                                               \
            if (lane == 0u && r0 + m < p.rows) {                                                   \
                const uint index = (r0 + m) * p.outStride + n0 + t;                                \
                out[index] = p.hasResidual != 0u ? total + residual[index] : total;                \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
}

SP_DEFINE_GEMM_TILE_Q8(sp_gemm_q8_r8t4, 8u, 4u)

// Split-K lane kernels for the tiled layout: one or two rows. A threadgroup covers 32 output
// rows (lane = row) and P simdgroups; simdgroup s sums quant groups [s * G/P, (s+1) * G/P), the
// 32 lanes reading two contiguous kibibytes per quant group. Partials meet in threadgroup
// memory. Threadgroup grid: (outDim / 32, ceil(rows / R)), P * 32 threads per group.
#define SP_DEFINE_GEMM_TILED_SPLIT_Q8(NAME, R, P)                                                  \
kernel void NAME(device const uint *packed [[buffer(0)]],                                          \
                 device const ushort *scales [[buffer(1)]],                                        \
                 device const ushort *biases [[buffer(2)]],                                        \
                 device const float *a [[buffer(3)]],                                              \
                 device float *out [[buffer(4)]],                                                  \
                 device const float *residual [[buffer(5)]],                                       \
                 constant SpGemmParams &p [[buffer(6)]],                                           \
                 uint3 group [[threadgroup_position_in_grid]],                                     \
                 uint sg [[simdgroup_index_in_threadgroup]],                                       \
                 uint lane [[thread_index_in_simdgroup]])                                          \
{                                                                                                  \
    threadgroup float partials[P * R * 32];                                                        \
    const uint n = group.x * 32u + lane;                                                           \
    const uint r0 = group.y * R;                                                                   \
    if (n >= p.outDim || r0 >= p.rows) return;                                                     \
    float acc[R];                                                                                  \
    for (uint m = 0; m < R; ++m) acc[m] = 0.0f;                                                    \
    const uint tileBase = (n >> 7) * p.groupsPerRow * 128u + (n & 127u);                           \
    const uint per = p.groupsPerRow / P;                                                           \
    for (uint g = sg * per; g < (sg + 1u) * per; ++g) {                                            \
        const uint record = tileBase + g * 128u;                                                   \
        const float s = sp_bf16(scales[record]), b = sp_bf16(biases[record]);                      \
        device const packed_uint4 *wp = (device const packed_uint4 *)(packed + record * 16u);      \
        for (uint quad = 0; quad < 4u; ++quad) {                                                   \
            const uint4 words = uint4(wp[quad]);                                                   \
            for (uint i = 0; i < 4u; ++i) {                                                        \
                const float4 weights = sp_q8(words[i], s, b);                                      \
                const uint k = g * 64u + quad * 16u + i * 4u;                                      \
                for (uint m = 0; m < R; ++m) {                                                     \
                    acc[m] += dot(float4(*(device const packed_float4 *)(a + (r0 + m) * p.inStride + k)), weights); \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
    for (uint m = 0; m < R; ++m) partials[(sg * R + m) * 32u + lane] = acc[m];                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                               \
    if (sg != 0u) return;                                                                          \
    for (uint m = 0; m < R; ++m) {                                                                 \
        if (r0 + m >= p.rows) continue;                                                            \
        float total = 0.0f;                                                                        \
        for (uint s = 0; s < P; ++s) total += partials[(s * R + m) * 32u + lane];                  \
        const uint index = (r0 + m) * p.outStride + n;                                             \
        out[index] = p.hasResidual != 0u ? total + residual[index] : total;                        \
    }                                                                                              \
}

SP_DEFINE_GEMM_TILED_SPLIT_Q8(sp_gemm_q8_tiled_r1p8, 1u, 8u)
SP_DEFINE_GEMM_TILED_SPLIT_Q8(sp_gemm_q8_tiled_r2p8, 2u, 8u)

// out = silu(gate(a)) * up(a), both projections in one pass, for one or two rows.
#define SP_DEFINE_MLP_TILED_SPLIT_Q8(NAME, R, P)                                                   \
kernel void NAME(device const uint *gatePacked [[buffer(0)]],                                      \
                 device const ushort *gateScales [[buffer(1)]],                                    \
                 device const ushort *gateBiases [[buffer(2)]],                                    \
                 device const uint *upPacked [[buffer(3)]],                                        \
                 device const ushort *upScales [[buffer(4)]],                                      \
                 device const ushort *upBiases [[buffer(5)]],                                      \
                 device const float *a [[buffer(6)]],                                              \
                 device float *out [[buffer(7)]],                                                  \
                 constant SpGemmParams &p [[buffer(8)]],                                           \
                 uint3 group [[threadgroup_position_in_grid]],                                     \
                 uint sg [[simdgroup_index_in_threadgroup]],                                       \
                 uint lane [[thread_index_in_simdgroup]])                                          \
{                                                                                                  \
    threadgroup float partials[2 * P * R * 32];                                                    \
    const uint n = group.x * 32u + lane;                                                           \
    const uint r0 = group.y * R;                                                                   \
    if (n >= p.outDim || r0 >= p.rows) return;                                                     \
    float accG[R], accU[R];                                                                        \
    for (uint m = 0; m < R; ++m) { accG[m] = 0.0f; accU[m] = 0.0f; }                               \
    const uint tileBase = (n >> 7) * p.groupsPerRow * 128u + (n & 127u);                           \
    const uint per = p.groupsPerRow / P;                                                           \
    for (uint g = sg * per; g < (sg + 1u) * per; ++g) {                                            \
        const uint record = tileBase + g * 128u;                                                   \
        const float gs = sp_bf16(gateScales[record]), gb = sp_bf16(gateBiases[record]);            \
        const float us = sp_bf16(upScales[record]), ub = sp_bf16(upBiases[record]);                \
        device const packed_uint4 *gp = (device const packed_uint4 *)(gatePacked + record * 16u);  \
        device const packed_uint4 *up_ = (device const packed_uint4 *)(upPacked + record * 16u);   \
        for (uint quad = 0; quad < 4u; ++quad) {                                                   \
            const uint4 gateWords = uint4(gp[quad]), upWords = uint4(up_[quad]);                   \
            for (uint i = 0; i < 4u; ++i) {                                                        \
                const float4 gateWeights = sp_q8(gateWords[i], gs, gb);                            \
                const float4 upWeights = sp_q8(upWords[i], us, ub);                                \
                const uint k = g * 64u + quad * 16u + i * 4u;                                      \
                for (uint m = 0; m < R; ++m) {                                                     \
                    const float4 activations = float4(*(device const packed_float4 *)(a + (r0 + m) * p.inStride + k)); \
                    accG[m] += dot(activations, gateWeights);                                      \
                    accU[m] += dot(activations, upWeights);                                        \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
    for (uint m = 0; m < R; ++m) {                                                                 \
        partials[(sg * R + m) * 32u + lane] = accG[m];                                             \
        partials[P * R * 32u + (sg * R + m) * 32u + lane] = accU[m];                               \
    }                                                                                              \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                               \
    if (sg != 0u) return;                                                                          \
    for (uint m = 0; m < R; ++m) {                                                                 \
        if (r0 + m >= p.rows) continue;                                                            \
        float gate = 0.0f, up = 0.0f;                                                              \
        for (uint s = 0; s < P; ++s) {                                                             \
            gate += partials[(s * R + m) * 32u + lane];                                            \
            up += partials[P * R * 32u + (s * R + m) * 32u + lane];                                \
        }                                                                                          \
        out[(r0 + m) * p.outStride + n] = sp_silu(gate) * up;                                      \
    }                                                                                              \
}

SP_DEFINE_MLP_TILED_SPLIT_Q8(sp_mlp_in_q8_tiled_r1p8, 1u, 8u)
SP_DEFINE_MLP_TILED_SPLIT_Q8(sp_mlp_in_q8_tiled_r2p8, 2u, 8u)
