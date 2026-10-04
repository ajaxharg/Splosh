#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// gemm_group.metal — accelerator GEMMs over quant groups of K columns, K not fixed at 64.
//
// The MLX packs quantise in groups of 64 columns; llama.cpp's GGUF formats use groups of 32
// (Q4_K, Q5_K, Q4_0, Q4_1, Q8_0, and IQ4 once its table is applied) or 16 (Q6_K, Q3_K), with
// unsigned codes and a scale and offset per group, or signed codes and a scale alone. These
// are the two shipped tile shapes with K, the code type and the form of the epilogue as
// template parameters: the 32 x 128 whole tile on four simdgroups (wide steps, see sp_na_wide
// in engine_na.metal) and the 16 x 64 split-K tile (steps of up to 16 rows, see
// sp_na_split_tiled).
//
// Weights are tiled as in engine_na.metal with a group of K codes as the element:
// [tile of 128 rows][quant group][row in tile][K codes]; scales and biases are bf16,
// [tile][group][row]; activation sums are per K columns, [row][group]. `p.groups` is inner / K.
//
// Candidates: nothing in the engine dispatches these yet. They exist to time a group size and
// a code type against the shipped q4 kernels before a GGUF path is built on them.

inline float4 sp_gk_bf16x4(device const ushort *p) {
    return as_type<float4>(uint4(ushort4(*(device const packed_ushort4 *)p)) << 16);
}

struct SpGkParams {
    uint rows;
    uint outDim;
    uint inner;
    uint groups;
    uint hasResidual;
    uint outStride;
};

// Whole 32-row tiles. Threadgroup grid: (row tiles in a block of four, outDim / 128, row
// blocks), 128 threads per group.
template <ushort K, class Code, ushort Bits, bool Affine>
inline void sp_gk_wide(device uchar *packed, device bfloat *scales, device bfloat *biases,
                       device bfloat *a, device float *out, device const float *sums,
                       constant SpGkParams &p, uint3 group)
{
    constexpr ushort M = 32, N = 128;
    constexpr uint GroupBytes = uint(K) * Bits / 8u;
    const uint tile = group.y;
    const uint n0 = tile * N;
    const uint r0 = (group.z * 4u + group.x) * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    constexpr auto descriptor = matmul2d_descriptor(M, N, K, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto a0 = activations.template slice<K, M>(0, 0);
    device uchar *tileWeights = packed + ulong(tile) * p.groups * (N * GroupBytes);
    using Weights = tensor<device Code, dextents<int, 2>, tensor_inline>;
    using Handle = typename Weights::data_handle_type;   // the bytes, as the code type's pointer
    Weights first((Handle)tileWeights, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    auto b0 = first.template slice<K, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
    for (uint g = 0; g < p.groups; ++g) {
        auto as = activations.template slice<K, M>(int(g * K), 0);
        Weights weights((Handle)(tileWeights + ulong(g) * (N * GroupBytes)), dextents<int, 2>{K, N}, array<int, 2>{1, K});
        auto bs = weights.template slice<K, N>(0, 0);
        auto partial = operation.template get_destination_cooperative_tensor<decltype(as), decltype(bs), float>();
        operation.run(as, bs, partial);
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            const auto index = accumulated.get_multidimensional_index(i);
            const ulong parameter = (ulong(tile) * p.groups + g) * N + index[0];
            if constexpr (Affine) {
                accumulated[i] += partial[i] * float(scales[parameter])
                    + sums[(r0 + uint(index[1])) * p.groups + g] * float(biases[parameter]);
            } else {
                accumulated[i] += partial[i] * float(scales[parameter]);
            }
        }
    }
    auto destination = tensor(out + ulong(r0) * p.outStride, dextents<int, 2>{int(p.outStride), M},
                              array<int, 2>{1, int(p.outStride)});
    accumulated.store(destination.template slice<N, M>(int(n0), 0));
}

// Split-K: a threadgroup of four simdgroups computes one 16 x 64 tile, each simdgroup running
// its own matmuls over a quarter of the quant groups. Threadgroup grid: (outDim / 64,
// ceil(rows / 16)), 128 threads per group.
template <ushort K, class Code, ushort Bits, bool Affine>
inline void sp_gk_split(device uchar *packed, device const ushort *scales, device const ushort *biases,
                        device bfloat *a, device float *out, device const float *sums,
                        constant SpGkParams &p, uint3 group, uint sg, uint tid, threadgroup float *partials)
{
    constexpr ushort M = 16, N = 64, P = 4, StorageN = 128;
    constexpr uint GroupBytes = uint(K) * Bits / 8u;
    const uint n0 = group.x * N;
    const uint r0 = group.y * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    const uint per = p.groups / P;
    const uint first = sg * per;
    const uint tile = n0 / StorageN, inTile = n0 % StorageN;

    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    device uchar *tileWeights = packed + (ulong(tile) * p.groups * StorageN + inTile) * GroupBytes;
    const uint parameterBase = tile * p.groups * StorageN + inTile;
    constexpr auto descriptor = matmul2d_descriptor(M, N, K, false, true, false);
    matmul2d<descriptor, execution_simdgroup> operation;
    auto a0 = activations.template slice<K, M>(0, 0);
    using Weights = tensor<device Code, dextents<int, 2>, tensor_inline>;
    using Handle = typename Weights::data_handle_type;
    Weights firstWeights((Handle)tileWeights, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    auto b0 = firstWeights.template slice<K, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;

    // The regular block of the shipped kernel (columns c0 ..< c0 + 4 and c0 + 32 ..< c0 + 36,
    // rows rb and rb + 8), checked per thread; any other partition takes the scalar epilogue.
    const auto origin = accumulated.get_multidimensional_index(ushort(0));
    const uint c0 = uint(origin[0]), rb = uint(origin[1]);
    bool regular = accumulated.get_capacity() == M;
#pragma unroll
    for (ushort i = 0; i < M; ++i) {
        const auto index = accumulated.get_multidimensional_index(regular ? i : ushort(0));
        const ushort block = i >> 2;
        regular = regular && accumulated.is_valid_element(i)
            && uint(index[0]) == c0 + (i & 3) + ((block >> 1) & 1) * (N / 2u)
            && uint(index[1]) == rb + (block & 1) * 8u;
    }
    for (uint g = first; g < first + per; ++g) {
        auto as = activations.template slice<K, M>(int(g * K), 0);
        Weights weights((Handle)(tileWeights + ulong(g) * (StorageN * GroupBytes)), dextents<int, 2>{K, N}, array<int, 2>{1, K});
        auto bs = weights.template slice<K, N>(0, 0);
        auto partial = operation.template get_destination_cooperative_tensor<decltype(as), decltype(bs), float>();
        operation.run(as, bs, partial);
        if (regular) {
            const uint pb = parameterBase + g * StorageN + c0;
            const float4 scaleA = sp_gk_bf16x4(scales + pb), scaleB = sp_gk_bf16x4(scales + pb + N / 2u);
            if constexpr (Affine) {
                const float4 biasA = sp_gk_bf16x4(biases + pb), biasB = sp_gk_bf16x4(biases + pb + N / 2u);
                device const float *rowSums = sums + (r0 + rb) * p.groups + g;
                const float2 sum = float2(rowSums[0], rowSums[8u * p.groups]);
#pragma unroll
                for (ushort block = 0; block < 4; ++block) {
                    const float4 scale = (block >> 1) != 0 ? scaleB : scaleA;
                    const float4 bias = (block >> 1) != 0 ? biasB : biasA;
#pragma unroll
                    for (ushort e = 0; e < 4; ++e) {
                        accumulated[block * 4 + e] += partial[block * 4 + e] * scale[e] + sum[block & 1] * bias[e];
                    }
                }
            } else {
#pragma unroll
                for (ushort block = 0; block < 4; ++block) {
                    const float4 scale = (block >> 1) != 0 ? scaleB : scaleA;
#pragma unroll
                    for (ushort e = 0; e < 4; ++e) accumulated[block * 4 + e] += partial[block * 4 + e] * scale[e];
                }
            }
            continue;
        }
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (!accumulated.is_valid_element(i)) continue;
            const auto index = accumulated.get_multidimensional_index(i);
            const uint parameter = parameterBase + g * StorageN + uint(index[0]);
            const float scale = as_type<float>(uint(scales[parameter]) << 16);
            if constexpr (Affine) {
                accumulated[i] += partial[i] * scale
                    + sums[(r0 + uint(index[1])) * p.groups + g] * as_type<float>(uint(biases[parameter]) << 16);
            } else {
                accumulated[i] += partial[i] * scale;
            }
        }
    }
    if (regular) {
#pragma unroll
        for (ushort block = 0; block < 4; ++block) {
            const uint row = rb + (block & 1) * 8u;
            const uint column = c0 + (block >> 1) * (N / 2u);
            *(threadgroup packed_float4 *)(partials + (sg * M + row) * N + column) =
                float4(accumulated[block * 4], accumulated[block * 4 + 1], accumulated[block * 4 + 2], accumulated[block * 4 + 3]);
        }
    } else {
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (!accumulated.is_valid_element(i)) continue;
            const auto index = accumulated.get_multidimensional_index(i);
            partials[(sg * M + uint(index[1])) * N + uint(index[0])] = accumulated[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Four columns per thread per pass.
    for (uint quad = tid; quad < uint(M) * N / 4u; quad += uint(P) * 32u) {
        const uint m = quad / (N / 4u), n = (quad % (N / 4u)) * 4u;
        if (r0 + m >= p.rows) continue;
        float4 total = float4(0.0f);
        for (uint s = 0; s < P; ++s) total += float4(*(threadgroup packed_float4 *)(partials + (s * M + m) * N + n));
        *(device packed_float4 *)(out + (r0 + m) * p.outStride + n0 + n) = total;
    }
}

#define SP_DEFINE_GK(SUFFIX, K, CODE, BITS, AFFINE)                                               \
kernel void sp_gk_wide_##SUFFIX(device uchar *packed [[buffer(0)]],                               \
                                device bfloat *scales [[buffer(1)]],                              \
                                device bfloat *biases [[buffer(2)]],                              \
                                device bfloat *a [[buffer(3)]],                                   \
                                device float *out [[buffer(4)]],                                  \
                                device const float *sums [[buffer(6)]],                           \
                                constant SpGkParams &p [[buffer(7)]],                             \
                                uint3 group [[threadgroup_position_in_grid]])                     \
{                                                                                                 \
    sp_gk_wide<K, CODE, BITS, AFFINE>(packed, scales, biases, a, out, sums, p, group);            \
}                                                                                                 \
kernel void sp_gk_split_##SUFFIX(device uchar *packed [[buffer(0)]],                              \
                                 device const ushort *scales [[buffer(1)]],                       \
                                 device const ushort *biases [[buffer(2)]],                       \
                                 device bfloat *a [[buffer(3)]],                                  \
                                 device float *out [[buffer(4)]],                                 \
                                 device const float *sums [[buffer(6)]],                          \
                                 constant SpGkParams &p [[buffer(7)]],                            \
                                 uint3 group [[threadgroup_position_in_grid]],                    \
                                 uint sg [[simdgroup_index_in_threadgroup]],                      \
                                 uint tid [[thread_index_in_threadgroup]])                        \
{                                                                                                 \
    threadgroup float partials[4 * 16 * 64];                                                      \
    sp_gk_split<K, CODE, BITS, AFFINE>(packed, scales, biases, a, out, sums, p, group, sg, tid, partials); \
}

// Suffix: code type, then the group size. u4/u8: unsigned codes with a scale and an offset a
// group; i4/i8: signed codes with a scale alone.
SP_DEFINE_GK(u4k64, 64, uint4b_format, 4, true)    // the MLX 4-bit pack, for a like-for-like baseline
SP_DEFINE_GK(u8k64, 64, uchar, 8, true)            // the MLX 8-bit pack
SP_DEFINE_GK(u4k32, 32, uint4b_format, 4, true)    // Q4_K, Q4_1
SP_DEFINE_GK(i4k32, 32, int4b_format, 4, false)    // Q4_0
SP_DEFINE_GK(u8k32, 32, uchar, 8, true)            // Q5_K
SP_DEFINE_GK(i8k32, 32, int8_t, 8, false)            // Q8_0; IQ4_XS and IQ4_NL with the table applied
// A sub-byte code type needs K to be a multiple of 32 (a static_assert in the matmul), so a
// format with groups of 16 is 8-bit here whatever its own code width: Q6_K, and Q3_K too.
SP_DEFINE_GK(i8k16, 16, int8_t, 8, false)
