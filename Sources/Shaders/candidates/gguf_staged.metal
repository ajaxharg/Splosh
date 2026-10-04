#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// gguf_staged.metal — a GEMM that dequantises its weights into threadgroup memory first.
//
// Most of what a GGUF file holds cannot go to the accelerator as stored: 5- and 6-bit codes,
// and codes that index a table. Widening them to bytes on disk doubles the weight traffic (see
// gemm_group.metal and its probe). The alternative, which is how Splash runs GGUF, keeps the
// codes at their own width and has each tile's threads turn a block of them into fp16 weights,
// scale and offset applied, in threadgroup memory; the accelerator then multiplies bf16
// activations by that block and accumulates. There is no epilogue, since the scales are
// already in the weights.
//
// These are timing probes for that scheme with the simplest format: 4-bit unsigned codes in
// groups of 32 with a bf16 scale and offset a group (Q4_K with its scales decoded), tiled as
// [tile of 128 rows][group of 32][row in tile][16 bytes], sidecars [tile][group][row].
// `p.groups` is inner / 32. Nothing in the engine dispatches these.

struct SpGsParams {
    uint rows;
    uint outDim;
    uint inner;
    uint groups;
    uint hasResidual;
    uint outStride;
};

inline float sp_gs_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }

// One row's 32 codes of one group, as fp16 weights.
inline void sp_gs_dequant32(device const uchar *codes, float s, float m, threadgroup half *dst)
{
    const uint4 words = uint4(*(device const packed_uint4 *)codes);
#pragma unroll
    for (ushort w = 0; w < 4; ++w) {
        const uint word = words[w];
        const float4 low = float4((uint4(word) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu));
        const float4 high = float4((uint4(word) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu));
        *(threadgroup packed_half4 *)(dst + w * 8) = half4(fma(low, float4(s), float4(m)));
        *(threadgroup packed_half4 *)(dst + w * 8 + 4) = half4(fma(high, float4(s), float4(m)));
    }
}

// Steps of up to M rows: a threadgroup of P simdgroups computes one M x 32 tile. Simdgroup s
// takes quant groups [s * G/P, (s+1) * G/P): for each, its 32 lanes dequantise one row's codes
// apiece into the simdgroup's own 2 KiB stage, and one accelerator matmul (K = 32) accumulates.
// Partials meet in threadgroup memory. Threadgroup grid: (outDim / 32, ceil(rows / M)),
// P * 32 threads per group.
template <ushort M, ushort P>
inline void sp_gs_split(device const uchar *packed, device const ushort *scales, device const ushort *mins,
                        device bfloat *a, device float *out, constant SpGsParams &p, uint3 group,
                        uint sg, uint lane, uint tid, threadgroup half *stage, threadgroup float *partials)
{
    constexpr ushort N = 32, K = 32, StorageN = 128;
    const uint n0 = group.x * N;
    const uint r0 = group.y * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    const uint per = p.groups / P;
    const uint first = sg * per;
    const uint tile = n0 / StorageN, inTile = n0 % StorageN;

    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    threadgroup half *mine = stage + sg * (N * K);
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> staged(mine, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    constexpr auto descriptor = matmul2d_descriptor(M, N, K, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> operation;
    auto a0 = activations.template slice<K, M>(0, 0);
    auto b0 = staged.template slice<K, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;

    const ulong recordBase = ulong(tile) * p.groups * StorageN + inTile + lane;
    for (uint g = first; g < first + per; ++g) {
        const ulong record = recordBase + ulong(g) * StorageN;
        sp_gs_dequant32(packed + record * 16ul, sp_gs_bf16(scales[record]), sp_gs_bf16(mins[record]), mine + lane * K);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto as = activations.template slice<K, M>(int(g * K), 0);
        operation.run(as, b0, accumulated);
        // The stage is single: the next group's weights must not land while this run reads it.
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        if (!accumulated.is_valid_element(i)) continue;
        const auto index = accumulated.get_multidimensional_index(i);
        partials[(sg * M + uint(index[1])) * N + uint(index[0])] = accumulated[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint quad = tid; quad < uint(M) * N / 4u; quad += uint(P) * 32u) {
        const uint m = quad / (N / 4u), n = (quad % (N / 4u)) * 4u;
        if (r0 + m >= p.rows) continue;
        float4 total = float4(0.0f);
        for (uint s = 0; s < P; ++s) total += float4(*(threadgroup packed_float4 *)(partials + (s * M + m) * N + n));
        *(device packed_float4 *)(out + (r0 + m) * p.outStride + n0 + n) = total;
    }
}

#define SP_DEFINE_GS_SPLIT(NAME, M, P)                                                            \
kernel void NAME(device const uchar *packed [[buffer(0)]],                                        \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *mins [[buffer(2)]],                                         \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 constant SpGsParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                      \
                 uint lane [[thread_index_in_simdgroup]],                                         \
                 uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                                 \
    threadgroup half stage[P * 32 * 32];                                                          \
    threadgroup float partials[P * M * 32];                                                       \
    sp_gs_split<M, P>(packed, scales, mins, a, out, p, group, sg, lane, tid, stage, partials);    \
}
SP_DEFINE_GS_SPLIT(sp_gs_split_m16p2, 16, 2)
SP_DEFINE_GS_SPLIT(sp_gs_split_m16p4, 16, 4)
SP_DEFINE_GS_SPLIT(sp_gs_split_m16p8, 16, 8)

// Wide steps: a threadgroup of four simdgroups takes 128 rows against 64 output columns, so a
// block of weights is dequantised once for all of them. Each matmul covers two quant groups
// (K = 64): the 128 threads dequantise one (row, group) apiece into a shared 8 KiB stage, and
// each simdgroup then runs its own 32 rows against it. Whole 32-row tiles only: a simdgroup
// with any row in the step stores all 32.
// Threadgroup grid: (outDim / 64, ceil(rows / 128)), 128 threads per group.
kernel void sp_gs_wide(device const uchar *packed [[buffer(0)]],
                       device const ushort *scales [[buffer(1)]],
                       device const ushort *mins [[buffer(2)]],
                       device bfloat *a [[buffer(3)]],
                       device float *out [[buffer(4)]],
                       constant SpGsParams &p [[buffer(7)]],
                       uint3 group [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint tid [[thread_index_in_threadgroup]])
{
    constexpr ushort M = 32, N = 64, K = 64, StorageN = 128;
    threadgroup half stage[N * K];
    const uint n0 = group.x * N;
    if (n0 >= p.outDim || group.y * 4u * M >= p.rows) return;
    const uint r0 = (group.y * 4u + sg) * M;
    // A simdgroup past the last row still dequantises its share of the stage.
    const bool live = r0 < p.rows;
    const uint tile = n0 / StorageN, inTile = n0 % StorageN;

    auto activations = tensor(a + ulong(live ? r0 : 0u) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> staged(stage, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    constexpr auto descriptor = matmul2d_descriptor(M, N, K, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> operation;
    auto a0 = activations.template slice<K, M>(0, 0);
    auto b0 = staged.template slice<K, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;

    const uint column = tid & 63u, part = tid >> 6;
    for (uint pair = 0; pair < p.groups / 2u; ++pair) {
        const ulong record = (ulong(tile) * p.groups + (pair * 2u + part)) * StorageN + inTile + column;
        sp_gs_dequant32(packed + record * 16ul, sp_gs_bf16(scales[record]), sp_gs_bf16(mins[record]),
                        stage + column * K + part * 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            auto as = activations.template slice<K, M>(int(pair * K), 0);
            operation.run(as, b0, accumulated);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (live) {
        auto destination = tensor(out + ulong(r0) * p.outStride, dextents<int, 2>{int(p.outStride), M},
                                  array<int, 2>{1, int(p.outStride)});
        accumulated.store(destination.template slice<N, M>(int(n0), 0));
    }
}
