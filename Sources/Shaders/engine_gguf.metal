// The decoders' float operations keep their source order: a weight is d * scale * code as the
// reference makes it, and Metal's fast math would otherwise be free to regroup the product.
#pragma clang fp reassociate(off)

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "gguf_formats.h"

using namespace metal;
using namespace mpp::tensor_ops;

// engine_gguf.metal — GEMM on the accelerator from llama.cpp GGUF weights at their native size.
//
// The accelerator takes raw 4-bit and 8-bit codes (engine_na.metal), which is all an MLX pack
// holds. A GGUF file holds 3-, 5- and 6-bit codes and codes that index a table, and widening
// them on disk costs their traffic twice over. So the codes stay as they are stored, in the
// planes of gguf_formats.h, and each tile's threads turn a block of them into fp16 weights in
// threadgroup memory, scale and offset applied; `matmul2d` then multiplies bf16 activations by
// that block and accumulates. There is no epilogue and no sidecar of activation sums: the
// scales are already in the weights. This is how Splash and Splish run GGUF (Apache-2.0:
// kernels/common/gguf_staged_tile.h and kernels/shared/gguf_linear.metal), in the two tile
// shapes candidates/gguf_staged.metal timed here.
//
// Each format is its own instantiation of each shape; no kernel chooses a format at run time.
// `p.groups` is inner / 32. A kernel reads whole tiles of activation rows (M for the split
// kernels, 32 for the wide one), so the activation buffer holds them; rows at or past `p.rows`
// are computed and never written.
//
// Three more kernels a format stand beside the tiles: each tile shape again with its activation
// columns in the engine's head order (`heads`, for the one weight whose columns are in the
// file's), a GEMM for a weight of fewer rows than a tile (sp_gguf_small), and the embedding's
// row gather (sp_gguf_embed). The last two decode to fp32 and stage nothing.
//
// A residual is not added at the end but loaded into the accumulator at the start, by the
// accelerator's own load, and the tile is stored by its store, both through tensors whose
// extent is the step's rows: that is what keeps a partial tile's rows past the step unread and
// unwritten. Doing either by hand, an element at a time, costs the wide kernel 15% at whole
// tiles with no residual to add; what a kernel carries decides how many of its threadgroups run
// at once (engine_na.metal, sp_na_wide).
//
// Bindings: plane0 at 0, plane1 at 1 (unread by the formats without one), meta at 2, bf16
// activations at 3, fp32 out at 4, the residual at 5 (read only; the tensor that loads it cannot
// be of a const type), parameters at 7.

struct SpGgufParams {
    uint rows;
    uint outDim;
    uint inner;
    uint groups;
    uint hasResidual;
    uint outStride;
};

// The out projection of a gated-delta layer keeps its columns in the file's order of the 48
// value heads, which is not the engine's (SploshModel/GgufNames.swift): the file's head h is the
// engine's head 3 (h % 16) + h / 16. A head is 128 columns, four groups, and inner is 6144.
// Permuting columns would re-quantise every block, so the weight stays as it is and the `heads`
// kernels read the activations of group g from where the engine put its head. The order is a
// template argument, not a parameter: the kernels without it are compiled as if it did not
// exist, and one with it pays a few integer operations a matmul.
inline uint sp_gguf_head_group(uint g)
{
    const uint head = g >> 2;
    return ((3u * (head & 15u) + (head >> 4)) << 2) | (g & 3u);
}

// The first activation column of group g.
template <bool Heads>
inline uint sp_gguf_column(uint g)
{
    if constexpr (Heads) return sp_gguf_head_group(g) * 32u;
    else return g * 32u;
}

// Steps of up to M rows: a threadgroup of two simdgroups computes one M x 32 tile. Simdgroup 0
// takes the first half of the quant groups and simdgroup 1 the rest: for each group its 32
// lanes dequantise one weight row's codes apiece into the simdgroup's own 2 KiB stage, and one
// accelerator matmul (K = 32) accumulates. Simdgroup 1 leaves its sums in threadgroup memory
// and simdgroup 0 adds them to its own and stores the tile, so the partials are one tile, not
// two. Threadgroup grid: (outDim / 32, ceil(rows / M)), 64 threads per group.
template <class F, ushort M, bool Heads>
inline void sp_gguf_split(device const uchar *plane0, device const uchar *plane1, device const uchar *meta,
                          device bfloat *a, device float *out, device float *residual,
                          constant SpGgufParams &p, uint3 group, uint sg, uint lane,
                          threadgroup half *stage, threadgroup float *partials, threadgroup half2 *table)
{
    constexpr ushort N = 32, K = 32;
    const uint n0 = group.x * N;
    const uint r0 = group.y * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    sp_gguf_pair_table<F>(table, sg * 32u + lane, 64u);
    // An odd count of groups leaves simdgroup 1 the one more.
    const uint first = sg == 0u ? 0u : p.groups / 2u;
    const uint last = sg == 0u ? p.groups / 2u : p.groups;

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
    if (p.hasResidual != 0u && sg == 0u) {
        auto prior = tensor(residual, dextents<int, 2>{int(p.outStride), int(p.rows)}, array<int, 2>{1, int(p.outStride)});
        accumulated.load(prior.slice(int(n0), int(r0)));
    } else {
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
    }

    auto cursor = sp_gguf_seek<F>(sp_gguf_row<F>(plane0, plane1, meta, n0 + lane, p.groups), first);
    for (uint g = first; g < last; ++g) {
        sp_gguf_stage<F>(cursor, g, table, mine + lane * K);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // The next group's records are fetched while the accelerator runs on this one: a
        // quarter of the Q8_0 kernel's time when they are fetched after it.
        if (g + 1u < last) sp_gguf_advance<F>(cursor, g + 1u);
        auto as = activations.template slice<K, M>(int(sp_gguf_column<Heads>(g)), 0);
        operation.run(as, b0, accumulated);
        // The stage is single: the next group's weights must not land while this run reads it.
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg != 0u) {
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (!accumulated.is_valid_element(i)) continue;
            const auto index = accumulated.get_multidimensional_index(i);
            partials[uint(index[1]) * N + uint(index[0])] = accumulated[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0u) return;
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        if (!accumulated.is_valid_element(i)) continue;
        const auto index = accumulated.get_multidimensional_index(i);
        accumulated[i] += partials[uint(index[1]) * N + uint(index[0])];
    }
    auto destination = tensor(out, dextents<int, 2>{int(p.outStride), int(p.rows)}, array<int, 2>{1, int(p.outStride)});
    accumulated.store(destination.slice(int(n0), int(r0)));
}

#define SP_DEFINE_GGUF_SPLIT(NAME, F, M, HEADS)                                                   \
kernel void NAME(device const uchar *plane0 [[buffer(0)]],                                        \
                 device const uchar *plane1 [[buffer(1)]],                                        \
                 device const uchar *meta [[buffer(2)]],                                          \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device float *residual [[buffer(5)]],                                            \
                 constant SpGgufParams &p [[buffer(7)]],                                          \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                      \
                 uint lane [[thread_index_in_simdgroup]])                                         \
{                                                                                                 \
    threadgroup half stage[2 * 32 * 32];                                                          \
    threadgroup float partials[M * 32];                                                           \
    threadgroup half2 table[F::PairEntries];                                                      \
    sp_gguf_split<F, M, HEADS>(plane0, plane1, meta, a, out, residual, p, group, sg, lane, stage, partials, table); \
}

// Wide steps: a threadgroup of four simdgroups takes 128 rows against 64 output columns, so a
// block of weights is dequantised once for all of them. Each matmul covers two quant groups
// (K = 64): the 128 threads dequantise one (weight row, group) apiece into a shared 8 KiB
// stage, and each simdgroup then runs its own 32 rows against it. A simdgroup with no row in
// the step stages its share and runs nothing. `inner` is a multiple of 64. A matmul's two
// groups are of one head, so in head order too their activations are 64 columns together.
// Threadgroup grid: (outDim / 64, ceil(rows / 128)), 128 threads per group.
template <class F, bool Heads>
inline void sp_gguf_wide(device const uchar *plane0, device const uchar *plane1, device const uchar *meta,
                         device bfloat *a, device float *out, device float *residual,
                         constant SpGgufParams &p, uint3 group, uint sg, uint tid,
                         threadgroup half *stage, threadgroup half2 *table)
{
    constexpr ushort M = 32, N = 64, K = 64;
    const uint n0 = group.x * N;
    if (n0 >= p.outDim || group.y * 4u * M >= p.rows) return;
    sp_gguf_pair_table<F>(table, tid, 128u);
    const uint r0 = (group.y * 4u + sg) * M;
    const bool live = r0 < p.rows;

    auto activations = tensor(a + ulong(live ? r0 : 0u) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> staged(stage, dextents<int, 2>{K, N}, array<int, 2>{1, K});
    constexpr auto descriptor = matmul2d_descriptor(M, N, K, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> operation;
    auto a0 = activations.template slice<K, M>(0, 0);
    auto b0 = staged.template slice<K, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    if (p.hasResidual != 0u && live) {
        auto prior = tensor(residual, dextents<int, 2>{int(p.outStride), int(p.rows)}, array<int, 2>{1, int(p.outStride)});
        accumulated.load(prior.slice(int(n0), int(r0)));
    } else {
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
    }

    // A thread's weight row, and which of a matmul's two groups it stages.
    const uint column = tid & 63u, part = tid >> 6;
    auto cursor = sp_gguf_seek<F>(sp_gguf_row<F>(plane0, plane1, meta, n0 + column, p.groups), part);
    for (uint pair = 0; pair < p.groups / 2u; ++pair) {
        const uint g = pair * 2u + part;
        sp_gguf_stage<F>(cursor, g, table, stage + column * K + part * 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The next pair's records are fetched while the accelerator runs on this one.
        if (g + 2u < p.groups) sp_gguf_advance<F>(cursor, g + 2u);
        if (live) {
            auto as = activations.template slice<K, M>(int(sp_gguf_column<Heads>(pair * 2u)), 0);
            operation.run(as, b0, accumulated);
        }
        // The stage is single: the next pair's weights must not land while a run reads it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live) return;
    auto destination = tensor(out, dextents<int, 2>{int(p.outStride), int(p.rows)}, array<int, 2>{1, int(p.outStride)});
    accumulated.store(destination.slice(int(n0), int(r0)));
}

#define SP_DEFINE_GGUF_WIDE(NAME, F, HEADS)                                                       \
kernel void NAME(device const uchar *plane0 [[buffer(0)]],                                        \
                 device const uchar *plane1 [[buffer(1)]],                                        \
                 device const uchar *meta [[buffer(2)]],                                          \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device float *residual [[buffer(5)]],                                            \
                 constant SpGgufParams &p [[buffer(7)]],                                          \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                      \
                 uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                                 \
    threadgroup half stage[64 * 64];                                                              \
    threadgroup half2 table[F::PairEntries];                                                      \
    sp_gguf_wide<F, HEADS>(plane0, plane1, meta, a, out, residual, p, group, sg, tid, stage, table); \
}

// A weight of any number of rows, which the tiles of 32 above cannot take: the 48-row
// projections of a gated-delta layer. One threadgroup of 32 lanes makes one output of one row.
// Lane l takes groups l, l + 32, ... of the weight row, decodes each to fp32 as the embedding
// does, and sums its products with the row's bf16 activations; lane 0 writes the lanes' total,
// with the residual when there is one. Nothing is staged and the accelerator is not used: these
// weights are a hundredth of a layer. Bindings and parameters are the kernels' above, the
// activations read as their bits. Threadgroup grid: (outDim, rows), 32 threads per group.
template <class F>
inline void sp_gguf_small(device const uchar *plane0, device const uchar *plane1, device const uchar *meta,
                          device const ushort *a, device float *out, device const float *residual,
                          constant SpGgufParams &p, uint3 group, uint lane)
{
    const uint n = group.x, r = group.y;
    if (n >= p.outDim || r >= p.rows) return;
    const SpGgufRow row = sp_gguf_row<F>(plane0, plane1, meta, n, p.groups);
    device const ushort *x = a + ulong(r) * p.inner;
    float sum = 0.0f;
    for (uint g = lane; g < p.groups; g += 32u) {
        const typename F::Payload w = sp_gguf_codes<F>(row, g);
        const SpGgufCoef k = sp_gguf_coef<F>(w, sp_gguf_scales<F>(row, g / F::MetaGroups), ushort(g % F::MetaGroups));
#pragma unroll
        for (ushort c = 0; c < 4; ++c) {
            const SpGgufChunk32 weights = sp_gguf_chunk32<F>(F::chunk(w, c), k);
            // bf16 is the top half of an fp32.
            const uint4 low = uint4(*(device const packed_ushort4 *)(x + g * 32u + 4u * c)) << 16;
            const uint4 high = uint4(*(device const packed_ushort4 *)(x + g * 32u + 16u + 4u * c)) << 16;
            sum += dot(weights.low, as_type<float4>(low)) + dot(weights.high, as_type<float4>(high));
        }
    }
    const float total = simd_sum(sum);
    if (lane != 0u) return;
    const ulong index = ulong(r) * p.outStride + n;
    out[index] = p.hasResidual != 0u ? total + residual[index] : total;
}

#define SP_DEFINE_GGUF_SMALL(NAME, F)                                                             \
kernel void NAME(device const uchar *plane0 [[buffer(0)]],                                        \
                 device const uchar *plane1 [[buffer(1)]],                                        \
                 device const uchar *meta [[buffer(2)]],                                          \
                 device const ushort *a [[buffer(3)]],                                            \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 constant SpGgufParams &p [[buffer(7)]],                                          \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint lane [[thread_index_in_threadgroup]])                                       \
{                                                                                                 \
    sp_gguf_small<F>(plane0, plane1, meta, a, out, residual, p, group, lane);                     \
}

// Embedding: gather a row per token and dequantise it to fp32, the reference decoder's values
// exactly (gguf_formats.h, sp_gguf_chunk32). One thread decodes one group of 32 of one row.
// Bindings and parameters are sp_embed_q4's (engine.metal), so the host fills them the same
// way: the planes at 0, 1 and 2, the token ids at 3, fp32 out [rows, hidden] at 4, parameters
// at 5. `groupsPerRow` is hidden / 32 and `strideWords` is not read.
// Thread grid: (groupsPerRow, rows).
struct SpGgufEmbedParams {
    uint rows;
    uint strideWords;
    uint groupsPerRow;
    uint hidden;
};

template <class F>
inline void sp_gguf_embed(device const uchar *plane0, device const uchar *plane1, device const uchar *meta,
                          device const uint *tokens, device float *out,
                          constant SpGgufEmbedParams &p, uint2 gid)
{
    if (gid.x >= p.groupsPerRow || gid.y >= p.rows) return;
    const SpGgufRow row = sp_gguf_row<F>(plane0, plane1, meta, tokens[gid.y], p.groupsPerRow);
    const typename F::Payload w = sp_gguf_codes<F>(row, gid.x);
    const SpGgufCoef k = sp_gguf_coef<F>(w, sp_gguf_scales<F>(row, gid.x / F::MetaGroups), ushort(gid.x % F::MetaGroups));
    device float *dst = out + ulong(gid.y) * p.hidden + gid.x * 32u;
#pragma unroll
    for (ushort c = 0; c < 4; ++c) {
        const SpGgufChunk32 weights = sp_gguf_chunk32<F>(F::chunk(w, c), k);
        *(device packed_float4 *)(dst + 4 * c) = weights.low;
        *(device packed_float4 *)(dst + 16 + 4 * c) = weights.high;
    }
}

#define SP_DEFINE_GGUF_EMBED(NAME, F)                                                             \
kernel void NAME(device const uchar *plane0 [[buffer(0)]],                                        \
                 device const uchar *plane1 [[buffer(1)]],                                        \
                 device const uchar *meta [[buffer(2)]],                                          \
                 device const uint *tokens [[buffer(3)]],                                         \
                 device float *out [[buffer(4)]],                                                 \
                 constant SpGgufEmbedParams &p [[buffer(5)]],                                     \
                 uint2 gid [[thread_position_in_grid]])                                           \
{                                                                                                 \
    sp_gguf_embed<F>(plane0, plane1, meta, tokens, out, p, gid);                                  \
}

#define SP_DEFINE_GGUF(F, SUFFIX)                                                                 \
SP_DEFINE_GGUF_SPLIT(sp_gguf_split_m16_##SUFFIX, F, 16, false)                                    \
SP_DEFINE_GGUF_SPLIT(sp_gguf_split_m32_##SUFFIX, F, 32, false)                                    \
SP_DEFINE_GGUF_WIDE(sp_gguf_wide_##SUFFIX, F, false)                                              \
SP_DEFINE_GGUF_SPLIT(sp_gguf_split_m16_heads_##SUFFIX, F, 16, true)                               \
SP_DEFINE_GGUF_SPLIT(sp_gguf_split_m32_heads_##SUFFIX, F, 32, true)                               \
SP_DEFINE_GGUF_WIDE(sp_gguf_wide_heads_##SUFFIX, F, true)                                         \
SP_DEFINE_GGUF_SMALL(sp_gguf_small_##SUFFIX, F)                                                   \
SP_DEFINE_GGUF_EMBED(sp_gguf_embed_##SUFFIX, F)

SP_DEFINE_GGUF(SpGgufQ4K, q4k)
SP_DEFINE_GGUF(SpGgufQ5K, q5k)
SP_DEFINE_GGUF(SpGgufQ6K, q6k)
SP_DEFINE_GGUF(SpGgufQ3K, q3k)
SP_DEFINE_GGUF(SpGgufQ80, q80)
SP_DEFINE_GGUF(SpGgufIQ4NL, iq4nl)
SP_DEFINE_GGUF(SpGgufIQ4XS, iq4xs)
SP_DEFINE_GGUF(SpGgufIQ3S, iq3s)
