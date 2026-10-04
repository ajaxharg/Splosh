#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// attn_scan_r.metal — the attention scan with the softmax laid out by row.
//
// The same contract as spi_scan in engine_na.metal (bindings, grid, partial records), so the
// engine and sp_attn_merge use it unchanged: SPLOSH_ATTN_SHAPE=r48c64s8.
//
// What differs is the work between the two matmuls. spi_scan leaves the scores in the score
// product's cooperative tensor, where each thread holds sixteen elements scattered over four
// queries, and has every thread turn its own elements into probabilities: a scalar load of
// each token's key scale and value scale, a scalar store of each probability, a validity test
// per element, and the row totals gathered at the end through threadgroup memory. Here the
// score tensor is stored to threadgroup memory in one operation and the softmax is done by
// threads laid out for it: four lanes to a query, each with a run of consecutive tokens, so
// scores, scales and probabilities move as vectors of four and a query's maximum and total
// come from the three other lanes by shuffles and stay in registers.
//
// The reference of a row's exponentials is the row's maximum where it was last fixed: at the
// span's first chunk, and again whenever a chunk's maximum passes it by more than SPR_MARGIN.
// Then what the row has accumulated is scaled down, which every thread must do for the value
// tensor; so the reference follows the maximum only in steps, and that is rare.

struct SpaScanParams {
    uint blocks;
    uint heads;
    uint kvHeads;
    uint maxPages;
    uint spans;
    uint pagesPerSpan;
    uint rowCap;
    float scale;
    uint aliasTokens;
    uint blocksAcross;
};

#define SPR_MARGIN 30.0f

// M fused queries per block (6 per row), C tokens per chunk, SGQ simdgroups on the score
// product and SGV on the value product (the threadgroup has the larger of the two).
// Threadgroup grid: (kvHeads * spans, blocks), 32 * max(SGQ, SGV) threads per group.
template <ushort M, ushort C, ushort SGQ, ushort SGV = 8>
inline void spr_scan(device half *qB, device const uint *blocks, device const uint *rowSlot,
                     device const uint *rowPos, device const uint *pageTable,
                     device int8_t *kCodes, device const float *kScale,
                     device int8_t *vCodes, device const float *vScale,
                     device float *partials, constant SpaScanParams &p, uint3 group, uint tid,
                     threadgroup float *scores, threadgroup bfloat *weighted,
                     threadgroup float *shift, threadgroup uint *raised)
{
    // The products' tile: the queries padded to what the accelerator takes (a multiple of 16).
    // Padding queries are read from whatever follows the block's and their results dropped.
    constexpr ushort MT = (M + 15) / 16 * 16;
    constexpr uint LANES = 4;                   // softmax threads to a query
    constexpr uint RUN = C / LANES;             // consecutive tokens a lane takes
    constexpr uint QUADS = RUN / 4;
    // A query's lanes are an aligned four, so the shuffles stay inside it; a last simdgroup the
    // softmax threads only half fill is fine.
    static_assert(uint(M) * LANES <= 32u * (SGQ > SGV ? SGQ : SGV), "a softmax thread for each lane of each query");

    const uint kvHead = group.x % p.kvHeads;
    const uint span = p.blocksAcross != 0u ? group.y : group.x / p.kvHeads;
    const uint block = p.blocksAcross != 0u ? group.x / p.kvHeads : group.y;
    if (span >= p.spans || block >= p.blocks) return;
    const uint r0 = blocks[block * 2u], count = blocks[block * 2u + 1u];
    const uint perKV = p.heads / p.kvHeads;
    const uint firstPos = rowPos[r0];
    const uint maxVisible = firstPos + count;
    const uint pages = (maxVisible + 255u) / 256u;
    const uint firstPage = span * p.pagesPerSpan;
    const uint lastPage = min(pages, firstPage + p.pagesPerSpan);
    const uint tableBase = rowSlot[r0] * p.maxPages;
    const uint queryBase = (kvHead * p.rowCap + r0) * perKV;

    constexpr auto scoreDescriptor = matmul2d_descriptor(MT, C, 256, false, true, false);
    matmul2d<scoreDescriptor, execution_simdgroups<SGQ>> scoreOp;
    constexpr auto valueDescriptor = matmul2d_descriptor(MT, 256, C, false, false, false,
                                                         matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<valueDescriptor, execution_simdgroups<SGV>> valueOp;
    const bool scoring = tid / 32u < SGQ;

    auto queries = tensor(qB + ulong(queryBase) * 256u, dextents<int, 2>{256, MT}, array<int, 2>{1, 256});
    auto q0 = queries.template slice<256, MT>(0, 0);
    auto kProto = tensor(kCodes, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
    auto k0 = kProto.template slice<256, C>(0, 0);
    auto score = scoreOp.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    auto sProto = tensor(scores, dextents<int, 2>{C, MT}, array<int, 2>{1, C});
    auto s0 = sProto.template slice<C, MT>(0, 0);
    auto wProto = tensor(weighted, dextents<int, 2>{C, MT}, array<int, 2>{1, C});
    auto w0 = wProto.template slice<C, MT>(0, 0);
    auto vProto = tensor(vCodes, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
    auto v0 = vProto.template slice<256, C>(0, 0);
    auto running = valueOp.template get_destination_cooperative_tensor<decltype(w0), decltype(v0), float>();
#pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0.0f;

    // The softmax threads: query m, lane, and the lane's first token within a chunk.
    const bool lanes = tid < uint(M) * LANES;
    const uint m = tid / LANES, column = (tid % LANES) * RUN;
    const uint rowInBlock = m / perKV;
    // The row sees tokens below this (up to and including its own position); none if it is
    // past the block's rows.
    const uint limit = lanes && rowInBlock < count ? firstPos + rowInBlock + 1u : 0u;
    float reference = -INFINITY, total = 0.0f;  // the same in a query's four lanes

    if (tid < 2u) raised[tid] = 0u;
    uint parity = 0u;
    const uint chunksPerPage = 256u / C;
    for (uint pg = firstPage; pg < lastPage; ++pg) {
        const uint pageTokens = (pageTable[tableBase + pg] * p.kvHeads + kvHead) * 256u;
        for (uint chunk = 0; chunk < chunksPerPage; ++chunk) {
            const uint t0 = pg * 256u + chunk * C;
            if (t0 >= maxVisible) break;
            const uint tokenBase = p.aliasTokens != 0u ? kvHead * 256u + (t0 % p.aliasTokens) * p.kvHeads
                                                       : pageTokens + chunk * C;
            if (scoring) {
                auto kt = tensor(kCodes + ulong(tokenBase) * 256u, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
                auto ks = kt.template slice<256, C>(0, 0);
                scoreOp.run(q0, ks, score);
                score.store(s0);
            }
            // Before this barrier every thread has finished the last chunk: its value product
            // has read the probabilities and it has read that chunk's flag.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0u) raised[parity ^ 1u] = 0u;
            if (lanes) {
                threadgroup const float4 *s4 = reinterpret_cast<threadgroup const float4 *>(scores + m * C + column);
                device const packed_float4 *k4 = reinterpret_cast<device const packed_float4 *>(kScale + tokenBase + column);
                device const packed_float4 *v4 = reinterpret_cast<device const packed_float4 *>(vScale + tokenBase + column);
                threadgroup bfloat4 *w4 = reinterpret_cast<threadgroup bfloat4 *>(weighted + m * C + column);
                const uint token = t0 + column;
                float4 s[QUADS];
                bool4 seen[QUADS];
                float4 top4 = float4(-INFINITY);
#pragma unroll
                for (uint q = 0; q < QUADS; ++q) {
                    seen[q] = (uint4(token + q * 4u) + uint4(0u, 1u, 2u, 3u)) < uint4(limit);
                    s[q] = select(float4(-INFINITY), s4[q] * float4(k4[q]) * p.scale, seen[q]);
                    top4 = max(top4, s[q]);
                }
                float top = max(max(top4.x, top4.y), max(top4.z, top4.w));
                top = max(top, simd_shuffle_xor(top, ushort(1)));
                top = max(top, simd_shuffle_xor(top, ushort(2)));
                // The reference moves only when the chunk's maximum is well past it (or is the first).
                const float next = top > reference + SPR_MARGIN ? top : reference;
                float4 sum4 = float4(0.0f);
#pragma unroll
                for (uint q = 0; q < QUADS; ++q) {
                    const float4 probability = select(float4(0.0f), exp(s[q] - next), seen[q]);
                    sum4 += probability;
                    // An unseen token's stored scale may be anything: its product is not used.
                    w4[q] = bfloat4(select(float4(0.0f), probability * float4(v4[q]), seen[q]));
                }
                float sum = (sum4.x + sum4.y) + (sum4.z + sum4.w);
                sum += simd_shuffle_xor(sum, ushort(1));
                sum += simd_shuffle_xor(sum, ushort(2));
                const float factor = next == reference || reference == -INFINITY ? 1.0f : exp(reference - next);
                total = total * factor + sum;
                reference = next;
                if (column == 0u) {
                    shift[m] = factor;
                    if (factor != 1.0f) raised[parity] = 1u;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (raised[parity] != 0u) {
                // Rare: some query's reference rose. Scale down what it has accumulated.
#pragma unroll
                for (ushort i = 0; i < running.get_capacity(); ++i) {
                    if (!running.is_valid_element(i)) continue;
                    running[i] *= shift[uint(running.get_multidimensional_index(i)[1])];
                }
            }
            auto vt = tensor(vCodes + ulong(tokenBase) * 256u, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
            auto vs = vt.template slice<256, C>(0, 0);
            valueOp.run(w0, vs, running);
            parity ^= 1u;
        }
    }

#pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!running.is_valid_element(i)) continue;
        const auto index = running.get_multidimensional_index(i);
        const uint mm = uint(index[1]), d = uint(index[0]);
        if (mm / perKV >= count) continue;
        const uint row = r0 + mm / perKV, head = kvHead * perKV + mm % perKV;
        partials[((row * p.heads + head) * p.spans + span) * 258u + 2u + d] = running[i];
    }
    if (lanes && column == 0u && rowInBlock < count) {
        const uint row = r0 + rowInBlock, head = kvHead * perKV + m % perKV;
        device float *partial = partials + ((row * p.heads + head) * p.spans + span) * 258u;
        partial[0] = reference;
        partial[1] = total;
    }
}

#define SP_DEFINE_ATTN_SCAN_R(NAME, M, C, SGQ) SP_DEFINE_ATTN_SCAN_RV(NAME, M, C, SGQ, 8)
#define SP_DEFINE_ATTN_SCAN_RV(NAME, M, C, SGQ, SGV)                                              \
kernel void NAME(device half *qB [[buffer(0)]],                                                   \
                 device const uint *blocks [[buffer(2)]],                                         \
                 device const uint *rowSlot [[buffer(3)]],                                        \
                 device const uint *rowPos [[buffer(4)]],                                         \
                 device const uint *pageTable [[buffer(5)]],                                      \
                 device int8_t *kCodes [[buffer(6)]],                                             \
                 device const float *kScale [[buffer(7)]],                                        \
                 device int8_t *vCodes [[buffer(8)]],                                             \
                 device const float *vScale [[buffer(9)]],                                        \
                 device float *partials [[buffer(10)]],                                           \
                 constant SpaScanParams &p [[buffer(11)]],                                        \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                                 \
    threadgroup float scores[(M + 15) / 16 * 16 * C];                                             \
    threadgroup bfloat weighted[(M + 15) / 16 * 16 * C];                                          \
    threadgroup float shift[(M + 15) / 16 * 16];                                                  \
    threadgroup uint raised[2];                                                                   \
    spr_scan<M, C, SGQ, SGV>(qB, blocks, rowSlot, rowPos, pageTable, kCodes, kScale, vCodes,      \
                             vScale, partials, p, group, tid, scores, weighted, shift, raised);   \
}
SP_DEFINE_ATTN_SCAN_R(sp_attn_scan_i8_r48c64s8, 48, 64, 8)
// Ten rows a block: 60 of the 64 query rows the accelerator pads to are real, against 48, with
// the value product's 64 x 256 outputs over sixteen simdgroups (32 x 32 each). Correct, and
// slower (2026-10-03, 40K tokens, 128 rows: 11.0 and 11.2 ms against 9.9 for r48c64s8).
SP_DEFINE_ATTN_SCAN_RV(sp_attn_scan_i8_r60c64s8v16, 60, 64, 8, 16)
SP_DEFINE_ATTN_SCAN_RV(sp_attn_scan_i8_r60c64s16v16, 60, 64, 16, 16)
// The value split alone, at the shipped tile: 11.3 ms.
SP_DEFINE_ATTN_SCAN_RV(sp_attn_scan_i8_r48c64s8v16, 48, 64, 8, 16)
SP_DEFINE_ATTN_SCAN_R(sp_attn_scan_i8_r48c64s8q4, 48, 64, 4)
SP_DEFINE_ATTN_SCAN_R(sp_attn_scan_i8_r48c32s8, 48, 32, 8)
SP_DEFINE_ATTN_SCAN_R(sp_attn_scan_i8_r48c32s8q4, 48, 32, 4)
SP_DEFINE_ATTN_SCAN_R(sp_attn_scan_i8_r48c32s8q2, 48, 32, 2)
