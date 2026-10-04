#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// attn_scan_b.metal — the int8 attention scan with its inner loop laid out by row.
//
// Same contract as spi_scan in engine_na.metal (bindings, SpaScanParams, grid, partial records),
// so the engine dispatches it and sp_attn_merge reads its output unchanged. What differs is the
// work between the two accelerator products of a chunk:
//
//   1. The score product's cooperative tensor goes to threadgroup memory in one store, as an
//      [M][C] fp32 tile. No thread post-processes its own scattered elements.
//   2. The softmax runs on threads laid out by row: four lanes of a simdgroup own one fused
//      query and take C / 4 consecutive tokens each. Scores, key scales and value scales are
//      float4 loads, the row maximum and row sum are two xor shuffles between the four lanes,
//      and the weighted probabilities leave as bf16 vectors. With M = 48 that is 192 threads;
//      the last two simdgroups only take part in the products.
//   3. Each row keeps a running maximum as its reference. When a chunk raises it, the lane
//      publishes exp(old - new) and every thread rescales its part of the accumulated value
//      tensor before the chunk's value product; the row sum is rescaled in the lane's register.
//      The maximum lives in a register of each of the four lanes (they compute the same
//      number) and the sum in the first lane's, so the softmax reads threadgroup memory for
//      the scores only.
//   4. Three barriers a chunk: scores stored, probabilities stored, value product done.
//
// The reference a record reports is whatever the row ended on; sp_attn_merge combines spans by
// their references, so a reference above the true maximum is as good as the maximum itself.
// HEADROOM uses that: a raised reference is set that far above the new maximum, so the rescale
// runs only when a score exceeds everything before it by more than the headroom, not on every
// new maximum. 0 is the plain running maximum.

struct SpaScanParams {
    uint blocks;
    uint heads;
    uint kvHeads;
    uint maxPages;
    uint spans;
    uint pagesPerSpan;
    uint rowCap;
    float scale;
    uint aliasTokens;    // probe: read every chunk from the pool's first this-many tokens (0: off)
    uint blocksAcross;   // 1: the grid is (kvHeads * blocks, spans)
};

// One lane's share of a chunk: V vectors of four consecutive tokens of one fused query.
// Returns the factor the row's earlier accumulation is to be multiplied by (1 when the
// reference did not move); only the row's first lane (`lead`) works it out and keeps the sum.
// `masked` is false when every live query of the block sees every token of the chunk; the two
// forms are separate calls so that the common one carries no selects.
template <ushort V, ushort HEADROOM>
inline float spb_softmax(threadgroup const float4 *scores, threadgroup bfloat4 *weighted,
                         device const packed_float4 *kScales, device const packed_float4 *vScales,
                         float scale, uint token, uint limit, bool masked, bool lead,
                         thread float &rowMax, thread float &rowSum)
{
    float4 s[V];
    bool4 visible[V];
#pragma unroll
    for (ushort v = 0; v < V; ++v) {
        s[v] = scores[v] * float4(kScales[v]) * scale;
        if (masked) {
            visible[v] = (uint4(token + 4u * v) + uint4(0u, 1u, 2u, 3u)) < uint4(limit);
            s[v] = select(float4(-INFINITY), s[v], visible[v]);
        }
    }
    float4 top4 = s[0];
#pragma unroll
    for (ushort v = 1; v < V; ++v) top4 = max(top4, s[v]);
    float top = max(max(top4.x, top4.y), max(top4.z, top4.w));
    top = max(top, simd_shuffle_xor(top, ushort(1)));
    top = max(top, simd_shuffle_xor(top, ushort(2)));
    const float next = top > rowMax ? top + float(HEADROOM) : rowMax;

    float4 total4 = float4(0.0f);
#pragma unroll
    for (ushort v = 0; v < V; ++v) {
        float4 probability = exp(s[v] - next);
        float4 product = probability * float4(vScales[v]);
        if (masked) {
            // A token no row may see contributes exactly nothing, whatever its scales hold
            // (and a row that has seen nothing yet has next = -inf).
            probability = select(float4(0.0f), probability, visible[v]);
            product = select(float4(0.0f), product, visible[v]);
        }
        total4 += probability;
        weighted[v] = bfloat4(product);
    }
    float total = (total4.x + total4.y) + (total4.z + total4.w);
    total += simd_shuffle_xor(total, ushort(1));
    total += simd_shuffle_xor(total, ushort(2));

    float factor = 1.0f;
    if (lead) {
        // Nothing was accumulated under a reference of -inf, so that first rise rescales nothing.
        if (next != rowMax && rowMax != -INFINITY) factor = exp(rowMax - next);
        rowSum = rowSum * factor + total;
    }
    rowMax = next;
    return factor;
}

// M fused queries per block (6 per row), C tokens per chunk, SG simdgroups for both products.
// Threadgroup grid: (kvHeads * spans, blocks), SG * 32 threads per group.
// EXPECT is the number of valid score elements the threadgroup must hold between its threads
// (M * C); a kernel defined with another number poisons its output and exists to test that.
template <ushort M, ushort C, ushort SG, ushort HEADROOM, uint EXPECT>
inline void spb_scan(device half *qB, device const uint *blocks, device const uint *rowSlot,
                     device const uint *rowPos, device const uint *pageTable,
                     device int8_t *kCodes, device const float *kScale,
                     device int8_t *vCodes, device const float *vScale,
                     device float *partials, constant SpaScanParams &p, uint3 group, uint tid,
                     threadgroup float4 *scores4, threadgroup bfloat4 *weighted4,
                     threadgroup float *rowFactor, threadgroup uint *shared)
{
    // Two grids, as in spi_scan: (kvHeads * spans, blocks), or with blocksAcross
    // (kvHeads * blocks, spans).
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

    constexpr ushort LANES = 4;             // lanes of a simdgroup that share a fused query
    constexpr ushort PER_LANE = C / LANES;  // consecutive tokens a lane takes
    constexpr ushort V = PER_LANE / 4;      // as float4 vectors

    constexpr auto scoreDescriptor = matmul2d_descriptor(M, C, 256, false, true, false);
    matmul2d<scoreDescriptor, execution_simdgroups<SG>> scoreOp;
    constexpr auto valueDescriptor = matmul2d_descriptor(M, 256, C, false, false, false,
                                                         matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<valueDescriptor, execution_simdgroups<SG>> valueOp;

    auto queries = tensor(qB + ulong(queryBase) * 256u, dextents<int, 2>{256, M}, array<int, 2>{1, 256});
    auto q0 = queries.template slice<256, M>(0, 0);
    auto kProto = tensor(kCodes, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
    auto k0 = kProto.template slice<256, C>(0, 0);
    auto score = scoreOp.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();

    threadgroup float *scores = reinterpret_cast<threadgroup float *>(scores4);
    threadgroup bfloat *weighted = reinterpret_cast<threadgroup bfloat *>(weighted4);
    auto sProto = tensor(scores, dextents<int, 2>{C, M}, array<int, 2>{1, C});
    auto s0 = sProto.template slice<C, M>(0, 0);
    auto wProto = tensor(weighted, dextents<int, 2>{C, M}, array<int, 2>{1, C});
    auto w0 = wProto.template slice<C, M>(0, 0);
    auto vProto = tensor(vCodes, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
    auto v0 = vProto.template slice<256, C>(0, 0);
    auto running = valueOp.template get_destination_cooperative_tensor<decltype(w0), decltype(v0), float>();
#pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0.0f;

    // The store below writes whatever partition the accelerator chose, so the kernel is not
    // sized for one. What it does depend on is that the threadgroup's valid score elements
    // are the whole M x C tile: counted once here, and a different count poisons the output.
    uint valid = 0u;
#pragma unroll
    for (ushort i = 0; i < score.get_capacity(); ++i) valid += score.is_valid_element(i) ? 1u : 0u;
    valid = simd_sum(valid);
    // The softmax also depends on the four lanes of a fused query being xor neighbours in one
    // simdgroup. A simdgroup where a shuffle does not return the neighbouring thread index
    // reports a count that cannot add up to EXPECT.
    const uint misplaced = simd_sum((simd_shuffle_xor(tid, ushort(1)) != (tid ^ 1u) ||
                                     simd_shuffle_xor(tid, ushort(2)) != (tid ^ 2u)) ? 1u : 0u);
    if (tid % 32u == 0u) shared[1u + tid / 32u] = misplaced != 0u ? EXPECT + 1u : valid;
    if (tid == 0u) shared[0] = 0u;

    // This thread's place in the softmax.
    const bool soft = tid < uint(LANES) * M;
    const uint m = tid / LANES;
    const uint column = (tid % LANES) * PER_LANE;
    const bool lead = soft && column == 0u;
    const uint rowInBlock = m / perKV;
    // A short block's queries beyond its count are computed and never published or written.
    const bool live = soft && rowInBlock < count;
    // Rows see tokens up to and including their own position.
    const uint limit = live ? firstPos + rowInBlock + 1u : 0u;
    threadgroup const float4 *laneScores = scores4 + (m * C + column) / 4u;
    threadgroup bfloat4 *laneWeighted = weighted4 + (m * C + column) / 4u;
    float rowMax = -INFINITY, rowSum = 0.0f;
    // A row's factor is 1 except between the softmax of a chunk that raised its reference and
    // the end of that chunk.
    if (lead) rowFactor[m] = 1.0f;
    bool raised = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool poisoned = false;
    if (lead) {
        uint total = 0u;
#pragma unroll
        for (ushort g = 0; g < SG; ++g) total += shared[1u + g];
        poisoned = total != EXPECT;
    }

    const uint chunksPerPage = 256u / C;
    for (uint pg = firstPage; pg < lastPage; ++pg) {
        const uint pageTokens = (pageTable[tableBase + pg] * p.kvHeads + kvHead) * 256u;
        for (uint chunk = 0; chunk < chunksPerPage; ++chunk) {
            const uint t0 = pg * 256u + chunk * C;
            if (t0 >= maxVisible) break;
            // aliasTokens (SPLOSH_ATTN_ALIAS): the chunk read from the pool's first tokens.
            const uint tokenBase = p.aliasTokens != 0u ? kvHead * 256u + (t0 % p.aliasTokens) * p.kvHeads
                                                       : pageTokens + chunk * C;
            auto kt = tensor(kCodes + ulong(tokenBase) * 256u, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
            auto ks = kt.template slice<256, C>(0, 0);
            scoreOp.run(q0, ks, score);
            score.store(s0);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (soft) {
                device const packed_float4 *kScales = (device const packed_float4 *)(kScale + tokenBase + column);
                device const packed_float4 *vScales = (device const packed_float4 *)(vScale + tokenBase + column);
                // A chunk that ends at or before the block's first position is seen whole by
                // every live row. In a short block the queries beyond the count then take
                // the unmasked form too: their tokens are stored ones, so the numbers are
                // finite, and nothing of theirs is published or written.
                const bool masked = t0 + C > firstPos + 1u;
                float factor;
                if (masked) {
                    factor = spb_softmax<V, HEADROOM>(laneScores, laneWeighted, kScales, vScales, p.scale,
                                                      t0 + column, limit, true, lead, rowMax, rowSum);
                } else {
                    factor = spb_softmax<V, HEADROOM>(laneScores, laneWeighted, kScales, vScales, p.scale,
                                                      t0 + column, limit, false, lead, rowMax, rowSum);
                }
                if (live && factor != 1.0f) {
                    rowFactor[m] = factor;
                    // Every writer writes the same value; thread 0 clears it after the readers.
                    shared[0] = 1u;
                    raised = true;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (shared[0] != 0u) {
#pragma unroll
                for (ushort i = 0; i < running.get_capacity(); ++i) {
                    if (!running.is_valid_element(i)) continue;
                    running[i] *= rowFactor[uint(running.get_multidimensional_index(i)[1])];
                }
            }
            auto vt = tensor(vCodes + ulong(tokenBase) * 256u, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
            auto vs = vt.template slice<256, C>(0, 0);
            valueOp.run(w0, vs, running);
            // Every thread has read the flag, the factors and the probabilities before the
            // next chunk replaces them.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) shared[0] = 0u;
            if (raised) { rowFactor[m] = 1.0f; raised = false; }
        }
    }

#pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!running.is_valid_element(i)) continue;
        const auto index = running.get_multidimensional_index(i);
        const uint em = uint(index[1]), d = uint(index[0]);
        if (em / perKV >= count) continue;
        const uint row = r0 + em / perKV, head = kvHead * perKV + em % perKV;
        partials[((row * p.heads + head) * p.spans + span) * 258u + 2u + d] = running[i];
    }
    if (lead && rowInBlock < count) {
        const uint row = r0 + rowInBlock, head = kvHead * perKV + m % perKV;
        device float *partial = partials + ((row * p.heads + head) * p.spans + span) * 258u;
        // A row that saw no token of the span leaves reference -inf, total 0 and a zero vector,
        // which sp_attn_merge gives no weight.
        partial[0] = poisoned ? NAN : rowMax;
        partial[1] = rowSum;
    }
}

#define SP_DEFINE_ATTN_SCAN_B(NAME, M, C, SG, HEADROOM, EXPECT)                                    \
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
    threadgroup float4 scores[M * C / 4];                                                         \
    threadgroup bfloat4 weighted[M * C / 4];                                                      \
    threadgroup float rowFactor[M];                                                               \
    threadgroup uint shared[1 + SG];                                                              \
    spb_scan<M, C, SG, HEADROOM, EXPECT>(qB, blocks, rowSlot, rowPos, pageTable, kCodes, kScale,  \
                                         vCodes, vScale, partials, p, group, tid, scores,         \
                                         weighted, rowFactor, shared);                            \
}

// The plain running maximum, 64- and 32-token chunks.
SP_DEFINE_ATTN_SCAN_B(sp_attn_scan_i8_b48c64s8, 48, 64, 8, 0, 48 * 64)
SP_DEFINE_ATTN_SCAN_B(sp_attn_scan_i8_b48c32s8, 48, 32, 8, 0, 48 * 32)
// The reference raised 16 above a new maximum, so that most new maxima rescale nothing.
SP_DEFINE_ATTN_SCAN_B(sp_attn_scan_i8_h48c64s8, 48, 64, 8, 16, 48 * 64)
SP_DEFINE_ATTN_SCAN_B(sp_attn_scan_i8_h48c32s8, 48, 32, 8, 16, 48 * 32)
// Test only: expects a score partition that does not exist, so its references must be NaN.
SP_DEFINE_ATTN_SCAN_B(sp_attn_scan_i8_x48c64s8, 48, 64, 8, 0, 48 * 64 + 1)
