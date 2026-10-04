#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// engine.metal — the production execution kernels (prefix `sp_`).
//
// The older kernels in this directory are scalar oracles. These are the ones the whole-model
// path dispatches. Formulas follow audit/QWEN35-REFERENCE-MATH.md.
//
// Conventions
//   * Activations are fp32, row-major `[rows, dim]`. A "row" is one token position of one
//     session; a step may mix rows from many sessions (batched decode) and many positions of
//     one session (prefill).
//   * Reductions use one simdgroup per threadgroup: the host dispatches 32-thread groups and
//     verifies `threadExecutionWidth == 32`, so `simd_sum` spans the whole group.
//   * q4 weights are MLX affine, group 64: eight low-first nibbles per U32 word.

#define SP_LANES 32u
#define SP_ROW_BLOCK 8u
#define SP_PAGE_TOKENS 256u
#define SP_HEAD_DIM 256u

inline float sp_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline float sp_silu(float x) { return x / (1.0f + exp(-x)); }
inline float sp_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// ---------------------------------------------------------------------------------------------
// Embedding: gather a q4 row per token and dequantise it.
// Thread grid: (strideWords, rows).

struct SpEmbedParams {
    uint rows;
    uint strideWords;
    uint groupsPerRow;
    uint hidden;
};

kernel void sp_embed_q4(device const uint *packed [[buffer(0)]],
                        device const ushort *scales [[buffer(1)]],
                        device const ushort *biases [[buffer(2)]],
                        device const uint *tokens [[buffer(3)]],
                        device float *out [[buffer(4)]],
                        constant SpEmbedParams &p [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.strideWords || gid.y >= p.rows) return;
    const uint token = tokens[gid.y];
    const uint word = packed[token * p.strideWords + gid.x];
    const uint group = token * p.groupsPerRow + (gid.x >> 3);
    const float s = sp_bf16(scales[group]);
    const float b = sp_bf16(biases[group]);
    const uint base = gid.y * p.hidden + gid.x * 8u;
    for (uint nib = 0; nib < 8u; ++nib) {
        out[base + nib] = s * float((word >> (nib * 4u)) & 0xFu) + b;
    }
}

// ---------------------------------------------------------------------------------------------
// q4 GEMM: out[r][n] = sum_k a[r][k] * dequant(W)[n][k]  (+ residual[r][n])
//
// One 32-lane threadgroup per (output column n, row block). Lane L reads packed words
// L, L+32, ... of weight row n, so adjacent lanes read adjacent words and every weight byte is
// read exactly once per row block. Up to SP_ROW_BLOCK activation rows share one pass over the
// weights; that sharing is what makes batched decode nearly free in bandwidth terms.
// Threadgroup grid: (outDim, ceil(rows / SP_ROW_BLOCK)).

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

kernel void sp_gemm_q4(device const uint *packed [[buffer(0)]],
                       device const ushort *scales [[buffer(1)]],
                       device const ushort *biases [[buffer(2)]],
                       device const float *a [[buffer(3)]],
                       device float *out [[buffer(4)]],
                       device const float *residual [[buffer(5)]],
                       constant SpGemmParams &p [[buffer(6)]],
                       uint3 group [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]])
{
    const uint n = group.x;
    const uint r0 = group.y * SP_ROW_BLOCK;
    if (n >= p.outDim || r0 >= p.rows) return;
    const uint count = min(SP_ROW_BLOCK, p.rows - r0);

    float acc[SP_ROW_BLOCK];
    for (uint m = 0; m < SP_ROW_BLOCK; ++m) acc[m] = 0.0f;

    const uint words = p.inner >> 3;
    const uint wordBase = n * p.strideWords;
    const uint groupBase = n * p.groupsPerRow;
    for (uint w = lane; w < words; w += SP_LANES) {
        const uint word = packed[wordBase + w];
        const float s = sp_bf16(scales[groupBase + (w >> 3)]);
        const float b = sp_bf16(biases[groupBase + (w >> 3)]);
        const float4 q0 = float4(float(word & 0xFu), float((word >> 4) & 0xFu),
                                 float((word >> 8) & 0xFu), float((word >> 12) & 0xFu));
        const float4 q1 = float4(float((word >> 16) & 0xFu), float((word >> 20) & 0xFu),
                                 float((word >> 24) & 0xFu), float((word >> 28) & 0xFu));
        const float4 w0 = s * q0 + b;
        const float4 w1 = s * q1 + b;
        const uint k = w << 3;
        for (uint m = 0; m < count; ++m) {
            device const packed_float4 *ap =
                (device const packed_float4 *)(a + (r0 + m) * p.inStride + k);
            acc[m] += dot(float4(ap[0]), w0) + dot(float4(ap[1]), w1);
        }
    }
    for (uint m = 0; m < count; ++m) {
        const float total = simd_sum(acc[m]);
        if (lane == 0u) {
            const uint index = (r0 + m) * p.outStride + n;
            out[index] = p.hasResidual != 0u ? total + residual[index] : total;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Bandwidth probe: sum a buffer with fully coalesced 16-byte loads. One 32-lane group per
// 64 KiB chunk. Threadgroup grid: (chunks).
struct SpProbeParams { uint chunks; };
kernel void sp_probe_read(device const uint *data [[buffer(0)]],
                          device uint *out [[buffer(1)]],
                          constant SpProbeParams &p [[buffer(2)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= p.chunks) return;
    device const packed_uint4 *wp = (device const packed_uint4 *)(data + group.x * 16384u);
    uint4 total = uint4(0u);
    for (uint i = lane; i < 4096u; i += SP_LANES) total += uint4(wp[i]);
    const uint sum = simd_sum(total.x + total.y + total.z + total.w);
    if (lane == 0u) out[group.x] = sum;
}

// ---------------------------------------------------------------------------------------------
// RMSNorm over `count` vectors of `dim`: x * rsqrt(mean(x^2)+eps) * w.
//
// The reference model computes (1 + w) with zero-initialised w, but the MLX conversion that
// produced this artifact stores w + 1 for every such norm (mlx_lm qwen3_5 `sanitize`), so the
// stored weight is applied directly. `mode` is retained for a future un-baked artifact.
// Threadgroup grid: (count).

struct SpNormParams {
    uint count;
    uint dim;
    uint mode;
    float eps;
};

kernel void sp_rmsnorm(device const float *input [[buffer(0)]],
                       device const ushort *weight [[buffer(1)]],
                       device float *output [[buffer(2)]],
                       constant SpNormParams &p [[buffer(3)]],
                       uint3 group [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= p.count) return;
    const uint base = group.x * p.dim;
    float partial = 0.0f;
    for (uint i = lane; i < p.dim; i += SP_LANES) {
        const float x = input[base + i];
        partial += x * x;
    }
    const float inv = rsqrt(simd_sum(partial) / float(p.dim) + p.eps);
    for (uint i = lane; i < p.dim; i += SP_LANES) {
        const float w = sp_bf16(weight[i]);
        output[base + i] = input[base + i] * inv * (p.mode == 0u ? 1.0f + w : w);
    }
}

// ---------------------------------------------------------------------------------------------
// Elementwise helpers. Thread grids.

struct SpCountParams { uint count; };

// out = silu(gate) * up
kernel void sp_silu_mul(device const float *gate [[buffer(0)]],
                        device const float *up [[buffer(1)]],
                        device float *out [[buffer(2)]],
                        constant SpCountParams &p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]])
{
    if (gid >= p.count) return;
    out[gid] = sp_silu(gate[gid]) * up[gid];
}

// The same product emitted as the accelerator operand of the following GEMM: bf16 values and
// their per-64 sums. Saves the separate conversion stage.
// Threadgroup grid: (dim / 64, rows), 32 threads per group.
struct SpSiluNaParams { uint rows; uint dim; };
kernel void sp_silu_mul_na(device const float *gate [[buffer(0)]],
                           device const float *up [[buffer(1)]],
                           device bfloat *outB [[buffer(2)]],
                           device float *sums [[buffer(3)]],
                           constant SpSiluNaParams &p [[buffer(4)]],
                           device const float *rowInv [[buffer(5)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= p.dim / 64u || group.y >= p.rows) return;
    const uint i0 = group.y * p.dim + group.x * 64u + lane, i1 = i0 + 32u;
    // Both projections carry the deferred scale of the norm that fed them (1 when it was applied).
    const float scale = rowInv[group.y];
    const bfloat b0 = bfloat(sp_silu(gate[i0] * scale) * (up[i0] * scale));
    const bfloat b1 = bfloat(sp_silu(gate[i1] * scale) * (up[i1] * scale));
    outB[i0] = b0;
    outB[i1] = b1;
    const float total = simd_sum(float(b0) + float(b1));
    if (lane == 0u) sums[group.y * (p.dim / 64u) + group.x] = total;
}

// The per-row RMSNorm scale from per-64 sums of squares (see SpNaEmit in engine_na.metal).
// Threadgroup grid: (rows), 32 threads per group.
struct SpRowInvParams { uint rows; uint groups; uint dim; float eps; };
kernel void sp_row_inv(device const float *squares [[buffer(0)]],
                       device float *inv [[buffer(1)]],
                       constant SpRowInvParams &p [[buffer(2)]],
                       uint3 group [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= p.rows) return;
    float partial = 0.0f;
    for (uint g = lane; g < p.groups; g += SP_LANES) partial += squares[group.x * p.groups + g];
    const float total = simd_sum(partial);
    if (lane == 0u) inv[group.x] = rsqrt(total / float(p.dim) + p.eps);
}

// q4 GEMM for a small row-major weight from a bf16 operand (the 48-row a and b projections,
// whose input exists only in that form when the norm scale is deferred).
// Threadgroup grid: (outDim, rows), 32 threads per group.
kernel void sp_gemm_q4_b(device const uint *packed [[buffer(0)]],
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
    const uint words = p.inner >> 3;
    const uint wordBase = n * p.strideWords;
    const uint groupBase = n * p.groupsPerRow;
    float acc = 0.0f;
    for (uint w = lane; w < words; w += SP_LANES) {
        const uint word = packed[wordBase + w];
        const float s = sp_bf16(scales[groupBase + (w >> 3)]);
        const float b = sp_bf16(biases[groupBase + (w >> 3)]);
        const float4 q0 = float4(float(word & 0xFu), float((word >> 4) & 0xFu),
                                 float((word >> 8) & 0xFu), float((word >> 12) & 0xFu));
        const float4 q1 = float4(float((word >> 16) & 0xFu), float((word >> 20) & 0xFu),
                                 float((word >> 24) & 0xFu), float((word >> 28) & 0xFu));
        device const packed_bfloat4 *ap = (device const packed_bfloat4 *)(a + row * p.inStride + (w << 3));
        acc += dot(float4(bfloat4(ap[0])), s * q0 + b) + dot(float4(bfloat4(ap[1])), s * q1 + b);
    }
    const float total = simd_sum(acc);
    if (lane == 0u) out[row * p.outStride + n] = total;
}

struct SpGatherParams { uint count; uint dim; };

// out[i] = in[index[i]]
kernel void sp_gather_rows(device const float *input [[buffer(0)]],
                           device const uint *index [[buffer(1)]],
                           device float *out [[buffer(2)]],
                           constant SpGatherParams &p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.dim || gid.y >= p.count) return;
    out[gid.y * p.dim + gid.x] = input[index[gid.y] * p.dim + gid.x];
}

// ---------------------------------------------------------------------------------------------
// Full attention.

inline float sp_rope(float x, float paired, uint i, uint rotary, uint position, float theta) {
    const uint half_ = rotary >> 1;
    const uint slot = i < half_ ? i : i - half_;
    const float angle = float(position) * pow(theta, -(2.0f * float(slot)) / float(rotary));
    return x * cos(angle) + (i < half_ ? -paired : paired) * sin(angle);
}

struct SpAttnPrepParams {
    uint rows;
    uint heads;        // q heads for the query kernel, kv heads for the store kernel
    uint rotary;
    uint inHeadStride; // 512 for q_proj ([query | gate] per head), 256 for k_proj
    uint maxPages;
    float eps;
    float theta;
};

// q: RMSNorm over the head (stored weight already includes the +1), then RoPE. Threadgroup grid: (heads, rows).
kernel void sp_attn_q_prepare(device const float *qproj [[buffer(0)]],
                              device const ushort *normWeight [[buffer(1)]],
                              device const uint *rowPos [[buffer(2)]],
                              device float *out [[buffer(3)]],
                              constant SpAttnPrepParams &p [[buffer(4)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]])
{
    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    const uint base = (row * p.heads + head) * p.inHeadStride;
    float partial = 0.0f;
    for (uint i = lane; i < SP_HEAD_DIM; i += SP_LANES) {
        const float x = qproj[base + i];
        partial += x * x;
    }
    const float inv = rsqrt(simd_sum(partial) / float(SP_HEAD_DIM) + p.eps);
    const uint position = rowPos[row];
    const uint half_ = p.rotary >> 1;
    const uint outBase = (row * p.heads + head) * SP_HEAD_DIM;
    for (uint i = lane; i < SP_HEAD_DIM; i += SP_LANES) {
        float x = qproj[base + i] * inv * sp_bf16(normWeight[i]);
        if (i < p.rotary) {
            const uint pi = i < half_ ? i + half_ : i - half_;
            const float paired = qproj[base + pi] * inv * sp_bf16(normWeight[pi]);
            x = sp_rope(x, paired, i, p.rotary, position, p.theta);
        }
        out[outBase + i] = x;
    }
}

// k: RMSNorm + RoPE, v: verbatim. Both are written into the paged KV pool as fp16.
// Threadgroup grid: (kvHeads, rows).
kernel void sp_attn_kv_store(device const float *kproj [[buffer(0)]],
                             device const float *vproj [[buffer(1)]],
                             device const ushort *normWeight [[buffer(2)]],
                             device const uint *rowSlot [[buffer(3)]],
                             device const uint *rowPos [[buffer(4)]],
                             device const uint *pageTable [[buffer(5)]],
                             device half *kpool [[buffer(6)]],
                             device half *vpool [[buffer(7)]],
                             constant SpAttnPrepParams &p [[buffer(8)]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]])
{
    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    const uint base = (row * p.heads + head) * SP_HEAD_DIM;
    float partial = 0.0f;
    for (uint i = lane; i < SP_HEAD_DIM; i += SP_LANES) {
        const float x = kproj[base + i];
        partial += x * x;
    }
    const float inv = rsqrt(simd_sum(partial) / float(SP_HEAD_DIM) + p.eps);
    const uint position = rowPos[row];
    const uint page = pageTable[rowSlot[row] * p.maxPages + position / SP_PAGE_TOKENS];
    const uint dst = ((page * SP_PAGE_TOKENS + position % SP_PAGE_TOKENS) * p.heads + head) * SP_HEAD_DIM;
    const uint half_ = p.rotary >> 1;
    for (uint i = lane; i < SP_HEAD_DIM; i += SP_LANES) {
        float x = kproj[base + i] * inv * sp_bf16(normWeight[i]);
        if (i < p.rotary) {
            const uint pi = i < half_ ? i + half_ : i - half_;
            const float paired = kproj[base + pi] * inv * sp_bf16(normWeight[pi]);
            x = sp_rope(x, paired, i, p.rotary, position, p.theta);
        }
        kpool[dst + i] = half(x);
        vpool[dst + i] = half(vproj[base + i]);
    }
}

struct SpAttnParams {
    uint rows;
    uint heads;
    uint kvHeads;
    uint maxPages;
    float scale;
};

// Causal GQA attention with an online softmax over KV pages, then the sigmoid output gate.
// One 256-thread group per (head, row): thread j scores token j of each page and owns output
// channel j. Threadgroup grid: (heads, rows), 256 threads per group.
kernel void sp_attention(device const float *qn [[buffer(0)]],
                         device const float *qproj [[buffer(1)]],
                         device const uint *rowSlot [[buffer(2)]],
                         device const uint *rowPos [[buffer(3)]],
                         device const uint *pageTable [[buffer(4)]],
                         device const half *kpool [[buffer(5)]],
                         device const half *vpool [[buffer(6)]],
                         device float *out [[buffer(7)]],
                         constant SpAttnParams &p [[buffer(8)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint j [[thread_index_in_threadgroup]])
{
    threadgroup float shared[SP_PAGE_TOKENS];

    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    const uint kvHead = head / (p.heads / p.kvHeads);
    const uint visible = rowPos[row] + 1u;
    const uint pages = (visible + SP_PAGE_TOKENS - 1u) / SP_PAGE_TOKENS;
    const uint tableBase = rowSlot[row] * p.maxPages;
    const uint qBase = (row * p.heads + head) * SP_HEAD_DIM;

    float runMax = -INFINITY, runSum = 0.0f, acc = 0.0f;
    for (uint pg = 0; pg < pages; ++pg) {
        const uint pageBase = pageTable[tableBase + pg] * SP_PAGE_TOKENS;
        float score = -INFINITY;
        if (pg * SP_PAGE_TOKENS + j < visible) {
            const uint kBase = ((pageBase + j) * p.kvHeads + kvHead) * SP_HEAD_DIM;
            float d = 0.0f;
            for (uint i = 0; i < SP_HEAD_DIM; ++i) d += qn[qBase + i] * float(kpool[kBase + i]);
            score = d * p.scale;
        }
        shared[j] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float pageMax = -INFINITY;
        for (uint t = 0; t < SP_PAGE_TOKENS; ++t) pageMax = max(pageMax, shared[t]);
        const float newMax = max(runMax, pageMax);
        const float rescale = exp(runMax - newMax);
        acc *= rescale;
        runSum *= rescale;
        runMax = newMax;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared[j] = exp(score - runMax);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0; t < SP_PAGE_TOKENS; ++t) {
            const float prob = shared[t];
            if (prob > 0.0f) {
                runSum += prob;
                acc += prob * float(vpool[((pageBase + t) * p.kvHeads + kvHead) * SP_HEAD_DIM + j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float gate = qproj[(row * p.heads + head) * (2u * SP_HEAD_DIM) + SP_HEAD_DIM + j];
    out[qBase + j] = (acc / runSum) * sp_sigmoid(gate);
}

// ---------------------------------------------------------------------------------------------
// Gated DeltaNet. `runs` holds (slot, startRow, length) triples: the rows of one session inside
// a step are consecutive and must be scanned in order.

struct SpGdnParams {
    uint rows;
    uint runs;
    uint keyHeads;     // 16
    uint valueHeads;   // 48
    uint headDim;      // 128 (key and value head dims are equal for this checkpoint)
    uint channels;     // 2 * keyDim + valueDim = 10240
    float eps;
    uint debug;        // timing probes for sp_gdn_fast; 0 in normal operation
};

// Depthwise causal conv (kernel 4, no bias) + SiLU, in place over the qkv projection, carrying
// the three-sample history per (slot, channel). Thread grid: (channels, runs).
kernel void sp_gdn_conv(device float *mixed [[buffer(0)]],
                        device const ushort *weight [[buffer(1)]],
                        device float *convState [[buffer(2)]],
                        device const uint *runs [[buffer(3)]],
                        constant SpGdnParams &p [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.channels || gid.y >= p.runs) return;
    const uint c = gid.x;
    const uint slot = runs[gid.y * 3u], start = runs[gid.y * 3u + 1u], length = runs[gid.y * 3u + 2u];
    const uint sb = (slot * p.channels + c) * 3u;
    float s0 = convState[sb], s1 = convState[sb + 1u], s2 = convState[sb + 2u];
    const float w0 = sp_bf16(weight[c * 4u]), w1 = sp_bf16(weight[c * 4u + 1u]);
    const float w2 = sp_bf16(weight[c * 4u + 2u]), w3 = sp_bf16(weight[c * 4u + 3u]);
    for (uint t = 0; t < length; ++t) {
        const uint index = (start + t) * p.channels + c;
        const float x = mixed[index];
        mixed[index] = sp_silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x);
        s0 = s1; s1 = s2; s2 = x;
    }
    convState[sb] = s0; convState[sb + 1u] = s1; convState[sb + 2u] = s2;
}

// L2-normalise q and k per key head, in place; q is additionally scaled by 1/sqrt(headDim).
// Threadgroup grid: (2 * keyHeads, rows) — vectors 0..15 are q heads, 16..31 are k heads.
kernel void sp_gdn_l2norm(device float *mixed [[buffer(0)]],
                          constant SpGdnParams &p [[buffer(1)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= 2u * p.keyHeads || group.y >= p.rows) return;
    const uint base = group.y * p.channels + group.x * p.headDim;
    float partial = 0.0f;
    for (uint i = lane; i < p.headDim; i += SP_LANES) {
        const float x = mixed[base + i];
        partial += x * x;
    }
    float inv = rsqrt(simd_sum(partial) + 1e-6f);
    if (group.x < p.keyHeads) inv *= rsqrt(float(p.headDim));
    for (uint i = lane; i < p.headDim; i += SP_LANES) mixed[base + i] *= inv;
}

// The gated delta rule. Thread j of value head h owns column j of that head's state, so threads
// never share state and no barrier is needed. Thread grid: (headDim, valueHeads, runs).
kernel void sp_gdn_scan(device const float *mixed [[buffer(0)]],
                        device const float *a [[buffer(1)]],
                        device const float *b [[buffer(2)]],
                        device const ushort *aLog [[buffer(3)]],
                        device const ushort *dtBias [[buffer(4)]],
                        device float *state [[buffer(5)]],
                        device float *core [[buffer(6)]],
                        device const uint *runs [[buffer(7)]],
                        constant SpGdnParams &p [[buffer(8)]],
                        uint3 gid [[thread_position_in_grid]])
{
    const uint j = gid.x, h = gid.y;
    if (j >= p.headDim || h >= p.valueHeads || gid.z >= p.runs) return;
    const uint slot = runs[gid.z * 3u], start = runs[gid.z * 3u + 1u], length = runs[gid.z * 3u + 2u];
    const uint d = p.headDim;
    const uint keyDim = p.keyHeads * d;
    const uint hk = h / (p.valueHeads / p.keyHeads);
    const uint sBase = ((slot * p.valueHeads + h) * d) * d + j;
    const float negA = -exp(sp_bf16(aLog[h]));
    const float bias = sp_bf16(dtBias[h]);
    for (uint t = 0; t < length; ++t) {
        const uint row = start + t;
        const uint qBase = row * p.channels + hk * d;
        const uint kBase = qBase + keyDim;
        const float value = mixed[row * p.channels + 2u * keyDim + h * d + j];
        const float x = a[row * p.valueHeads + h] + bias;
        const float softplus = x > 20.0f ? x : log(1.0f + exp(x));
        const float decay = exp(negA * softplus);
        const float beta = sp_sigmoid(b[row * p.valueHeads + h]);
        float memory = 0.0f;
        for (uint i = 0; i < d; ++i) {
            const float s = state[sBase + i * d] * decay;
            state[sBase + i * d] = s;
            memory += s * mixed[kBase + i];
        }
        const float delta = (value - memory) * beta;
        float o = 0.0f;
        for (uint i = 0; i < d; ++i) {
            const float s = state[sBase + i * d] + mixed[kBase + i] * delta;
            state[sBase + i * d] = s;
            o += s * mixed[qBase + i];
        }
        core[row * (p.valueHeads * d) + h * d + j] = o;
    }
}

// RMSNormGated per value head: weight * x * rsqrt(mean(x^2)+eps) * silu(z). In place.
// Threadgroup grid: (valueHeads, rows).
kernel void sp_gdn_gated_norm(device float *core [[buffer(0)]],
                              device const float *z [[buffer(1)]],
                              device const ushort *weight [[buffer(2)]],
                              constant SpGdnParams &p [[buffer(3)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= p.valueHeads || group.y >= p.rows) return;
    const uint base = (group.y * p.valueHeads + group.x) * p.headDim;
    float partial = 0.0f;
    for (uint i = lane; i < p.headDim; i += SP_LANES) {
        const float x = core[base + i];
        partial += x * x;
    }
    const float inv = rsqrt(simd_sum(partial) / float(p.headDim) + p.eps);
    for (uint i = lane; i < p.headDim; i += SP_LANES) {
        core[base + i] = sp_bf16(weight[i]) * core[base + i] * inv * sp_silu(z[base + i]);
    }
}

// Zero a float range (state reset when a slot is reused). Thread grid: (count).
struct SpFillParams { uint offset; uint count; };
kernel void sp_zero(device float *buffer [[buffer(0)]],
                    constant SpFillParams &p [[buffer(1)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.count) return;
    buffer[p.offset + gid] = 0.0f;
}

// =============================================================================================
// Fused stage kernels.
//
// A dependency barrier between dispatches costs ~32 us on this GPU while a concurrent dispatch
// costs ~0.4 us, so a decode step is paced by how many dependent stages each layer needs, not
// by how many kernels it runs. These kernels exist to collapse stages.

// MLP input: act = silu(gate_proj(x)) * up_proj(x), both projections in one pass.
// Threadgroup grid: (intermediate, ceil(rows / SP_ROW_BLOCK)).
kernel void sp_mlp_in(device const uint *gatePacked [[buffer(0)]],
                      device const ushort *gateScales [[buffer(1)]],
                      device const ushort *gateBiases [[buffer(2)]],
                      device const uint *upPacked [[buffer(3)]],
                      device const ushort *upScales [[buffer(4)]],
                      device const ushort *upBiases [[buffer(5)]],
                      device const float *a [[buffer(6)]],
                      device float *out [[buffer(7)]],
                      constant SpGemmParams &p [[buffer(8)]],
                      uint3 group [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_threadgroup]])
{
    const uint n = group.x;
    const uint r0 = group.y * SP_ROW_BLOCK;
    if (n >= p.outDim || r0 >= p.rows) return;
    const uint count = min(SP_ROW_BLOCK, p.rows - r0);
    float accG[SP_ROW_BLOCK], accU[SP_ROW_BLOCK];
    for (uint m = 0; m < SP_ROW_BLOCK; ++m) { accG[m] = 0.0f; accU[m] = 0.0f; }
    const uint words = p.inner >> 3;
    const uint wordBase = n * p.strideWords;
    const uint groupBase = n * p.groupsPerRow;
    for (uint w = lane; w < words; w += SP_LANES) {
        const uint gw = gatePacked[wordBase + w];
        const uint uw = upPacked[wordBase + w];
        const uint gi = groupBase + (w >> 3);
        const float gs = sp_bf16(gateScales[gi]), gb = sp_bf16(gateBiases[gi]);
        const float us = sp_bf16(upScales[gi]), ub = sp_bf16(upBiases[gi]);
        const float4 g0 = gs * float4((uint4(gw) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + gb;
        const float4 g1 = gs * float4((uint4(gw) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + gb;
        const float4 u0 = us * float4((uint4(uw) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + ub;
        const float4 u1 = us * float4((uint4(uw) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + ub;
        const uint k = w << 3;
        for (uint m = 0; m < count; ++m) {
            device const packed_float4 *ap = (device const packed_float4 *)(a + (r0 + m) * p.inStride + k);
            const float4 a0 = float4(ap[0]), a1 = float4(ap[1]);
            accG[m] += dot(a0, g0) + dot(a1, g1);
            accU[m] += dot(a0, u0) + dot(a1, u1);
        }
    }
    for (uint m = 0; m < count; ++m) {
        const float gate = simd_sum(accG[m]);
        const float up = simd_sum(accU[m]);
        if (lane == 0u) out[(r0 + m) * p.outStride + n] = sp_silu(gate) * up;
    }
}

// Journal entry per (slot, speculative row, value head): k[128], delta[128], decay, padding.
#define SP_GDN_JOURNAL_ROWS 16u
#define SP_GDN_JOURNAL_WIDTH 260u

// The same layer in three phases inside one threadgroup, so that the per-token recurrence — the
// only strictly sequential part — runs without any barrier:
//
//   A  conv + SiLU for q, k and v, and decay/beta, for every row of the run at once
//   B  the recurrence, state in registers: read from memory once per step, written once
//      (or once per speculative row)
//   C  gated RMSNorm, the bf16 operand and its per-64 sums, for every row at once
//
// Threadgroup grid: (valueHeads, runs), 1024 threads. In phase B simdgroup s owns state columns
// 4s ..< 4s + 4; lane l holds rows 16(l & 7) ..< 16(l & 7) + 16 of column 4s + (l >> 3), and a
// sum over a column's eight parts is three xor-shuffles inside the simdgroup.
// The state update S' = S * decay + k * delta. A speculative row's update is replayed later
// from its journal entry; written once, with the rounding spelled out, the replay reproduces
// the original bit for bit.
inline float4 sp_gdn_update(float4 state, float decay, float4 k, float delta) {
    return fma(state, float4(decay), k * delta);
}

inline float sp_sum_parts(float x) {
    x += simd_shuffle_xor(x, ushort(1));
    x += simd_shuffle_xor(x, ushort(2));
    x += simd_shuffle_xor(x, ushort(4));
    return x;
}

kernel void sp_gdn_fast(device const float *mixed [[buffer(0)]],
                        device const float *z [[buffer(1)]],
                        device float *a [[buffer(2)]],
                        device float *b [[buffer(3)]],
                        device const ushort *convWeight [[buffer(4)]],
                        device const ushort *aLog [[buffer(5)]],
                        device const ushort *dtBias [[buffer(6)]],
                        device const ushort *normWeight [[buffer(7)]],
                        device float *convState [[buffer(8)]],
                        device float *state [[buffer(9)]],
                        device float *core [[buffer(10)]],
                        device const uint *runs [[buffer(11)]],
                        constant SpGdnParams &p [[buffer(12)]],
                        device const uint *rowRead [[buffer(13)]],
                        device const uint *rowWrite [[buffer(14)]],
                        device bfloat *coreB [[buffer(15)]],
                        device float *coreSums [[buffer(16)]],
                        device float *gq [[buffer(17)]],
                        device float *gk [[buffer(18)]],
                        device float *gv [[buffer(19)]],
                        device float *gnorm [[buffer(20)]],
                        device float *journal [[buffer(21)]],
                        device const uint *runInfo [[buffer(22)]],
                        device const float *rowInv [[buffer(23)]],
                        uint3 group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint h = group.x;
    if (h >= p.valueHeads || group.y >= p.runs) return;
    const uint start = runs[group.y * 3u + 1u], length = runs[group.y * 3u + 2u];
    const uint d = p.headDim;                       // 128
    const uint keyDim = p.keyHeads * d;
    const uint valueDim = p.valueHeads * d;
    const uint hk = h / (p.valueHeads / p.keyHeads);
    const uint convPerUnit = p.valueHeads * 3u * d * 3u;

    // ---- Phase A. Threads 0 ..< 768: channel c = tid % 384 (q, k, v of this head), one half of
    // the rows each. The conv has no recurrence: its history is the raw input of earlier rows.
    const uint c = tid % 384u, kind = c / d, jc = c % d;
    const uint channel = kind == 0u ? hk * d + jc : kind == 1u ? keyDim + hk * d + jc : 2u * keyDim + h * d + jc;
    const uint convInUnit = ((h * 3u + kind) * d + jc) * 3u;
    float3 history = float3(0.0f);
    if (tid < 768u) {
        const uint first = rowRead[start] * convPerUnit + convInUnit;
        history = float3(convState[first], convState[first + 1u], convState[first + 2u]);
        const float4 w = float4(sp_bf16(convWeight[channel * 4u]), sp_bf16(convWeight[channel * 4u + 1u]),
                                sp_bf16(convWeight[channel * 4u + 2u]), sp_bf16(convWeight[channel * 4u + 3u]));
        device float *target = kind == 0u ? gq : kind == 1u ? gk : gv;
        const uint half_ = (length + 1u) / 2u;
        const uint from = (tid / 384u) * half_, to = min(length, from + half_);
        // Every projection of a row carries the deferred scale of the norm that fed it (1 when
        // the norm applied it); the stored history is already scaled.
        for (uint t = from; t < to; ++t) {
            const uint mb = (start + t) * p.channels + channel;
            const float x0 = mixed[mb] * rowInv[start + t];
            const float x1 = t >= 1u ? mixed[mb - p.channels] * rowInv[start + t - 1u] : history.z;
            const float x2 = t >= 2u ? mixed[mb - 2u * p.channels] * rowInv[start + t - 2u] : (t == 1u ? history.z : history.y);
            const float x3 = t >= 3u ? mixed[mb - 3u * p.channels] * rowInv[start + t - 3u] : (t == 2u ? history.z : t == 1u ? history.y : history.x);
            target[((start + t) * p.valueHeads + h) * d + jc] = sp_silu(dot(w, float4(x3, x2, x1, x0)));
        }
    } else {
        const float negA = -exp(sp_bf16(aLog[h]));
        const float bias = sp_bf16(dtBias[h]);
        for (uint t = tid - 768u; t < length; t += 256u) {
            const uint index = (start + t) * p.valueHeads + h;
            const float x = a[index] * rowInv[start + t] + bias;
            const float softplus = x > 20.0f ? x : log(1.0f + exp(x));
            a[index] = exp(negA * softplus);
            b[index] = sp_sigmoid(b[index] * rowInv[start + t]);
        }
    }

    // The slot's state, into registers. A speculative run does not write states: each of its
    // rows leaves a journal entry (k, delta, decay per head) from which its rank-1 update can be
    // replayed. Entries the caller accepted from the previous run are replayed here first.
    const uint slot = runs[group.y * 3u];
    const bool speculative = runInfo[group.y * 2u] != 0u;
    const uint pending = runInfo[group.y * 2u + 1u];
    const uint part = lane & 7u, j = sg * 4u + (lane >> 3);
    const uint statePerUnit = p.valueHeads * d * d;
    device packed_float4 *stored = (device packed_float4 *)(state + slot * statePerUnit + ((h * d) + j) * d + part * 16u);
    float4 s0 = float4(stored[0]), s1 = float4(stored[1]), s2 = float4(stored[2]), s3 = float4(stored[3]);
    device float *journalHead = journal + (slot * SP_GDN_JOURNAL_ROWS * p.valueHeads + h) * SP_GDN_JOURNAL_WIDTH;
    for (uint e = 0; e < pending; ++e) {
        device const float *entry = journalHead + e * p.valueHeads * SP_GDN_JOURNAL_WIDTH;
        device const packed_float4 *ks = (device const packed_float4 *)(entry + part * 16u);
        const float delta = entry[d + j], decay = entry[2u * d];
        s0 = sp_gdn_update(s0, decay, float4(ks[0]), delta); s1 = sp_gdn_update(s1, decay, float4(ks[1]), delta);
        s2 = sp_gdn_update(s2, decay, float4(ks[2]), delta); s3 = sp_gdn_update(s3, decay, float4(ks[3]), delta);
    }
    // An ordinary run writes the state at its end; a speculative one only when it replayed.
    if (speculative && pending != 0u) { stored[0] = s0; stored[1] = s1; stored[2] = s2; stored[3] = s3; }
    threadgroup_barrier(mem_flags::mem_device);

    // L2 norms of q and k per row (q additionally scaled by 1/sqrt(headDim)): simdgroup s takes
    // rows s, s + 32, ...; a lane four elements of each vector.
    for (uint t = sg; t < length; t += 32u) {
        const uint base = ((start + t) * p.valueHeads + h) * d + lane * 4u;
        const float4 qv = float4(*(device const packed_float4 *)(gq + base));
        const float4 kv = float4(*(device const packed_float4 *)(gk + base));
        const float qq = simd_sum(dot(qv, qv)), kk = simd_sum(dot(kv, kv));
        if (lane == 0u) {
            gnorm[((start + t) * p.valueHeads + h) * 2u] = rsqrt(qq + 1e-6f) * rsqrt(float(d));
            gnorm[((start + t) * p.valueHeads + h) * 2u + 1u] = rsqrt(kk + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- Phase B.
    {
        for (uint t = 0; t < length; ++t) {
            if (p.debug == 3u) break;
            const uint row = start + t;
            const uint base = (row * p.valueHeads + h) * d;
            device const packed_float4 *qs = (device const packed_float4 *)(gq + base + part * 16u);
            device const packed_float4 *ks = (device const packed_float4 *)(gk + base + part * 16u);
            float4 q0 = float4(0.01f), q1 = q0, q2 = q0, q3 = q0, k0 = q0, k1 = q0, k2 = q0, k3 = q0;
            if (p.debug != 2u) {
                q0 = float4(qs[0]); q1 = float4(qs[1]); q2 = float4(qs[2]); q3 = float4(qs[3]);
                k0 = float4(ks[0]); k1 = float4(ks[1]); k2 = float4(ks[2]); k3 = float4(ks[3]);
            }
            const float mem = sp_sum_parts(dot(s0, k0) + dot(s1, k1) + dot(s2, k2) + dot(s3, k3));
            const float invQ = gnorm[(row * p.valueHeads + h) * 2u];
            const float invK = gnorm[(row * p.valueHeads + h) * 2u + 1u];
            const float decay = a[row * p.valueHeads + h];
            const float beta = b[row * p.valueHeads + h];
            const float delta = (gv[base + j] - mem * decay * invK) * beta * invK;
            s0 = sp_gdn_update(s0, decay, k0, delta); s1 = sp_gdn_update(s1, decay, k1, delta);
            s2 = sp_gdn_update(s2, decay, k2, delta); s3 = sp_gdn_update(s3, decay, k3, delta);
            const float o = sp_sum_parts(dot(s0, q0) + dot(s1, q1) + dot(s2, q2) + dot(s3, q3)) * invQ;
            if (part == 0u) core[base + j] = o;
            if (speculative) {
                // Column 0's eight parts between them hold the whole of k.
                device float *entry = journalHead + t * p.valueHeads * SP_GDN_JOURNAL_WIDTH;
                if (sg == 0u && lane < 8u) {
                    device packed_float4 *kw = (device packed_float4 *)(entry + part * 16u);
                    kw[0] = k0; kw[1] = k1; kw[2] = k2; kw[3] = k3;
                }
                if (part == 0u) entry[d + j] = delta;
                if (sg == 0u && lane == 0u) entry[2u * d] = decay;
            } else if (t + 1u == length) {
                stored[0] = s0; stored[1] = s1; stored[2] = s2; stored[3] = s3;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ---- Phase C. Simdgroup s takes rows s, s + 32, ...; lane l the columns l, l + 32, l + 64, l + 96.
    const float4 gain = float4(sp_bf16(normWeight[lane]), sp_bf16(normWeight[lane + 32u]),
                               sp_bf16(normWeight[lane + 64u]), sp_bf16(normWeight[lane + 96u]));
    for (uint t = sg; t < length; t += 32u) {
        const uint row = start + t;
        const uint base = (row * p.valueHeads + h) * d + lane;
        const float4 o = float4(core[base], core[base + 32u], core[base + 64u], core[base + 96u]);
        const float4 gate = float4(z[base], z[base + 32u], z[base + 64u], z[base + 96u]) * rowInv[row];
        const float inv = rsqrt(simd_sum(dot(o, o)) / float(d) + p.eps);
        const float4 result = gain * o * inv * gate / (1.0f + exp(-gate));
        const float4 rounded = float4(float(bfloat(result.x)), float(bfloat(result.y)), float(bfloat(result.z)), float(bfloat(result.w)));
        core[base] = result.x; core[base + 32u] = result.y; core[base + 64u] = result.z; core[base + 96u] = result.w;
        coreB[base] = bfloat(result.x); coreB[base + 32u] = bfloat(result.y);
        coreB[base + 64u] = bfloat(result.z); coreB[base + 96u] = bfloat(result.w);
        // The following accelerator GEMM's per-64 sums (a 128-wide head is two quant groups).
        const float low = simd_sum(rounded.x + rounded.y), high = simd_sum(rounded.z + rounded.w);
        if (lane == 0u) {
            coreSums[row * (valueDim / 64u) + h * 2u] = low;
            coreSums[row * (valueDim / 64u) + h * 2u + 1u] = high;
        }
    }
    // Conv history for every unit a row writes: the last three raw inputs up to that row.
    if (tid < 384u) {
        for (uint t = 0; t < length; ++t) {
            const uint row = start + t;
            if (t + 1u != length && rowWrite[row + 1u] == rowWrite[row]) continue;
            const uint mb = row * p.channels + channel;
            const float x1 = mixed[mb] * rowInv[row];
            const float x2 = t >= 1u ? mixed[mb - p.channels] * rowInv[row - 1u] : history.z;
            const float x3 = t >= 2u ? mixed[mb - 2u * p.channels] * rowInv[row - 2u] : (t == 1u ? history.z : history.y);
            const uint conv = rowWrite[row] * convPerUnit + convInUnit;
            convState[conv] = x3; convState[conv + 1u] = x2; convState[conv + 2u] = x1;
        }
    }
}

// The same layer as three dispatches, for steps wide enough that a stage is cheap beside the
// work. In sp_gdn_fast phases A and C are loops over the run inside 48 large threadgroups, as
// long as the recurrence itself though nothing in them is sequential; here every (row, head)
// is a threadgroup of its own. And the recurrence, whose columns are independent chains that
// never leave their simdgroup, is 8 threadgroups of 4 simdgroups per head instead of one of 32,
// which a GPU of 20 cores fills evenly.
//
//   sp_gdn_prepare   (valueHeads, rows), 96 threads: conv + SiLU of q, k, v; their norms;
//                    decay and beta
//   sp_gdn_chains    (valueHeads * 8, runs), 128 threads: the recurrence
//   sp_gdn_finish    (valueHeads, rows), 32 threads: gated RMSNorm, the bf16 operand, sums
//   sp_gdn_history   (valueHeads, runs), 384 threads: the conv history a run leaves
//
// The bindings are sp_gdn_fast's.
#define SP_GDN_ARGS                                                                               \
    device const float *mixed [[buffer(0)]], device const float *z [[buffer(1)]],                 \
    device float *a [[buffer(2)]], device float *b [[buffer(3)]],                                 \
    device const ushort *convWeight [[buffer(4)]], device const ushort *aLog [[buffer(5)]],       \
    device const ushort *dtBias [[buffer(6)]], device const ushort *normWeight [[buffer(7)]],     \
    device float *convState [[buffer(8)]], device float *state [[buffer(9)]],                     \
    device float *core [[buffer(10)]], device const uint *runs [[buffer(11)]],                    \
    constant SpGdnParams &p [[buffer(12)]], device const uint *rowRead [[buffer(13)]],            \
    device const uint *rowWrite [[buffer(14)]], device bfloat *coreB [[buffer(15)]],              \
    device float *coreSums [[buffer(16)]], device float *gq [[buffer(17)]],                       \
    device float *gk [[buffer(18)]], device float *gv [[buffer(19)]],                             \
    device float *gnorm [[buffer(20)]], device float *journal [[buffer(21)]],                     \
    device const uint *runInfo [[buffer(22)]], device const float *rowInv [[buffer(23)]]

kernel void sp_gdn_prepare(SP_GDN_ARGS,
                           uint3 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint h = group.x, row = group.y;
    if (h >= p.valueHeads || row >= p.rows || sg >= 3u) return;
    const uint d = p.headDim;
    const uint keyDim = p.keyHeads * d;
    const uint hk = h / (p.valueHeads / p.keyHeads);
    const uint convPerUnit = p.valueHeads * 3u * d * 3u;
    // The run this row is in: its history before the run is the stored one.
    uint start = 0u;
    for (uint r = 0; r < p.runs; ++r) {
        const uint first = runs[r * 3u + 1u];
        if (row >= first && row < first + runs[r * 3u + 2u]) start = first;
    }
    const uint t = row - start;
    const uint kind = sg;                                           // q, k, v
    device float *target = kind == 0u ? gq : kind == 1u ? gk : gv;
    const float inv0 = rowInv[row];
    float4 out;
    for (uint e = 0; e < 4u; ++e) {
        const uint jc = lane * 4u + e;
        const uint channel = kind == 0u ? hk * d + jc : kind == 1u ? keyDim + hk * d + jc : 2u * keyDim + h * d + jc;
        const float4 w = float4(sp_bf16(convWeight[channel * 4u]), sp_bf16(convWeight[channel * 4u + 1u]),
                                sp_bf16(convWeight[channel * 4u + 2u]), sp_bf16(convWeight[channel * 4u + 3u]));
        float3 history = float3(0.0f);
        if (t < 3u) {
            const uint stored = rowRead[start] * convPerUnit + ((h * 3u + kind) * d + jc) * 3u;
            history = float3(convState[stored], convState[stored + 1u], convState[stored + 2u]);
        }
        const uint mb = row * p.channels + channel;
        const float x0 = mixed[mb] * inv0;
        const float x1 = t >= 1u ? mixed[mb - p.channels] * rowInv[row - 1u] : history.z;
        const float x2 = t >= 2u ? mixed[mb - 2u * p.channels] * rowInv[row - 2u] : (t == 1u ? history.z : history.y);
        const float x3 = t >= 3u ? mixed[mb - 3u * p.channels] * rowInv[row - 3u] : (t == 2u ? history.z : t == 1u ? history.y : history.x);
        out[e] = sp_silu(dot(w, float4(x3, x2, x1, x0)));
    }
    *(device packed_float4 *)(target + (row * p.valueHeads + h) * d + lane * 4u) = out;
    const uint index = row * p.valueHeads + h;
    if (kind < 2u) {
        // L2 norms of q and k (q additionally scaled by 1/sqrt(headDim)).
        const float squares = simd_sum(dot(out, out));
        if (lane == 0u) gnorm[index * 2u + kind] = kind == 0u ? rsqrt(squares + 1e-6f) * rsqrt(float(d)) : rsqrt(squares + 1e-6f);
    } else if (lane == 0u) {
        const float negA = -exp(sp_bf16(aLog[h]));
        const float x = a[index] * inv0 + sp_bf16(dtBias[h]);
        const float softplus = x > 20.0f ? x : log(1.0f + exp(x));
        a[index] = exp(negA * softplus);
        b[index] = sp_sigmoid(b[index] * inv0);
    }
}

kernel void sp_gdn_chains(SP_GDN_ARGS,
                        uint3 group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sgLocal [[simdgroup_index_in_threadgroup]])
{
    const uint h = group.x / 8u;
    if (h >= p.valueHeads || group.y >= p.runs) return;
    // The simdgroup's place among the head's 32: it owns state columns 4 sg ..< 4 sg + 4.
    const uint sg = (group.x % 8u) * 4u + sgLocal;
    const uint start = runs[group.y * 3u + 1u], length = runs[group.y * 3u + 2u];
    const uint d = p.headDim;
    const uint slot = runs[group.y * 3u];
    const bool speculative = runInfo[group.y * 2u] != 0u;
    const uint part = lane & 7u, j = sg * 4u + (lane >> 3);
    const uint statePerUnit = p.valueHeads * d * d;
    // Journal entries accepted from the slot's last speculative run are already in the stored
    // state: sp_gdn_commit ran in the stage before (a speculative run here writes over those
    // entries, and the head's eight threadgroups could not all have read them first).
    device packed_float4 *stored = (device packed_float4 *)(state + slot * statePerUnit + ((h * d) + j) * d + part * 16u);
    float4 s0 = float4(stored[0]), s1 = float4(stored[1]), s2 = float4(stored[2]), s3 = float4(stored[3]);
    device float *journalHead = journal + (slot * SP_GDN_JOURNAL_ROWS * p.valueHeads + h) * SP_GDN_JOURNAL_WIDTH;
    for (uint t = 0; t < length; ++t) {
        const uint row = start + t;
        const uint base = (row * p.valueHeads + h) * d;
        device const packed_float4 *qs = (device const packed_float4 *)(gq + base + part * 16u);
        device const packed_float4 *ks = (device const packed_float4 *)(gk + base + part * 16u);
        const float4 q0 = float4(qs[0]), q1 = float4(qs[1]), q2 = float4(qs[2]), q3 = float4(qs[3]);
        const float4 k0 = float4(ks[0]), k1 = float4(ks[1]), k2 = float4(ks[2]), k3 = float4(ks[3]);
        const float mem = sp_sum_parts(dot(s0, k0) + dot(s1, k1) + dot(s2, k2) + dot(s3, k3));
        const float invQ = gnorm[(row * p.valueHeads + h) * 2u];
        const float invK = gnorm[(row * p.valueHeads + h) * 2u + 1u];
        const float decay = a[row * p.valueHeads + h];
        const float beta = b[row * p.valueHeads + h];
        const float delta = (gv[base + j] - mem * decay * invK) * beta * invK;
        s0 = sp_gdn_update(s0, decay, k0, delta); s1 = sp_gdn_update(s1, decay, k1, delta);
        s2 = sp_gdn_update(s2, decay, k2, delta); s3 = sp_gdn_update(s3, decay, k3, delta);
        const float o = sp_sum_parts(dot(s0, q0) + dot(s1, q1) + dot(s2, q2) + dot(s3, q3)) * invQ;
        if (part == 0u) core[base + j] = o;
        if (speculative) {
            device float *entry = journalHead + t * p.valueHeads * SP_GDN_JOURNAL_WIDTH;
            if (sg == 0u && lane < 8u) {
                device packed_float4 *kw = (device packed_float4 *)(entry + part * 16u);
                kw[0] = k0; kw[1] = k1; kw[2] = k2; kw[3] = k3;
            }
            if (part == 0u) entry[d + j] = delta;
            if (sg == 0u && lane == 0u) entry[2u * d] = decay;
        } else if (t + 1u == length) {
            stored[0] = s0; stored[1] = s1; stored[2] = s2; stored[3] = s3;
        }
    }
}

kernel void sp_gdn_finish(SP_GDN_ARGS,
                          uint3 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint h = group.x, row = group.y;
    if (h >= p.valueHeads || row >= p.rows || sg != 0u) return;
    const uint d = p.headDim;
    const uint valueDim = p.valueHeads * d;
    // Lane l takes the columns l, l + 32, l + 64, l + 96.
    const float4 gain = float4(sp_bf16(normWeight[lane]), sp_bf16(normWeight[lane + 32u]),
                               sp_bf16(normWeight[lane + 64u]), sp_bf16(normWeight[lane + 96u]));
    const uint base = (row * p.valueHeads + h) * d + lane;
    const float4 o = float4(core[base], core[base + 32u], core[base + 64u], core[base + 96u]);
    const float4 gate = float4(z[base], z[base + 32u], z[base + 64u], z[base + 96u]) * rowInv[row];
    const float inv = rsqrt(simd_sum(dot(o, o)) / float(d) + p.eps);
    const float4 result = gain * o * inv * gate / (1.0f + exp(-gate));
    const float4 rounded = float4(float(bfloat(result.x)), float(bfloat(result.y)), float(bfloat(result.z)), float(bfloat(result.w)));
    core[base] = result.x; core[base + 32u] = result.y; core[base + 64u] = result.z; core[base + 96u] = result.w;
    coreB[base] = bfloat(result.x); coreB[base + 32u] = bfloat(result.y);
    coreB[base + 64u] = bfloat(result.z); coreB[base + 96u] = bfloat(result.w);
    // The following accelerator GEMM's per-64 sums (a 128-wide head is two quant groups).
    const float low = simd_sum(rounded.x + rounded.y), high = simd_sum(rounded.z + rounded.w);
    if (lane == 0u) {
        coreSums[row * (valueDim / 64u) + h * 2u] = low;
        coreSums[row * (valueDim / 64u) + h * 2u + 1u] = high;
    }
}

kernel void sp_gdn_history(SP_GDN_ARGS,
                           uint3 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]])
{
    const uint h = group.x;
    if (h >= p.valueHeads || group.y >= p.runs || tid >= 384u) return;
    const uint start = runs[group.y * 3u + 1u], length = runs[group.y * 3u + 2u];
    const uint d = p.headDim;
    const uint keyDim = p.keyHeads * d;
    const uint hk = h / (p.valueHeads / p.keyHeads);
    const uint convPerUnit = p.valueHeads * 3u * d * 3u;
    const uint kind = tid / d, jc = tid % d;
    const uint channel = kind == 0u ? hk * d + jc : kind == 1u ? keyDim + hk * d + jc : 2u * keyDim + h * d + jc;
    const uint convInUnit = ((h * 3u + kind) * d + jc) * 3u;
    const uint first = rowRead[start] * convPerUnit + convInUnit;
    const float3 history = float3(convState[first], convState[first + 1u], convState[first + 2u]);
    // For every unit a row writes: the last three raw inputs up to that row.
    for (uint t = 0; t < length; ++t) {
        const uint row = start + t;
        if (t + 1u != length && rowWrite[row + 1u] == rowWrite[row]) continue;
        const uint mb = row * p.channels + channel;
        const float x1 = mixed[mb] * rowInv[row];
        const float x2 = t >= 1u ? mixed[mb - p.channels] * rowInv[row - 1u] : history.z;
        const float x3 = t >= 2u ? mixed[mb - 2u * p.channels] * rowInv[row - 2u] : (t == 1u ? history.z : history.y);
        const uint conv = rowWrite[row] * convPerUnit + convInUnit;
        convState[conv] = x3; convState[conv + 1u] = x2; convState[conv + 2u] = x1;
    }
}

// Apply a slot's accepted journal entries to its stored state, outside a step (before the state
// is exported). Threadgroup grid: (valueHeads, 1), 1024 threads, laid out as in sp_gdn_fast.
struct SpGdnCommitParams { uint slot; uint pending; uint valueHeads; uint headDim; };
kernel void sp_gdn_commit(device float *state [[buffer(0)]],
                          device const float *journal [[buffer(1)]],
                          constant SpGdnCommitParams &p [[buffer(2)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint h = group.x, d = p.headDim;
    if (h >= p.valueHeads) return;
    const uint part = lane & 7u, j = sg * 4u + (lane >> 3);
    device packed_float4 *stored = (device packed_float4 *)(state + p.slot * p.valueHeads * d * d + ((h * d) + j) * d + part * 16u);
    float4 s0 = float4(stored[0]), s1 = float4(stored[1]), s2 = float4(stored[2]), s3 = float4(stored[3]);
    device const float *journalHead = journal + (p.slot * SP_GDN_JOURNAL_ROWS * p.valueHeads + h) * SP_GDN_JOURNAL_WIDTH;
    for (uint e = 0; e < p.pending; ++e) {
        device const float *entry = journalHead + e * p.valueHeads * SP_GDN_JOURNAL_WIDTH;
        device const packed_float4 *ks = (device const packed_float4 *)(entry + part * 16u);
        const float delta = entry[d + j], decay = entry[2u * d];
        s0 = sp_gdn_update(s0, decay, float4(ks[0]), delta); s1 = sp_gdn_update(s1, decay, float4(ks[1]), delta);
        s2 = sp_gdn_update(s2, decay, float4(ks[2]), delta); s3 = sp_gdn_update(s3, decay, float4(ks[3]), delta);
    }
    stored[0] = s0; stored[1] = s1; stored[2] = s2; stored[3] = s3;
}

// Copy `rows` rows of `dim` floats into a wider destination row at a column offset.
// Thread grid: (dim, rows).
struct SpCopyParams { uint rows; uint dim; uint dstStride; uint dstOffset; };
kernel void sp_copy_rows(device const float *src [[buffer(0)]],
                         device float *dst [[buffer(1)]],
                         constant SpCopyParams &p [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.dim || gid.y >= p.rows) return;
    dst[gid.y * p.dstStride + p.dstOffset + gid.x] = src[gid.y * p.dim + gid.x];
}

// Attention with the query preparation (RMSNorm + RoPE) folded in. Same structure as
// sp_attention; the normalised, rotated query lives in threadgroup memory.
// Threadgroup grid: (heads, rows), 256 threads per group.
struct SpAttnFusedParams {
    uint rows;
    uint heads;
    uint kvHeads;
    uint maxPages;
    uint rotary;
    float scale;
    float eps;
    float theta;
};

kernel void sp_attention_fused(device const float *qproj [[buffer(0)]],
                               device const ushort *normWeight [[buffer(1)]],
                               device const uint *rowSlot [[buffer(2)]],
                               device const uint *rowPos [[buffer(3)]],
                               device const uint *pageTable [[buffer(4)]],
                               device const half *kpool [[buffer(5)]],
                               device const half *vpool [[buffer(6)]],
                               device float *out [[buffer(7)]],
                               constant SpAttnFusedParams &p [[buffer(8)]],
                               device bfloat *outB [[buffer(9)]],
                               device float *outSums [[buffer(10)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint j [[thread_index_in_threadgroup]])
{
    threadgroup float shared[SP_PAGE_TOKENS];
    threadgroup float query[SP_HEAD_DIM];

    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    const uint inBase = (row * p.heads + head) * (2u * SP_HEAD_DIM);
    const uint position = rowPos[row];

    shared[j] = qproj[inBase + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    for (uint i = 0; i < SP_HEAD_DIM; ++i) ss += shared[i] * shared[i];
    const float inv = rsqrt(ss / float(SP_HEAD_DIM) + p.eps);
    float qv = shared[j] * inv * sp_bf16(normWeight[j]);
    if (j < p.rotary) {
        const uint half_ = p.rotary >> 1;
        const uint pj = j < half_ ? j + half_ : j - half_;
        qv = sp_rope(qv, shared[pj] * inv * sp_bf16(normWeight[pj]), j, p.rotary, position, p.theta);
    }
    query[j] = qv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kvHead = head / (p.heads / p.kvHeads);
    const uint visible = position + 1u;
    const uint pages = (visible + SP_PAGE_TOKENS - 1u) / SP_PAGE_TOKENS;
    const uint tableBase = rowSlot[row] * p.maxPages;

    float runMax = -INFINITY, runSum = 0.0f, acc = 0.0f;
    for (uint pg = 0; pg < pages; ++pg) {
        const uint pageBase = pageTable[tableBase + pg] * SP_PAGE_TOKENS;
        float score = -INFINITY;
        if (pg * SP_PAGE_TOKENS + j < visible) {
            const uint kBase = ((pageBase + j) * p.kvHeads + kvHead) * SP_HEAD_DIM;
            float dsum = 0.0f;
            for (uint i = 0; i < SP_HEAD_DIM; ++i) dsum += query[i] * float(kpool[kBase + i]);
            score = dsum * p.scale;
        }
        shared[j] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float pageMax = -INFINITY;
        for (uint t = 0; t < SP_PAGE_TOKENS; ++t) pageMax = max(pageMax, shared[t]);
        const float newMax = max(runMax, pageMax);
        const float rescale = exp(runMax - newMax);
        acc *= rescale;
        runSum *= rescale;
        runMax = newMax;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared[j] = exp(score - runMax);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0; t < SP_PAGE_TOKENS; ++t) {
            const float prob = shared[t];
            if (prob > 0.0f) {
                runSum += prob;
                acc += prob * float(vpool[((pageBase + t) * p.kvHeads + kvHead) * SP_HEAD_DIM + j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint index = (row * p.heads + head) * SP_HEAD_DIM + j;
    const float result = (acc / runSum) * sp_sigmoid(qproj[inBase + SP_HEAD_DIM + j]);
    out[index] = result;
    // bf16 operand and per-64 sums for the following accelerator GEMM (a head is four groups).
    const bfloat rounded = bfloat(result);
    outB[index] = rounded;
    shared[j] = float(rounded);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((j & 63u) == 0u) {
        float total = 0.0f;
        for (uint i = 0; i < 64u; ++i) total += shared[j + i];
        outSums[row * (p.heads * SP_HEAD_DIM / 64u) + (head * SP_HEAD_DIM + j) / 64u] = total;
    }
}

// =============================================================================================
// Fixed-width GEMM variants.
//
// The row count per pass is a compile-time constant here so the accumulators are registers and
// the row loop unrolls. The host picks the width and pads the last block; padded rows read
// valid scratch memory and their results are discarded by the bounds check on the write.

#define SP_DEFINE_GEMM(NAME, R)                                                                    \
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
    const uint words = p.inner >> 3;                                                               \
    const uint wordBase = n * p.strideWords;                                                       \
    const uint groupBase = n * p.groupsPerRow;                                                     \
    for (uint w = lane; w < words; w += SP_LANES) {                                                \
        const uint word = packed[wordBase + w];                                                    \
        const float s = sp_bf16(scales[groupBase + (w >> 3)]);                                     \
        const float b = sp_bf16(biases[groupBase + (w >> 3)]);                                     \
        const float4 w0 = s * float4((uint4(word) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + b;   \
        const float4 w1 = s * float4((uint4(word) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + b;\
        const uint k = w << 3;                                                                     \
        for (uint m = 0; m < R; ++m) {                                                             \
            device const packed_float4 *ap =                                                       \
                (device const packed_float4 *)(a + (r0 + m) * p.inStride + k);                     \
            acc[m] += dot(float4(ap[0]), w0) + dot(float4(ap[1]), w1);                             \
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

SP_DEFINE_GEMM(sp_gemm_q4_r1, 1u)
SP_DEFINE_GEMM(sp_gemm_q4_r2, 2u)
SP_DEFINE_GEMM(sp_gemm_q4_r4, 4u)
SP_DEFINE_GEMM(sp_gemm_q4_r8, 8u)

#define SP_DEFINE_MLP_IN(NAME, R)                                                                  \
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
                 uint lane [[thread_index_in_threadgroup]])                                        \
{                                                                                                  \
    const uint n = group.x;                                                                        \
    const uint r0 = group.y * R;                                                                   \
    if (n >= p.outDim || r0 >= p.rows) return;                                                     \
    float accG[R], accU[R];                                                                        \
    for (uint m = 0; m < R; ++m) { accG[m] = 0.0f; accU[m] = 0.0f; }                               \
    const uint words = p.inner >> 3;                                                               \
    const uint wordBase = n * p.strideWords;                                                       \
    const uint groupBase = n * p.groupsPerRow;                                                     \
    for (uint w = lane; w < words; w += SP_LANES) {                                                \
        const uint gw = gatePacked[wordBase + w];                                                  \
        const uint uw = upPacked[wordBase + w];                                                    \
        const uint gi = groupBase + (w >> 3);                                                      \
        const float gs = sp_bf16(gateScales[gi]), gb = sp_bf16(gateBiases[gi]);                    \
        const float us = sp_bf16(upScales[gi]), ub = sp_bf16(upBiases[gi]);                        \
        const float4 g0 = gs * float4((uint4(gw) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + gb;   \
        const float4 g1 = gs * float4((uint4(gw) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + gb;\
        const float4 u0 = us * float4((uint4(uw) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + ub;   \
        const float4 u1 = us * float4((uint4(uw) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + ub;\
        const uint k = w << 3;                                                                     \
        for (uint m = 0; m < R; ++m) {                                                             \
            device const packed_float4 *ap =                                                       \
                (device const packed_float4 *)(a + (r0 + m) * p.inStride + k);                     \
            const float4 a0 = float4(ap[0]), a1 = float4(ap[1]);                                   \
            accG[m] += dot(a0, g0) + dot(a1, g1);                                                  \
            accU[m] += dot(a0, u0) + dot(a1, u1);                                                  \
        }                                                                                          \
    }                                                                                              \
    for (uint m = 0; m < R; ++m) {                                                                 \
        const float gate = simd_sum(accG[m]);                                                      \
        const float up = simd_sum(accU[m]);                                                        \
        if (lane == 0u && r0 + m < p.rows) out[(r0 + m) * p.outStride + n] = sp_silu(gate) * up;   \
    }                                                                                              \
}

SP_DEFINE_MLP_IN(sp_mlp_in_r1, 1u)
SP_DEFINE_MLP_IN(sp_mlp_in_r2, 2u)
SP_DEFINE_MLP_IN(sp_mlp_in_r4, 4u)
SP_DEFINE_MLP_IN(sp_mlp_in_r8, 8u)

// =============================================================================================
// Register-tiled GEMM for multi-row passes (prefill and batched decode).
//
// A lane computes an R x T tile of outputs: R activation rows against T weight rows. Each
// activation load is reused T times and each dequantised weight R times, so loads per
// multiply-add fall from ~1 to ~1/T. Threadgroup grid: (outDim / T, ceil(rows / R)).

#define SP_DEFINE_GEMM_TILE(NAME, R, T)                                                            \
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
    const uint words = p.inner >> 3;                                                               \
    for (uint w = lane; w < words; w += SP_LANES) {                                                \
        float4 w0[T], w1[T];                                                                       \
        for (uint t = 0; t < T; ++t) {                                                             \
            const uint word = packed[(n0 + t) * p.strideWords + w];                                \
            const uint gi = (n0 + t) * p.groupsPerRow + (w >> 3);                                  \
            const float s = sp_bf16(scales[gi]), b = sp_bf16(biases[gi]);                          \
            w0[t] = s * float4((uint4(word) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + b;         \
            w1[t] = s * float4((uint4(word) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + b;      \
        }                                                                                          \
        const uint k = w << 3;                                                                     \
        for (uint m = 0; m < R; ++m) {                                                             \
            device const packed_float4 *ap =                                                       \
                (device const packed_float4 *)(a + (r0 + m) * p.inStride + k);                     \
            const float4 a0 = float4(ap[0]), a1 = float4(ap[1]);                                   \
            for (uint t = 0; t < T; ++t) acc[t][m] += dot(a0, w0[t]) + dot(a1, w1[t]);             \
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

SP_DEFINE_GEMM_TILE(sp_gemm_q4_r8t2, 8u, 2u)
SP_DEFINE_GEMM_TILE(sp_gemm_q4_r8t4, 8u, 4u)
SP_DEFINE_GEMM_TILE(sp_gemm_q4_r8t8, 8u, 8u)
SP_DEFINE_GEMM_TILE(sp_gemm_q4_r16t4, 16u, 4u)

#define SP_DEFINE_MLP_TILE(NAME, R, T)                                                             \
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
                 uint lane [[thread_index_in_threadgroup]])                                        \
{                                                                                                  \
    const uint n0 = group.x * T;                                                                   \
    const uint r0 = group.y * R;                                                                   \
    if (n0 >= p.outDim || r0 >= p.rows) return;                                                    \
    float accG[T][R], accU[T][R];                                                                  \
    for (uint t = 0; t < T; ++t) for (uint m = 0; m < R; ++m) { accG[t][m] = 0.0f; accU[t][m] = 0.0f; } \
    const uint words = p.inner >> 3;                                                               \
    for (uint w = lane; w < words; w += SP_LANES) {                                                \
        float4 g0[T], g1[T], u0[T], u1[T];                                                         \
        for (uint t = 0; t < T; ++t) {                                                             \
            const uint gw = gatePacked[(n0 + t) * p.strideWords + w];                              \
            const uint uw = upPacked[(n0 + t) * p.strideWords + w];                                \
            const uint gi = (n0 + t) * p.groupsPerRow + (w >> 3);                                  \
            const float gs = sp_bf16(gateScales[gi]), gb = sp_bf16(gateBiases[gi]);                \
            const float us = sp_bf16(upScales[gi]), ub = sp_bf16(upBiases[gi]);                    \
            g0[t] = gs * float4((uint4(gw) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + gb;         \
            g1[t] = gs * float4((uint4(gw) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + gb;      \
            u0[t] = us * float4((uint4(uw) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + ub;         \
            u1[t] = us * float4((uint4(uw) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + ub;      \
        }                                                                                          \
        const uint k = w << 3;                                                                     \
        for (uint m = 0; m < R; ++m) {                                                             \
            device const packed_float4 *ap =                                                       \
                (device const packed_float4 *)(a + (r0 + m) * p.inStride + k);                     \
            const float4 a0 = float4(ap[0]), a1 = float4(ap[1]);                                   \
            for (uint t = 0; t < T; ++t) {                                                         \
                accG[t][m] += dot(a0, g0[t]) + dot(a1, g1[t]);                                     \
                accU[t][m] += dot(a0, u0[t]) + dot(a1, u1[t]);                                     \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
    for (uint t = 0; t < T; ++t) {                                                                 \
        for (uint m = 0; m < R; ++m) {                                                             \
            const float gate = simd_sum(accG[t][m]);                                               \
            const float up = simd_sum(accU[t][m]);                                                 \
            if (lane == 0u && r0 + m < p.rows) out[(r0 + m) * p.outStride + n0 + t] = sp_silu(gate) * up; \
        }                                                                                          \
    }                                                                                              \
}

SP_DEFINE_MLP_TILE(sp_mlp_in_r8t2, 8u, 2u)
SP_DEFINE_MLP_TILE(sp_mlp_in_r8t4, 8u, 4u)
SP_DEFINE_MLP_TILE(sp_mlp_in_r16t4, 16u, 4u)

// Arithmetic probes: register-only FMA chains, no memory traffic. Thread grid: (threads).
struct SpAluParams { uint threads; uint iterations; };
kernel void sp_probe_alu_f32(device float *out [[buffer(0)]], constant SpAluParams &p [[buffer(1)]],
                             uint gid [[thread_position_in_grid]])
{
    if (gid >= p.threads) return;
    float4 a = float4(float(gid) * 1e-6f + 1.0f), b = float4(0.999f), c = float4(0.001f), d = float4(0.5f);
    for (uint i = 0; i < p.iterations; ++i) { a = fma(a, b, c); d = fma(d, b, a); }
    out[gid] = a.x + d.y;
}
// Scheduling probe: every thread of every threadgroup spins `iterations` FMA rounds, with
// `barriers` threadgroup barriers spread through them. Any grid, any threadgroup size: how long
// a dispatch takes against its total work says how many cores it was spread over.
struct SpSpinParams { uint iterations; uint barriers; };
kernel void sp_probe_spin(device float *out [[buffer(0)]], constant SpSpinParams &p [[buffer(1)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]])
{
    float4 a = float4(float(tid + group.x) * 1e-6f + 1.0f), b = float4(0.999f), c = float4(0.001f), d = float4(0.5f);
    const uint rounds = max(p.barriers, 1u), each = p.iterations / rounds;
    for (uint r = 0; r < rounds; ++r) {
        for (uint i = 0; i < each; ++i) { a = fma(a, b, c); d = fma(d, b, a); }
        if (p.barriers != 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) out[(group.y * 4096u + group.x) % 65536u] = a.x + d.y;
}
kernel void sp_probe_alu_f16(device float *out [[buffer(0)]], constant SpAluParams &p [[buffer(1)]],
                             uint gid [[thread_position_in_grid]])
{
    if (gid >= p.threads) return;
    half4 a = half4(half(gid & 1023u) * 1e-3h + 1.0h), b = half4(0.999h), c = half4(0.001h), d = half4(0.5h);
    for (uint i = 0; i < p.iterations; ++i) { a = fma(a, b, c); d = fma(d, b, a); }
    out[gid] = float(a.x + d.y);
}

// =============================================================================================
// Matrix-tiled q4 GEMM for multi-row passes.
//
// One 128-thread group (four simdgroups) computes a 32-row x 32-column output tile. Per step of
// 32 along the contracted axis, every thread dequantises one packed word into threadgroup
// memory (32 weight rows x 4 words), then each simdgroup runs 8x8 matrix multiply-accumulates:
// activations [8 rows x 8 k] times weights [8 k x 8 columns].
//
// Rows past `rows` in the last block read scratch padding and write scratch padding; the host
// sizes every activation buffer to a multiple of 32 rows.
// Threadgroup grid: (outDim / 32, ceil(rows / 32)), 128 threads per group.

#define SP_TILE 32u

kernel void sp_gemm_q4_mm(device const uint *packed [[buffer(0)]],
                          device const ushort *scales [[buffer(1)]],
                          device const ushort *biases [[buffer(2)]],
                          device const float *a [[buffer(3)]],
                          device float *out [[buffer(4)]],
                          device const float *residual [[buffer(5)]],
                          constant SpGemmParams &p [[buffer(6)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float wtile[SP_TILE * SP_TILE];

    const uint n0 = group.x * SP_TILE;
    const uint r0 = group.y * SP_TILE;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    const uint rowBlocks = min(4u, (p.rows - r0 + 7u) / 8u);

    // This simdgroup owns output columns n0 + sg*8 ..< +8 for every row block.
    simdgroup_float8x8 acc[4];
    for (uint rb = 0; rb < 4u; ++rb) {
        if (p.hasResidual != 0u && rb < rowBlocks) {
            simdgroup_load(acc[rb], residual + (r0 + rb * 8u) * p.outStride + n0 + sg * 8u, p.outStride);
        } else {
            acc[rb] = simdgroup_float8x8(0.0f);
        }
    }

    // Thread tid dequantises word (tid % 4) of weight row n0 + tid / 4.
    const uint wrow = n0 + (tid >> 2);
    const uint wsub = tid & 3u;
    const uint wordBase = wrow * p.strideWords + wsub;
    const uint groupBase = wrow * p.groupsPerRow;
    const uint steps = p.inner / SP_TILE;
    for (uint step = 0; step < steps; ++step) {
        const uint word = packed[wordBase + step * 4u];
        const uint gi = groupBase + (step >> 1);
        const float s = sp_bf16(scales[gi]), b = sp_bf16(biases[gi]);
        const float4 w0 = s * float4((uint4(word) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + b;
        const float4 w1 = s * float4((uint4(word) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + b;
        threadgroup float *dst = wtile + (tid >> 2) * SP_TILE + wsub * 8u;
        dst[0] = w0.x; dst[1] = w0.y; dst[2] = w0.z; dst[3] = w0.w;
        dst[4] = w1.x; dst[5] = w1.y; dst[6] = w1.z; dst[7] = w1.w;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint k = step * SP_TILE;
        for (uint kb = 0; kb < 4u; ++kb) {
            // wtile is [column][k]; transposed, the 8x8 block is [k][column].
            simdgroup_float8x8 wm;
            simdgroup_load(wm, wtile, SP_TILE, ulong2(kb * 8u, sg * 8u), true);
            for (uint rb = 0; rb < rowBlocks; ++rb) {
                simdgroup_float8x8 am;
                simdgroup_load(am, a + (r0 + rb * 8u) * p.inStride + k + kb * 8u, p.inStride);
                simdgroup_multiply_accumulate(acc[rb], am, wm, acc[rb]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint rb = 0; rb < rowBlocks; ++rb) {
        simdgroup_store(acc[rb], out + (r0 + rb * 8u) * p.outStride + n0 + sg * 8u, p.outStride);
    }
}

// simdgroup matrix probes: register-only 8x8 multiply-accumulate chains.
// Threadgroup grid: (threads / 32), 32 threads per group.
kernel void sp_probe_mm_f32(device float *out [[buffer(0)]], constant SpAluParams &p [[buffer(1)]],
                            uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]])
{
    simdgroup_float8x8 x = simdgroup_float8x8(0.5f), y = simdgroup_float8x8(0.001f);
    simdgroup_float8x8 c0 = simdgroup_float8x8(0.0f), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < p.iterations; ++i) {
        simdgroup_multiply_accumulate(c0, x, y, c0);
        simdgroup_multiply_accumulate(c1, x, y, c1);
        simdgroup_multiply_accumulate(c2, x, y, c2);
        simdgroup_multiply_accumulate(c3, x, y, c3);
    }
    simdgroup_multiply_accumulate(c0, c1, c2, c3);
    simdgroup_store(c0, out + group.x * 64u, 8u);
}
kernel void sp_probe_mm_f16(device half *out [[buffer(0)]], constant SpAluParams &p [[buffer(1)]],
                            uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]])
{
    simdgroup_half8x8 x = simdgroup_half8x8(0.5h), y = simdgroup_half8x8(0.001h);
    simdgroup_half8x8 c0 = simdgroup_half8x8(0.0h), c1 = c0, c2 = c0, c3 = c0;
    for (uint i = 0; i < p.iterations; ++i) {
        simdgroup_multiply_accumulate(c0, x, y, c0);
        simdgroup_multiply_accumulate(c1, x, y, c1);
        simdgroup_multiply_accumulate(c2, x, y, c2);
        simdgroup_multiply_accumulate(c3, x, y, c3);
    }
    simdgroup_multiply_accumulate(c0, c1, c2, c3);
    simdgroup_store(c0, out + group.x * 64u, 8u);
}
// Independent-chain scalar FMA probe (throughput rather than latency bound).
kernel void sp_probe_alu_wide(device float *out [[buffer(0)]], constant SpAluParams &p [[buffer(1)]],
                              uint gid [[thread_position_in_grid]])
{
    if (gid >= p.threads) return;
    float4 a0 = float4(1.0f), a1 = float4(1.1f), a2 = float4(1.2f), a3 = float4(1.3f);
    float4 a4 = float4(1.4f), a5 = float4(1.5f), a6 = float4(1.6f), a7 = float4(1.7f);
    const float4 b = float4(0.999f), c = float4(float(gid) * 1e-9f);
    for (uint i = 0; i < p.iterations; ++i) {
        a0 = fma(a0, b, c); a1 = fma(a1, b, c); a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c); a6 = fma(a6, b, c); a7 = fma(a7, b, c);
    }
    out[gid] = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7).x;
}

// RMSNorm that also emits the accelerator operand: the bf16 copy of its output and the per-64
// sums of that copy. Saves the separate conversion stage before a GEMM.
// Threadgroup grid: (count), 32 threads per group.
// The same norm with a row spread over dim / 512 simdgroups instead of walked by one: each
// takes 512 elements (eight quant groups), the squares meet in threadgroup memory. For wide
// steps, where one simdgroup a row left most of the GPU idle (38 us a dispatch, 128 a step).
// Threadgroup grid: (rows), dim / 512 * 32 threads.
kernel void sp_rmsnorm_wide(device const float *input [[buffer(0)]],
                            device const ushort *weight [[buffer(1)]],
                            device float *output [[buffer(2)]],
                            constant SpNormParams &p [[buffer(3)]],
                            device bfloat *outputB [[buffer(4)]],
                            device float *sums [[buffer(5)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float squares[32];
    if (group.x >= p.count) return;
    const uint base = group.x * p.dim, first = sg * 512u;
    float partial = 0.0f;
    for (uint g = 0; g < 8u; ++g) {
        const float x0 = input[base + first + g * 64u + lane], x1 = input[base + first + g * 64u + lane + 32u];
        partial += x0 * x0 + x1 * x1;
    }
    const float mine = simd_sum(partial);
    if (lane == 0u) squares[sg] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < p.dim / 512u; ++s) total += squares[s];
    const float inv = rsqrt(total / float(p.dim) + p.eps);
    for (uint g = 0; g < 8u; ++g) {
        const uint i0 = first + g * 64u + lane, i1 = i0 + 32u;
        const float w0 = sp_bf16(weight[i0]), w1 = sp_bf16(weight[i1]);
        const float y0 = input[base + i0] * inv * (p.mode == 0u ? 1.0f + w0 : w0);
        const float y1 = input[base + i1] * inv * (p.mode == 0u ? 1.0f + w1 : w1);
        // The fp32 copy is read too (by the gated-delta and attention kernels' own inputs).
        output[base + i0] = y0;
        output[base + i1] = y1;
        const bfloat b0 = bfloat(y0), b1 = bfloat(y1);
        outputB[base + i0] = b0;
        outputB[base + i1] = b1;
        const float groupSum = simd_sum(float(b0) + float(b1));
        if (lane == 0u) sums[group.x * (p.dim / 64u) + sg * 8u + g] = groupSum;
    }
}

kernel void sp_rmsnorm_na(device const float *input [[buffer(0)]],
                          device const ushort *weight [[buffer(1)]],
                          device float *output [[buffer(2)]],
                          constant SpNormParams &p [[buffer(3)]],
                          device bfloat *outputB [[buffer(4)]],
                          device float *sums [[buffer(5)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= p.count) return;
    const uint base = group.x * p.dim;
    float partial = 0.0f;
    for (uint i = lane; i < p.dim; i += SP_LANES) {
        const float x = input[base + i];
        partial += x * x;
    }
    const float inv = rsqrt(simd_sum(partial) / float(p.dim) + p.eps);
    const uint groups = p.dim / 64u;
    for (uint g = 0; g < groups; ++g) {
        const uint i0 = g * 64u + lane, i1 = i0 + 32u;
        const float w0 = sp_bf16(weight[i0]), w1 = sp_bf16(weight[i1]);
        const float y0 = input[base + i0] * inv * (p.mode == 0u ? 1.0f + w0 : w0);
        const float y1 = input[base + i1] * inv * (p.mode == 0u ? 1.0f + w1 : w1);
        output[base + i0] = y0;
        output[base + i1] = y1;
        const bfloat b0 = bfloat(y0), b1 = bfloat(y1);
        outputB[base + i0] = b0;
        outputB[base + i1] = b1;
        const float total = simd_sum(float(b0) + float(b1));
        if (lane == 0u) sums[group.x * groups + g] = total;
    }
}

// =============================================================================================
// 4-bit KV cache.
//
// Each K or V vector (256 values per token per KV head) is stored as a 144-byte record: 128
// bytes of 4-bit codes in four groups of 64 values, then four (scale, bias) fp16 pairs.
// value = scale[group] * code + bias[group].
//
// Within a group, byte r (0..31) holds element r in its low nibble and element r + 32 in its
// high nibble, so each of the 32 lanes that write a record owns whole bytes.
// Pool layout: [page][token][kvHead][144 bytes].

#define SP_KV_RECORD 144u

inline void sp_kv_quantize(float e0, float e1, device uchar *record, uint group_, uint lane) {
    const float lo = simd_min(min(e0, e1));
    const float hi = simd_max(max(e0, e1));
    const half scaleH = half(hi > lo ? (hi - lo) / 15.0f : 1.0f);
    const half biasH = half(lo);
    const float scale = float(scaleH), bias = float(biasH);
    const uint c0 = uint(clamp(round((e0 - bias) / scale), 0.0f, 15.0f));
    const uint c1 = uint(clamp(round((e1 - bias) / scale), 0.0f, 15.0f));
    record[group_ * 32u + lane] = uchar(c0 | (c1 << 4));
    if (lane == 0u) {
        device half *meta = (device half *)(record + 128u + group_ * 4u);
        meta[0] = scaleH;
        meta[1] = biasH;
    }
}

// k: RMSNorm + RoPE, v: verbatim; both quantised into the paged pool.
// Threadgroup grid: (kvHeads, rows), 32 threads per group.
kernel void sp_attn_kv_store_q4(device const float *kproj [[buffer(0)]],
                                device const float *vproj [[buffer(1)]],
                                device const ushort *normWeight [[buffer(2)]],
                                device const uint *rowSlot [[buffer(3)]],
                                device const uint *rowPos [[buffer(4)]],
                                device const uint *pageTable [[buffer(5)]],
                                device uchar *kpool [[buffer(6)]],
                                device uchar *vpool [[buffer(7)]],
                                constant SpAttnPrepParams &p [[buffer(8)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]])
{
    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    const uint base = (row * p.heads + head) * SP_HEAD_DIM;
    float partial = 0.0f;
    for (uint i = lane; i < SP_HEAD_DIM; i += SP_LANES) {
        const float x = kproj[base + i];
        partial += x * x;
    }
    const float inv = rsqrt(simd_sum(partial) / float(SP_HEAD_DIM) + p.eps);
    const uint position = rowPos[row];
    const uint page = pageTable[rowSlot[row] * p.maxPages + position / SP_PAGE_TOKENS];
    const uint record = ((page * SP_PAGE_TOKENS + position % SP_PAGE_TOKENS) * p.heads + head) * SP_KV_RECORD;
    const uint half_ = p.rotary >> 1;
    for (uint g = 0; g < 4u; ++g) {
        float ke[2], ve[2];
        for (uint s = 0; s < 2u; ++s) {
            const uint i = g * 64u + s * 32u + lane;
            float x = kproj[base + i] * inv * sp_bf16(normWeight[i]);
            if (i < p.rotary) {
                const uint pi = i < half_ ? i + half_ : i - half_;
                x = sp_rope(x, kproj[base + pi] * inv * sp_bf16(normWeight[pi]), i, p.rotary, position, p.theta);
            }
            ke[s] = x;
            ve[s] = vproj[base + i];
        }
        sp_kv_quantize(ke[0], ke[1], kpool + record, g, lane);
        sp_kv_quantize(ve[0], ve[1], vpool + record, g, lane);
    }
}

// Attention over the 4-bit pool. Scores are taken on the codes directly:
//   q . k = sum_g scale_g * (q_g . code_g) + bias_g * sum(q_g)
// Otherwise identical to sp_attention_fused.
// Threadgroup grid: (heads, rows), 256 threads per group.
kernel void sp_attention_fused_q4(device const float *qproj [[buffer(0)]],
                                  device const ushort *normWeight [[buffer(1)]],
                                  device const uint *rowSlot [[buffer(2)]],
                                  device const uint *rowPos [[buffer(3)]],
                                  device const uint *pageTable [[buffer(4)]],
                                  device const uchar *kpool [[buffer(5)]],
                                  device const uchar *vpool [[buffer(6)]],
                                  device float *out [[buffer(7)]],
                                  constant SpAttnFusedParams &p [[buffer(8)]],
                                  device bfloat *outB [[buffer(9)]],
                                  device float *outSums [[buffer(10)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint j [[thread_index_in_threadgroup]])
{
    threadgroup float shared[SP_PAGE_TOKENS];
    threadgroup float query[SP_HEAD_DIM];

    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    const uint inBase = (row * p.heads + head) * (2u * SP_HEAD_DIM);
    const uint position = rowPos[row];

    shared[j] = qproj[inBase + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    for (uint i = 0; i < SP_HEAD_DIM; ++i) ss += shared[i] * shared[i];
    const float inv = rsqrt(ss / float(SP_HEAD_DIM) + p.eps);
    float qv = shared[j] * inv * sp_bf16(normWeight[j]);
    if (j < p.rotary) {
        const uint half_ = p.rotary >> 1;
        const uint pj = j < half_ ? j + half_ : j - half_;
        qv = sp_rope(qv, shared[pj] * inv * sp_bf16(normWeight[pj]), j, p.rotary, position, p.theta);
    }
    query[j] = qv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 querySums = float4(0.0f);
    for (uint i = 0; i < 64u; ++i) {
        querySums += float4(query[i], query[64u + i], query[128u + i], query[192u + i]);
    }

    const uint kvHead = head / (p.heads / p.kvHeads);
    const uint visible = position + 1u;
    const uint pages = (visible + SP_PAGE_TOKENS - 1u) / SP_PAGE_TOKENS;
    const uint tableBase = rowSlot[row] * p.maxPages;
    // This thread's channel inside a V record.
    const uint vGroup = j >> 6, vByte = (j >> 6) * 32u + (j & 31u), vShift = ((j >> 5) & 1u) * 4u;

    float runMax = -INFINITY, runSum = 0.0f, acc = 0.0f;
    for (uint pg = 0; pg < pages; ++pg) {
        const uint pageBase = pageTable[tableBase + pg] * SP_PAGE_TOKENS;
        float score = -INFINITY;
        if (pg * SP_PAGE_TOKENS + j < visible) {
            device const uchar *record = kpool + ((pageBase + j) * p.kvHeads + kvHead) * SP_KV_RECORD;
            float total = 0.0f;
            for (uint g = 0; g < 4u; ++g) {
                float d = 0.0f;
                for (uint r = 0; r < 32u; ++r) {
                    const uint code = record[g * 32u + r];
                    d += query[g * 64u + r] * float(code & 0xFu) + query[g * 64u + 32u + r] * float(code >> 4);
                }
                device const half *meta = (device const half *)(record + 128u + g * 4u);
                total += float(meta[0]) * d + float(meta[1]) * querySums[g];
            }
            score = total * p.scale;
        }
        shared[j] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float pageMax = -INFINITY;
        for (uint t = 0; t < SP_PAGE_TOKENS; ++t) pageMax = max(pageMax, shared[t]);
        const float newMax = max(runMax, pageMax);
        const float rescale = exp(runMax - newMax);
        acc *= rescale;
        runSum *= rescale;
        runMax = newMax;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared[j] = exp(score - runMax);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0; t < SP_PAGE_TOKENS; ++t) {
            const float prob = shared[t];
            if (prob > 0.0f) {
                runSum += prob;
                device const uchar *record = vpool + ((pageBase + t) * p.kvHeads + kvHead) * SP_KV_RECORD;
                device const half *meta = (device const half *)(record + 128u + vGroup * 4u);
                acc += prob * (float(meta[0]) * float((uint(record[vByte]) >> vShift) & 0xFu) + float(meta[1]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint index = (row * p.heads + head) * SP_HEAD_DIM + j;
    const float result = (acc / runSum) * sp_sigmoid(qproj[inBase + SP_HEAD_DIM + j]);
    out[index] = result;
    const bfloat rounded = bfloat(result);
    outB[index] = rounded;
    shared[j] = float(rounded);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((j & 63u) == 0u) {
        float total = 0.0f;
        for (uint i = 0; i < 64u; ++i) total += shared[j + i];
        outSums[row * (p.heads * SP_HEAD_DIM / 64u) + (head * SP_HEAD_DIM + j) / 64u] = total;
    }
}

// =============================================================================================
// Lane kernels for the tiled weight layout ([tile of 128 rows][quant group][row][32 bytes]).
//
// One thread per output row. Thirty-two adjacent rows form a simdgroup, and for a given quant
// group their 32-byte chunks are adjacent, so each step of the group loop reads one contiguous
// kibibyte across the lanes. No reduction is needed: a lane owns its row's whole sum.
// Thread grid: (outDim, ceil(rows / R)).

#define SP_DEFINE_GEMM_TILED(NAME, R)                                                              \
kernel void NAME(device const uint *packed [[buffer(0)]],                                          \
                 device const ushort *scales [[buffer(1)]],                                        \
                 device const ushort *biases [[buffer(2)]],                                        \
                 device const float *a [[buffer(3)]],                                              \
                 device float *out [[buffer(4)]],                                                  \
                 device const float *residual [[buffer(5)]],                                       \
                 constant SpGemmParams &p [[buffer(6)]],                                           \
                 uint2 gid [[thread_position_in_grid]])                                            \
{                                                                                                  \
    const uint n = gid.x;                                                                          \
    const uint r0 = gid.y * R;                                                                     \
    if (n >= p.outDim || r0 >= p.rows) return;                                                     \
    float acc[R];                                                                                  \
    for (uint m = 0; m < R; ++m) acc[m] = 0.0f;                                                    \
    const uint tileBase = (n >> 7) * p.groupsPerRow * 128u + (n & 127u);                           \
    for (uint g = 0; g < p.groupsPerRow; ++g) {                                                    \
        const uint record = tileBase + g * 128u;                                                   \
        const float s = sp_bf16(scales[record]), b = sp_bf16(biases[record]);                      \
        device const packed_uint4 *wp = (device const packed_uint4 *)(packed + record * 8u);       \
        const uint4 lo = uint4(wp[0]), hi = uint4(wp[1]);                                          \
        for (uint w = 0; w < 8u; ++w) {                                                            \
            const uint word = w < 4u ? lo[w] : hi[w - 4u];                                         \
            const float4 w0 = s * float4((uint4(word) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + b;   \
            const float4 w1 = s * float4((uint4(word) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + b;\
            const uint k = g * 64u + w * 8u;                                                       \
            for (uint m = 0; m < R; ++m) {                                                         \
                device const packed_float4 *ap =                                                   \
                    (device const packed_float4 *)(a + (r0 + m) * p.inStride + k);                 \
                acc[m] += dot(float4(ap[0]), w0) + dot(float4(ap[1]), w1);                         \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
    for (uint m = 0; m < R; ++m) {                                                                 \
        if (r0 + m >= p.rows) continue;                                                            \
        const uint index = (r0 + m) * p.outStride + n;                                             \
        out[index] = p.hasResidual != 0u ? acc[m] + residual[index] : acc[m];                      \
    }                                                                                              \
}

SP_DEFINE_GEMM_TILED(sp_gemm_q4_tiled_r1, 1u)
SP_DEFINE_GEMM_TILED(sp_gemm_q4_tiled_r2, 2u)

#define SP_DEFINE_MLP_TILED(NAME, R)                                                               \
kernel void NAME(device const uint *gatePacked [[buffer(0)]],                                      \
                 device const ushort *gateScales [[buffer(1)]],                                    \
                 device const ushort *gateBiases [[buffer(2)]],                                    \
                 device const uint *upPacked [[buffer(3)]],                                        \
                 device const ushort *upScales [[buffer(4)]],                                      \
                 device const ushort *upBiases [[buffer(5)]],                                      \
                 device const float *a [[buffer(6)]],                                              \
                 device float *out [[buffer(7)]],                                                  \
                 constant SpGemmParams &p [[buffer(8)]],                                           \
                 uint2 gid [[thread_position_in_grid]])                                            \
{                                                                                                  \
    const uint n = gid.x;                                                                          \
    const uint r0 = gid.y * R;                                                                     \
    if (n >= p.outDim || r0 >= p.rows) return;                                                     \
    float accG[R], accU[R];                                                                        \
    for (uint m = 0; m < R; ++m) { accG[m] = 0.0f; accU[m] = 0.0f; }                               \
    const uint tileBase = (n >> 7) * p.groupsPerRow * 128u + (n & 127u);                           \
    for (uint g = 0; g < p.groupsPerRow; ++g) {                                                    \
        const uint record = tileBase + g * 128u;                                                   \
        const float gs = sp_bf16(gateScales[record]), gb = sp_bf16(gateBiases[record]);            \
        const float us = sp_bf16(upScales[record]), ub = sp_bf16(upBiases[record]);                \
        device const packed_uint4 *gp = (device const packed_uint4 *)(gatePacked + record * 8u);   \
        device const packed_uint4 *up_ = (device const packed_uint4 *)(upPacked + record * 8u);    \
        const uint4 glo = uint4(gp[0]), ghi = uint4(gp[1]), ulo = uint4(up_[0]), uhi = uint4(up_[1]); \
        for (uint w = 0; w < 8u; ++w) {                                                            \
            const uint gw = w < 4u ? glo[w] : ghi[w - 4u];                                         \
            const uint uw = w < 4u ? ulo[w] : uhi[w - 4u];                                         \
            const float4 g0 = gs * float4((uint4(gw) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + gb;   \
            const float4 g1 = gs * float4((uint4(gw) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + gb;\
            const float4 u0 = us * float4((uint4(uw) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + ub;   \
            const float4 u1 = us * float4((uint4(uw) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + ub;\
            const uint k = g * 64u + w * 8u;                                                       \
            for (uint m = 0; m < R; ++m) {                                                         \
                device const packed_float4 *ap =                                                   \
                    (device const packed_float4 *)(a + (r0 + m) * p.inStride + k);                 \
                const float4 a0 = float4(ap[0]), a1 = float4(ap[1]);                               \
                accG[m] += dot(a0, g0) + dot(a1, g1);                                              \
                accU[m] += dot(a0, u0) + dot(a1, u1);                                              \
            }                                                                                      \
        }                                                                                          \
    }                                                                                              \
    for (uint m = 0; m < R; ++m) {                                                                 \
        if (r0 + m < p.rows) out[(r0 + m) * p.outStride + n] = sp_silu(accG[m]) * accU[m];         \
    }                                                                                              \
}

SP_DEFINE_MLP_TILED(sp_mlp_in_tiled_r1, 1u)
SP_DEFINE_MLP_TILED(sp_mlp_in_tiled_r2, 2u)

// Split-K lane kernels for the tiled layout. A threadgroup covers 32 output rows (lane = row)
// and P simdgroups; simdgroup s sums quant groups [s * G/P, (s+1) * G/P). Within a simdgroup the
// 32 lanes still read one contiguous kibibyte per quant group. Partials meet in threadgroup
// memory. Threadgroup grid: (outDim / 32, ceil(rows / R)), P * 32 threads per group.

#define SP_DEFINE_GEMM_TILED_SPLIT(NAME, R, P)                                                     \
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
        device const packed_uint4 *wp = (device const packed_uint4 *)(packed + record * 8u);       \
        const uint4 lo = uint4(wp[0]), hi = uint4(wp[1]);                                          \
        for (uint w = 0; w < 8u; ++w) {                                                            \
            const uint word = w < 4u ? lo[w] : hi[w - 4u];                                         \
            const float4 w0 = s * float4((uint4(word) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + b;   \
            const float4 w1 = s * float4((uint4(word) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + b;\
            const uint k = g * 64u + w * 8u;                                                       \
            for (uint m = 0; m < R; ++m) {                                                         \
                device const packed_float4 *ap =                                                   \
                    (device const packed_float4 *)(a + (r0 + m) * p.inStride + k);                 \
                acc[m] += dot(float4(ap[0]), w0) + dot(float4(ap[1]), w1);                         \
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

SP_DEFINE_GEMM_TILED_SPLIT(sp_gemm_q4_tiled_r1p4, 1u, 4u)
SP_DEFINE_GEMM_TILED_SPLIT(sp_gemm_q4_tiled_r1p8, 1u, 8u)
SP_DEFINE_GEMM_TILED_SPLIT(sp_gemm_q4_tiled_r1p16, 1u, 16u)
SP_DEFINE_GEMM_TILED_SPLIT(sp_gemm_q4_tiled_r2p4, 2u, 4u)
SP_DEFINE_GEMM_TILED_SPLIT(sp_gemm_q4_tiled_r2p8, 2u, 8u)
SP_DEFINE_GEMM_TILED_SPLIT(sp_gemm_q4_tiled_r2p16, 2u, 16u)

#define SP_DEFINE_MLP_TILED_SPLIT(NAME, R, P)                                                      \
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
        device const packed_uint4 *gp = (device const packed_uint4 *)(gatePacked + record * 8u);   \
        device const packed_uint4 *up_ = (device const packed_uint4 *)(upPacked + record * 8u);    \
        const uint4 glo = uint4(gp[0]), ghi = uint4(gp[1]), ulo = uint4(up_[0]), uhi = uint4(up_[1]); \
        for (uint w = 0; w < 8u; ++w) {                                                            \
            const uint gw = w < 4u ? glo[w] : ghi[w - 4u];                                         \
            const uint uw = w < 4u ? ulo[w] : uhi[w - 4u];                                         \
            const float4 g0 = gs * float4((uint4(gw) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + gb;   \
            const float4 g1 = gs * float4((uint4(gw) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + gb;\
            const float4 u0 = us * float4((uint4(uw) >> uint4(0u, 4u, 8u, 12u)) & uint4(0xFu)) + ub;   \
            const float4 u1 = us * float4((uint4(uw) >> uint4(16u, 20u, 24u, 28u)) & uint4(0xFu)) + ub;\
            const uint k = g * 64u + w * 8u;                                                       \
            for (uint m = 0; m < R; ++m) {                                                         \
                device const packed_float4 *ap =                                                   \
                    (device const packed_float4 *)(a + (r0 + m) * p.inStride + k);                 \
                const float4 a0 = float4(ap[0]), a1 = float4(ap[1]);                               \
                accG[m] += dot(a0, g0) + dot(a1, g1);                                              \
                accU[m] += dot(a0, u0) + dot(a1, u1);                                              \
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

SP_DEFINE_MLP_TILED_SPLIT(sp_mlp_in_tiled_r1p4, 1u, 4u)
SP_DEFINE_MLP_TILED_SPLIT(sp_mlp_in_tiled_r1p8, 1u, 8u)
SP_DEFINE_MLP_TILED_SPLIT(sp_mlp_in_tiled_r1p16, 1u, 16u)
SP_DEFINE_MLP_TILED_SPLIT(sp_mlp_in_tiled_r2p4, 2u, 4u)
SP_DEFINE_MLP_TILED_SPLIT(sp_mlp_in_tiled_r2p8, 2u, 8u)
SP_DEFINE_MLP_TILED_SPLIT(sp_mlp_in_tiled_r2p16, 2u, 16u)

// =============================================================================================
// Two-phase attention over the 4-bit KV pool.
//
// Phase A (sp_attn_scan_q4): one 256-thread group per (KV head, span of pages, row). It serves
// all the query heads that share the KV head, so each key and value is read once for six
// queries, and spans run in parallel instead of one group walking the whole context. Each group
// leaves a softmax partial per query head: running max, sum, and the weighted value sum.
//
// Phase B (sp_attn_merge): one group per (head, row) combines the span partials and applies the
// output gate.
//
// Partials layout: [row][head][span] x (2 + 256) floats: max, sum, then 256 accumulators.

#define SP_GROUP_HEADS 6u
#define SP_PARTIAL 258u

struct SpAttnScanParams {
    uint rows;
    uint heads;
    uint kvHeads;
    uint maxPages;
    uint spans;         // spans per row in the partial buffer
    uint pagesPerSpan;
    float scale;
};

// Threadgroup grid: (kvHeads * spans, rows), 256 threads per group.
//
// Scoring uses all 256 threads (thread j scores token j of the page). Value accumulation uses
// the first 64: thread j owns four value channels — the two bytes 2j and 2j+1 of a record — so
// it reads one 16-bit code pair, one (scale, bias) pair and two packed probability vectors per
// token. Probabilities are stored token-major, eight floats per token, for those packed loads.
kernel void sp_attn_scan_q4(device const float *qn [[buffer(0)]],
                            device const uint *rowSlot [[buffer(1)]],
                            device const uint *rowPos [[buffer(2)]],
                            device const uint *pageTable [[buffer(3)]],
                            device const uchar *kpool [[buffer(4)]],
                            device const uchar *vpool [[buffer(5)]],
                            device float *partials [[buffer(6)]],
                            constant SpAttnScanParams &p [[buffer(7)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint j [[thread_index_in_threadgroup]])
{
    threadgroup float shared[8u * SP_PAGE_TOKENS];
    threadgroup float stat[SP_GROUP_HEADS];

    const uint kvHead = group.x % p.kvHeads;
    const uint span = group.x / p.kvHeads;
    const uint row = group.y;
    if (span >= p.spans || row >= p.rows) return;
    const uint visible = rowPos[row] + 1u;
    const uint pages = (visible + SP_PAGE_TOKENS - 1u) / SP_PAGE_TOKENS;
    const uint firstPage = span * p.pagesPerSpan;
    const uint lastPage = min(pages, firstPage + p.pagesPerSpan);
    const uint tableBase = rowSlot[row] * p.maxPages;
    const uint headBase = kvHead * SP_GROUP_HEADS;
    device const float *queries = qn + (row * p.heads + headBase) * SP_HEAD_DIM;
    const bool accumulates = j < 64u;
    const uint vGroup = j >> 4;   // bytes 2j, 2j+1 lie in group (2j) / 32

    float runMax[SP_GROUP_HEADS], runSum[SP_GROUP_HEADS];
    float4 acc[SP_GROUP_HEADS];
    for (uint h = 0; h < SP_GROUP_HEADS; ++h) { runMax[h] = -INFINITY; runSum[h] = 0.0f; acc[h] = float4(0.0f); }
    float querySum[SP_GROUP_HEADS][4];
    for (uint h = 0; h < SP_GROUP_HEADS; ++h) {
        for (uint g = 0; g < 4u; ++g) {
            device const packed_float4 *q = (device const packed_float4 *)(queries + h * SP_HEAD_DIM + g * 64u);
            float4 total = float4(0.0f);
            for (uint i = 0; i < 16u; ++i) total += float4(q[i]);
            querySum[h][g] = total.x + total.y + total.z + total.w;
        }
    }

    for (uint pg = firstPage; pg < lastPage; ++pg) {
        const uint pageBase = pageTable[tableBase + pg] * SP_PAGE_TOKENS;
        float score[SP_GROUP_HEADS];
        for (uint h = 0; h < SP_GROUP_HEADS; ++h) score[h] = -INFINITY;
        if (pg * SP_PAGE_TOKENS + j < visible) {
            device const uchar *record = kpool + ((pageBase + j) * p.kvHeads + kvHead) * SP_KV_RECORD;
            device const uint *words = (device const uint *)record;
            for (uint h = 0; h < SP_GROUP_HEADS; ++h) score[h] = 0.0f;
            for (uint g = 0; g < 4u; ++g) {
                float d[SP_GROUP_HEADS];
                for (uint h = 0; h < SP_GROUP_HEADS; ++h) d[h] = 0.0f;
                for (uint w = 0; w < 8u; ++w) {
                    // Four bytes: low nibbles are elements r..r+3, high nibbles r+32..r+35.
                    const uint word = words[g * 8u + w];
                    const float4 lo = float4((uint4(word) >> uint4(0u, 8u, 16u, 24u)) & uint4(0xFu));
                    const float4 hi = float4((uint4(word) >> uint4(4u, 12u, 20u, 28u)) & uint4(0xFu));
                    const uint e = g * 64u + w * 4u;
                    for (uint h = 0; h < SP_GROUP_HEADS; ++h) {
                        device const float *q = queries + h * SP_HEAD_DIM + e;
                        d[h] += dot(float4(*(device const packed_float4 *)q), lo)
                              + dot(float4(*(device const packed_float4 *)(q + 32u)), hi);
                    }
                }
                device const half *meta = (device const half *)(record + 128u + g * 4u);
                const float s = float(meta[0]), b = float(meta[1]);
                for (uint h = 0; h < SP_GROUP_HEADS; ++h) score[h] += s * d[h] + b * querySum[h][g];
            }
            for (uint h = 0; h < SP_GROUP_HEADS; ++h) score[h] *= p.scale;
        }
        for (uint h = 0; h < SP_GROUP_HEADS; ++h) shared[j * 8u + h] = score[h];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (j < SP_GROUP_HEADS) {
            float pageMax = -INFINITY;
            for (uint t = 0; t < SP_PAGE_TOKENS; ++t) pageMax = max(pageMax, shared[t * 8u + j]);
            stat[j] = pageMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rescale[SP_GROUP_HEADS];
        for (uint h = 0; h < SP_GROUP_HEADS; ++h) {
            const float newMax = max(runMax[h], stat[h]);
            rescale[h] = exp(runMax[h] - newMax);
            runMax[h] = newMax;
            shared[j * 8u + h] = exp(score[h] - newMax);
        }
        shared[j * 8u + 6u] = 0.0f;
        shared[j * 8u + 7u] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (accumulates) {
            for (uint h = 0; h < SP_GROUP_HEADS; ++h) { acc[h] *= rescale[h]; runSum[h] *= rescale[h]; }
            const uint tokens = min(SP_PAGE_TOKENS, visible - pg * SP_PAGE_TOKENS);
            for (uint t = 0; t < tokens; ++t) {
                device const uchar *record = vpool + ((pageBase + t) * p.kvHeads + kvHead) * SP_KV_RECORD;
                const uint pair = *(device const ushort *)(record + 2u * j);
                device const half *meta = (device const half *)(record + 128u + vGroup * 4u);
                // Channels: byte 2j -> (r, r + 32), byte 2j+1 -> (r + 1, r + 33).
                const float4 codes = float4(float(pair & 0xFu), float((pair >> 8) & 0xFu),
                                            float((pair >> 4) & 0xFu), float((pair >> 12) & 0xFu));
                const float4 value = float(meta[0]) * codes + float(meta[1]);
                threadgroup const packed_float4 *probs = (threadgroup const packed_float4 *)(shared + t * 8u);
                const float4 p0 = float4(probs[0]), p1 = float4(probs[1]);
                acc[0] += p0.x * value; acc[1] += p0.y * value; acc[2] += p0.z * value; acc[3] += p0.w * value;
                acc[4] += p1.x * value; acc[5] += p1.y * value;
                runSum[0] += p0.x; runSum[1] += p0.y; runSum[2] += p0.z; runSum[3] += p0.w;
                runSum[4] += p1.x; runSum[5] += p1.y;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (accumulates) {
        // Channel indices of this thread's four accumulators within the 256-wide vector.
        const uint r = (2u * j) & 31u;
        const uint c0 = vGroup * 64u + r;
        for (uint h = 0; h < SP_GROUP_HEADS; ++h) {
            device float *partial = partials + ((row * p.heads + headBase + h) * p.spans + span) * SP_PARTIAL;
            if (j == 0u) { partial[0] = runMax[h]; partial[1] = runSum[h]; }
            partial[2u + c0] = acc[h].x;
            partial[2u + c0 + 1u] = acc[h].y;
            partial[2u + c0 + 32u] = acc[h].z;
            partial[2u + c0 + 33u] = acc[h].w;
        }
    }
}

// Merge span partials, apply the sigmoid output gate, and emit the accelerator operand.
// Threadgroup grid: (heads, rows), 256 threads per group.
kernel void sp_attn_merge(device const float *partials [[buffer(0)]],
                          device const float *qproj [[buffer(1)]],
                          device float *out [[buffer(2)]],
                          device bfloat *outB [[buffer(3)]],
                          device float *outSums [[buffer(4)]],
                          constant SpAttnScanParams &p [[buffer(5)]],
                          device const float *rowInv [[buffer(6)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint j [[thread_index_in_threadgroup]])
{
    threadgroup float shared[SP_HEAD_DIM];
    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    device const float *base = partials + ((row * p.heads + head) * p.spans) * SP_PARTIAL;
    float top = -INFINITY;
    for (uint s = 0; s < p.spans; ++s) top = max(top, base[s * SP_PARTIAL]);
    float sum = 0.0f, acc = 0.0f;
    for (uint s = 0; s < p.spans; ++s) {
        const float weight = exp(base[s * SP_PARTIAL] - top);
        if (weight > 0.0f) {
            sum += base[s * SP_PARTIAL + 1u] * weight;
            acc += base[s * SP_PARTIAL + 2u + j] * weight;
        }
    }
    const uint index = (row * p.heads + head) * SP_HEAD_DIM + j;
    const float gate = qproj[(row * p.heads + head) * (2u * SP_HEAD_DIM) + SP_HEAD_DIM + j] * rowInv[row];
    const float result = (acc / sum) * sp_sigmoid(gate);
    out[index] = result;
    const bfloat rounded = bfloat(result);
    outB[index] = rounded;
    shared[j] = float(rounded);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((j & 63u) == 0u) {
        float total = 0.0f;
        for (uint i = 0; i < 64u; ++i) total += shared[j + i];
        outSums[row * (p.heads * SP_HEAD_DIM / 64u) + (head * SP_HEAD_DIM + j) / 64u] = total;
    }
}
