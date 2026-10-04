#include <metal_stdlib>

using namespace metal;

// draft.metal — kernels specific to the DFlash 2 block-diffusion draft model.
// Formulas follow dflash/model.py in z-lab/dflash (Apache-2.0); see audit/DFLASH2-REFERENCE.md.
// Everything linear in the draft reuses the q4 GEMM kernels in engine.metal.

#define SPD_LANES 32u
#define SPD_HEAD_DIM 128u

inline float spd_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }

inline float spd_rope(float x, float paired, uint i, uint position, float theta) {
    const uint half_ = SPD_HEAD_DIM >> 1;
    const uint slot = i < half_ ? i : i - half_;
    const float angle = float(position) * pow(theta, -(2.0f * float(slot)) / float(SPD_HEAD_DIM));
    return x * cos(angle) + (i < half_ ? -paired : paired) * sin(angle);
}

// Grouped dynamic causal convolution (kernel size 2) over the rows of one block:
//   out[t][c] = sum_{o in 0,1} (base[which][o][c] + dynamic[t][which][o][c / groupSize]) * in[t - o][c]
// with in[t - o] taken as zero before the first row of the block (`rowOffset` is a row's index
// within its block; blocks differ in length). `which` is 0 for the
// "prepare" pass and 1 for "finish"; `accumulate` adds the result into `out` (the residual).
// Thread grid: (hidden, rows).
struct SpdConvParams {
    uint rows;
    uint hidden;
    uint groupSize;
    uint groups;
    uint which;
    uint accumulate;
};

kernel void sp_dyn_conv(device const float *input [[buffer(0)]],
                        device const float *dynamic_ [[buffer(1)]],
                        device const ushort *base [[buffer(2)]],
                        device float *out [[buffer(3)]],
                        constant SpdConvParams &p [[buffer(4)]],
                        device const uint *rowOffset [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.hidden || gid.y >= p.rows) return;
    const uint c = gid.x, t = gid.y;
    const uint group = c / p.groupSize;
    const uint inBlock = rowOffset[t];
    float acc = 0.0f;
    for (uint o = 0; o < 2u; ++o) {
        if (inBlock < o) continue;
        const uint tap = p.which * 2u + o;
        const float kernel_ = spd_bf16(base[tap * p.hidden + c]) + dynamic_[t * (4u * p.groups) + tap * p.groups + group];
        acc += kernel_ * input[(t - o) * p.hidden + c];
    }
    const uint index = t * p.hidden + c;
    out[index] = p.accumulate != 0u ? out[index] + acc : acc;
}

// Draft K/V into the per-slot context ring: K is RMSNorm (plain weight) then full-width RoPE,
// V is stored verbatim. Ring layout: [slot][position % capacity][kvHead][128], fp16.
// Threadgroup grid: (kvHeads, rows), 32 threads per group.
struct SpdKVParams {
    uint rows;
    uint kvHeads;
    uint capacity;
    float eps;
    float theta;
};

kernel void sp_draft_kv_store(device const float *kproj [[buffer(0)]],
                              device const float *vproj [[buffer(1)]],
                              device const ushort *normWeight [[buffer(2)]],
                              device const uint *rowSlot [[buffer(3)]],
                              device const uint *rowPos [[buffer(4)]],
                              device half *kring [[buffer(5)]],
                              device half *vring [[buffer(6)]],
                              constant SpdKVParams &p [[buffer(7)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]])
{
    const uint head = group.x, row = group.y;
    if (head >= p.kvHeads || row >= p.rows) return;
    const uint base = (row * p.kvHeads + head) * SPD_HEAD_DIM;
    float partial = 0.0f;
    for (uint i = lane; i < SPD_HEAD_DIM; i += SPD_LANES) {
        const float x = kproj[base + i];
        partial += x * x;
    }
    const float inv = rsqrt(simd_sum(partial) / float(SPD_HEAD_DIM) + p.eps);
    const uint position = rowPos[row];
    const uint dst = ((rowSlot[row] * p.capacity + position % p.capacity) * p.kvHeads + head) * SPD_HEAD_DIM;
    const uint half_ = SPD_HEAD_DIM >> 1;
    for (uint i = lane; i < SPD_HEAD_DIM; i += SPD_LANES) {
        const uint pi = i < half_ ? i + half_ : i - half_;
        const float x = kproj[base + i] * inv * spd_bf16(normWeight[i]);
        const float paired = kproj[base + pi] * inv * spd_bf16(normWeight[pi]);
        kring[dst + i] = half(spd_rope(x, paired, i, position, p.theta));
        vring[dst + i] = half(vproj[base + i]);
    }
}

// Draft attention: non-causal within a sliding window. Query row r of a block sees every ring
// position in [rowLo[r], rowHi[r]) that is closer than `window` — the accumulated context plus
// all rows of its own block. One 128-thread group per (head, row); thread j owns channel j.
// Threadgroup grid: (heads, rows), 128 threads per group.
struct SpdAttnParams {
    uint rows;
    uint heads;
    uint kvHeads;
    uint capacity;
    uint window;
    float scale;
    float eps;
    float theta;
    uint spans;
};

kernel void sp_draft_attention(device const float *qproj [[buffer(0)]],
                               device const ushort *normWeight [[buffer(1)]],
                               device const uint *rowSlot [[buffer(2)]],
                               device const uint *rowPos [[buffer(3)]],
                               device const uint *rowLo [[buffer(4)]],
                               device const uint *rowHi [[buffer(5)]],
                               device const half *kring [[buffer(6)]],
                               device const half *vring [[buffer(7)]],
                               device float *out [[buffer(8)]],
                               constant SpdAttnParams &p [[buffer(9)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint j [[thread_index_in_threadgroup]])
{
    threadgroup float shared[SPD_HEAD_DIM];
    threadgroup float query[SPD_HEAD_DIM];

    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    const uint inBase = (row * p.heads + head) * SPD_HEAD_DIM;
    const uint position = rowPos[row];

    shared[j] = qproj[inBase + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    for (uint i = 0; i < SPD_HEAD_DIM; ++i) ss += shared[i] * shared[i];
    const float inv = rsqrt(ss / float(SPD_HEAD_DIM) + p.eps);
    const uint half_ = SPD_HEAD_DIM >> 1;
    const uint pj = j < half_ ? j + half_ : j - half_;
    query[j] = spd_rope(shared[j] * inv * spd_bf16(normWeight[j]),
                        shared[pj] * inv * spd_bf16(normWeight[pj]), j, position, p.theta);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kvHead = head / (p.heads / p.kvHeads);
    const uint firstVisible = position + 1u > p.window ? position + 1u - p.window : 0u;
    const uint lo = max(rowLo[row], firstVisible);
    const uint hi = rowHi[row];
    const uint chunks = (hi - lo + SPD_HEAD_DIM - 1u) / SPD_HEAD_DIM;
    const uint ringBase = rowSlot[row] * p.capacity;

    float runMax = -INFINITY, runSum = 0.0f, acc = 0.0f;
    for (uint chunk = 0; chunk < chunks; ++chunk) {
        const uint first = lo + chunk * SPD_HEAD_DIM;
        float score = -INFINITY;
        if (first + j < hi) {
            const uint kBase = ((ringBase + (first + j) % p.capacity) * p.kvHeads + kvHead) * SPD_HEAD_DIM;
            float d = 0.0f;
            for (uint i = 0; i < SPD_HEAD_DIM; ++i) d += query[i] * float(kring[kBase + i]);
            score = d * p.scale;
        }
        shared[j] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float chunkMax = -INFINITY;
        for (uint t = 0; t < SPD_HEAD_DIM; ++t) chunkMax = max(chunkMax, shared[t]);
        const float newMax = max(runMax, chunkMax);
        const float rescale = exp(runMax - newMax);
        acc *= rescale;
        runSum *= rescale;
        runMax = newMax;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared[j] = exp(score - runMax);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0; t < SPD_HEAD_DIM; ++t) {
            const float prob = shared[t];
            if (prob > 0.0f) {
                runSum += prob;
                acc += prob * float(vring[((ringBase + (first + t) % p.capacity) * p.kvHeads + kvHead) * SPD_HEAD_DIM + j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[inBase + j] = acc / runSum;
}

// The same attention with the window divided into spans that run in parallel: the kernel above
// walks a full 2304-token window as eighteen chunks in sequence, four barriers each, which is
// latency the draft pass pays on every cycle once the context is long.
// Threadgroup grid: (heads, rows, spans), 128 threads per group. Partial per (row, head, span):
// [running max, running sum, 128 accumulated values].
kernel void sp_draft_attention_span(device const float *qproj [[buffer(0)]],
                                    device const ushort *normWeight [[buffer(1)]],
                                    device const uint *rowSlot [[buffer(2)]],
                                    device const uint *rowPos [[buffer(3)]],
                                    device const uint *rowLo [[buffer(4)]],
                                    device const uint *rowHi [[buffer(5)]],
                                    device const half *kring [[buffer(6)]],
                                    device const half *vring [[buffer(7)]],
                                    device float *partials [[buffer(8)]],
                                    constant SpdAttnParams &p [[buffer(9)]],
                                    uint3 group [[threadgroup_position_in_grid]],
                                    uint j [[thread_index_in_threadgroup]])
{
    threadgroup float shared[SPD_HEAD_DIM];
    threadgroup float query[SPD_HEAD_DIM];

    const uint head = group.x, row = group.y, span = group.z;
    if (head >= p.heads || row >= p.rows || span >= p.spans) return;
    const uint inBase = (row * p.heads + head) * SPD_HEAD_DIM;
    const uint position = rowPos[row];

    shared[j] = qproj[inBase + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    for (uint i = 0; i < SPD_HEAD_DIM; ++i) ss += shared[i] * shared[i];
    const float inv = rsqrt(ss / float(SPD_HEAD_DIM) + p.eps);
    const uint half_ = SPD_HEAD_DIM >> 1;
    const uint pj = j < half_ ? j + half_ : j - half_;
    query[j] = spd_rope(shared[j] * inv * spd_bf16(normWeight[j]),
                        shared[pj] * inv * spd_bf16(normWeight[pj]), j, position, p.theta);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kvHead = head / (p.heads / p.kvHeads);
    const uint firstVisible = position + 1u > p.window ? position + 1u - p.window : 0u;
    const uint lo = max(rowLo[row], firstVisible);
    const uint hi = rowHi[row];
    const uint chunks = (hi - lo + SPD_HEAD_DIM - 1u) / SPD_HEAD_DIM;
    const uint perSpan = (chunks + p.spans - 1u) / p.spans;
    const uint firstChunk = span * perSpan, lastChunk = min(chunks, firstChunk + perSpan);
    const uint ringBase = rowSlot[row] * p.capacity;

    float runMax = -INFINITY, runSum = 0.0f, acc = 0.0f;
    for (uint chunk = firstChunk; chunk < lastChunk; ++chunk) {
        const uint first = lo + chunk * SPD_HEAD_DIM;
        float score = -INFINITY;
        if (first + j < hi) {
            const uint kBase = ((ringBase + (first + j) % p.capacity) * p.kvHeads + kvHead) * SPD_HEAD_DIM;
            float d = 0.0f;
            for (uint i = 0; i < SPD_HEAD_DIM; ++i) d += query[i] * float(kring[kBase + i]);
            score = d * p.scale;
        }
        shared[j] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float chunkMax = -INFINITY;
        for (uint t = 0; t < SPD_HEAD_DIM; ++t) chunkMax = max(chunkMax, shared[t]);
        const float newMax = max(runMax, chunkMax);
        const float rescale = exp(runMax - newMax);
        acc *= rescale;
        runSum *= rescale;
        runMax = newMax;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared[j] = exp(score - runMax);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = 0; t < SPD_HEAD_DIM; ++t) {
            const float prob = shared[t];
            if (prob > 0.0f) {
                runSum += prob;
                acc += prob * float(vring[((ringBase + (first + t) % p.capacity) * p.kvHeads + kvHead) * SPD_HEAD_DIM + j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *partial = partials + ((row * p.heads + head) * p.spans + span) * (SPD_HEAD_DIM + 2u);
    if (j == 0u) { partial[0] = runMax; partial[1] = runSum; }
    partial[2u + j] = acc;
}

// Threadgroup grid: (heads, rows), 128 threads per group.
kernel void sp_draft_attention_merge(device const float *partials [[buffer(0)]],
                                     device float *out [[buffer(1)]],
                                     constant SpdAttnParams &p [[buffer(2)]],
                                     uint3 group [[threadgroup_position_in_grid]],
                                     uint j [[thread_index_in_threadgroup]])
{
    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    device const float *base = partials + ((row * p.heads + head) * p.spans) * (SPD_HEAD_DIM + 2u);
    float top = -INFINITY;
    for (uint s = 0; s < p.spans; ++s) top = max(top, base[s * (SPD_HEAD_DIM + 2u)]);
    float sum = 0.0f, acc = 0.0f;
    for (uint s = 0; s < p.spans; ++s) {
        const float weight = exp(base[s * (SPD_HEAD_DIM + 2u)] - top);
        if (weight > 0.0f) {
            sum += base[s * (SPD_HEAD_DIM + 2u) + 1u] * weight;
            acc += base[s * (SPD_HEAD_DIM + 2u) + 2u + j] * weight;
        }
    }
    out[(row * p.heads + head) * SPD_HEAD_DIM + j] = acc / sum;
}

// Queries for the accelerator attention below: RMSNorm and RoPE, rounded to fp16, laid out
// [kvHead][row][query head in group][128] so one request's 8 rows x 4 heads are 32 contiguous
// queries. Threadgroup grid: (heads, rows), 128 threads per group.
struct SpdQPrepParams { uint rows; uint heads; uint kvHeads; uint rowCap; float eps; float theta; };

kernel void sp_draft_q_prepare(device const float *qproj [[buffer(0)]],
                               device const ushort *normWeight [[buffer(1)]],
                               device const uint *rowPos [[buffer(2)]],
                               device half *qB [[buffer(3)]],
                               constant SpdQPrepParams &p [[buffer(4)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint j [[thread_index_in_threadgroup]])
{
    threadgroup float shared[SPD_HEAD_DIM];
    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    const uint inBase = (row * p.heads + head) * SPD_HEAD_DIM;
    shared[j] = qproj[inBase + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    for (uint i = 0; i < SPD_HEAD_DIM; ++i) ss += shared[i] * shared[i];
    const float inv = rsqrt(ss / float(SPD_HEAD_DIM) + p.eps);
    const uint half_ = SPD_HEAD_DIM >> 1;
    const uint pj = j < half_ ? j + half_ : j - half_;
    const float value = spd_rope(shared[j] * inv * spd_bf16(normWeight[j]),
                                 shared[pj] * inv * spd_bf16(normWeight[pj]), j, rowPos[row], p.theta);
    const uint perKV = p.heads / p.kvHeads;
    qB[(((head / perKV) * p.rowCap + row) * perKV + head % perKV) * SPD_HEAD_DIM + j] = half(value);
}
