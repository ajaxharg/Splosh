#include <metal_stdlib>

using namespace metal;

// gdn_b.metal — candidate kernels for the gated-delta layer's split path (engine.metal:
// sp_gdn_prepare, sp_gdn_chains, sp_gdn_history). Same bindings (SP_GDN_ARGS), same buffers,
// same grids; selected by name through SPLOSH_GDN_PREPARE, SPLOSH_GDN_CHAINS and
// SPLOSH_GDN_HISTORY.
//
//   sp_gdn_prepare_b1                 drop-in for sp_gdn_prepare, bit-identical outputs
//   sp_gdn_chains_b1, _b2             drop-in for sp_gdn_chains, results equal to rounding
//   sp_gdn_history_b1                 drop-in for sp_gdn_history, bit-identical outputs
//   sp_gdn_prepare_bn1 with
//   sp_gdn_chains_bn1 or _bn2         a pair: the prepare stage hands the recurrence different
//                                     operands (see below), so neither works with the other
//                                     half from engine.metal
//   sp_gdn_chains_bs2, _bns2          as _b2 and _bn2, reading q and k (and their norms) at the
//                                     key head's first value head, where every prepare kernel
//                                     of the same form leaves them
//   sp_gdn_prepare_bs1, _bns1         as _b1 and _bn1, storing q and k there only; these need
//                                     the recurrence kernel of the same letters
//
// The letters after `_b` are what Sources/SploshRuntime/GdnKernelChoice.swift goes by to refuse
// a prepare kernel and a recurrence kernel that do not work together: `n` has to match, and an
// `s` prepare kernel needs an `s` recurrence. The kernels numbered 2 use the first two
// simdgroups of a threadgroup, so they can be dispatched with 64 threads
// (SPLOSH_GDN_CHAIN_THREADS=64) as well as the 128 the others need.
//
// Figures below are from the timing test in Tests/SploshOracleTests/GdnCandidateTests.swift: 48
// dispatches of a stage for one 128-row run, which for engine.metal's kernels gives prepare 2.4,
// chains 11.8, finish 1.05, history 1.07 ms.
//
// Where the time of sp_gdn_chains goes, measured by leaving parts of it out: the four arithmetic
// steps per state element (S.k, k * delta, the fused update, S.q) are 6 ms, at the same rate
// whether a lane holds 8, 16 or 32 elements and whether or not delta depends on the state; the
// sixteen-float loads of k and q 1.8 ms; the five scalar loads a row (v, decay, beta, two
// norms) 1.3 ms; the output store 1.0 ms; the shuffles 0.2 ms; launching the threads and the
// state's round trip 0.6 ms. Splash's scan has the same shape (a lane holds sixteen key columns
// of one state row, eight lanes to a row, three shuffle steps per sum, four arithmetic steps
// per element); what it has less of is loads: its prepare stage normalises q and k, so the scan
// reads decay, beta and v per row and nothing else.
//
// The `s` kernels in the same test (8-dispatch samples, scaled to 48): prepare 1.68 (_b1) to
// 1.27 (_bs1, _bns1); chains 10.57 (_b2) to 10.32 (_bs2), and 10.39 (_bn2) to 10.00 (_bns2).
// 64 threads a threadgroup against 128 for the kernels numbered 2: no difference there.
//
// Tried here and left out, because they were no faster than sp_gdn_chains or slower:
//   * k loaded again for the update instead of kept in registers across the shuffles: the same
//     within the noise (10.65 against 10.57 for _b2);
//   * a lane holding four key indices of four or eight columns (one packed load of k and of q a
//     row, but a column's sum becomes a simd_sum over the simdgroup): 13-14 ms;
//   * eight elements a lane, 256 threads a threadgroup: 13.6 ms;
//   * the decay carried as a scalar (registers hold S / product of decays, which removes a
//     multiply per element): 0.2 ms faster with no guard at all, slower with any guard against
//     the product leaving fp32's range (a test per row, or per block of 8, 32 or 128 rows);
//   * k and q as float4x4 loads, outputs gathered in threadgroup memory and stored at the end;
//   * sp_gdn_finish with packed loads and stores (1.04 against 1.05 ms).
// The arrangement of the row matters as much as its content: the same statements with the
// journal's (untaken) branch after the output store instead of before the output's sum, or the
// row loop split into blocks, are 0.6-0.9 ms slower.

struct SpGdnParams {
    uint rows;
    uint runs;
    uint keyHeads;     // 16
    uint valueHeads;   // 48
    uint headDim;      // 128
    uint channels;     // 2 * keyDim + valueDim = 10240
    float eps;
    uint debug;
};

#define SP_GDN_JOURNAL_ROWS 16u
#define SP_GDN_JOURNAL_WIDTH 260u

inline float sp_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline float sp_silu(float x) { return x / (1.0f + exp(-x)); }
inline float sp_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// engine.metal's state update, spelled the same way: a journal entry written here is replayed
// there (sp_gdn_commit) and has to give the same bits.
inline float4 sp_gdn_update(float4 state, float decay, float4 k, float delta) {
    return fma(state, float4(decay), k * delta);
}

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

// ---------------------------------------------------------------------------------------------
// Prepare. sp_gdn_prepare evaluates the conv of q and k once per value head, though they belong
// to the key head (three value heads share one): 48 * 384 conv outputs a row where 16 * 256 +
// 48 * 128 are distinct. Here the first value head of a key head computes q and k and stores
// them for all of the key head's value heads; the other two only do v. The four channels a lane
// takes are consecutive, so a tap is one packed_float4 load.
//
// The conv history of q and k is kept per value head and is the same for the value heads of a
// key head (sp_gdn_history writes it from the same inputs); this reads the first one's.
//
// Normalised: what the chain is handed.
//   false   as sp_gdn_prepare: gq, gk the conv outputs; gnorm their norms; a, b decay and beta
//   true    gq, gk scaled by their norms; gnorm = (decay, beta), the pair the chain reads in
//           one load; a, b decay and beta as well
//
// Shared: where q and k go.
//   false   to all of the key head's value heads, as sp_gdn_prepare leaves them
//   true    to its first value head only (and their norms likewise), for a recurrence kernel
//           that reads them there: a third of the q and k stores
//
// Threadgroup grid: (valueHeads, rows), 96 threads, as sp_gdn_prepare.
template <bool Normalised, bool Shared>
inline void sp_gdn_prepare_b(device const float *mixed, device float *a, device float *b,
                             device const ushort *convWeight, device const ushort *aLog,
                             device const ushort *dtBias, device const float *convState,
                             device const uint *runs, constant SpGdnParams &p,
                             device const uint *rowRead, device float *gq, device float *gk,
                             device float *gv, device float *gnorm, device const float *rowInv,
                             uint3 group, uint lane, uint sg)
{
    const uint h = group.x, row = group.y;
    if (h >= p.valueHeads || row >= p.rows || sg >= 3u) return;
    const uint d = p.headDim;
    const uint keyDim = p.keyHeads * d;
    const uint perKey = p.valueHeads / p.keyHeads;
    const uint hk = h / perKey;
    const uint kind = sg;                                           // q, k, v
    if (kind < 2u && h != hk * perKey) return;
    const uint convPerUnit = p.valueHeads * 3u * d * 3u;
    // The run this row is in: its history before the run is the stored one.
    uint start = 0u;
    for (uint r = 0; r < p.runs; ++r) {
        const uint first = runs[r * 3u + 1u];
        if (row >= first && row < first + runs[r * 3u + 2u]) start = first;
    }
    const uint t = row - start;
    const uint jc = lane * 4u;
    const uint channel = kind == 0u ? hk * d + jc : kind == 1u ? keyDim + hk * d + jc : 2u * keyDim + h * d + jc;
    // history[e] = the three inputs before the run of channel + e, oldest first.
    float3 history[4] = { float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f) };
    if (t < 3u) {
        device const packed_float4 *stored = (device const packed_float4 *)
            (convState + rowRead[start] * convPerUnit + ((h * 3u + kind) * d + jc) * 3u);
        const float4 h0 = float4(stored[0]), h1 = float4(stored[1]), h2 = float4(stored[2]);
        history[0] = h0.xyz;
        history[1] = float3(h0.w, h1.xy);
        history[2] = float3(h1.zw, h2.x);
        history[3] = h2.yzw;
    }
    device const float *in = mixed + row * p.channels + channel;
    const float inv0 = rowInv[row];
    const float4 x0 = float4(*(device const packed_float4 *)in) * inv0;
    float4 x1, x2, x3;
    if (t >= 1u) x1 = float4(*(device const packed_float4 *)(in - p.channels)) * rowInv[row - 1u];
    else x1 = float4(history[0].z, history[1].z, history[2].z, history[3].z);
    if (t >= 2u) x2 = float4(*(device const packed_float4 *)(in - 2u * p.channels)) * rowInv[row - 2u];
    else if (t == 1u) x2 = float4(history[0].z, history[1].z, history[2].z, history[3].z);
    else x2 = float4(history[0].y, history[1].y, history[2].y, history[3].y);
    if (t >= 3u) x3 = float4(*(device const packed_float4 *)(in - 3u * p.channels)) * rowInv[row - 3u];
    else if (t == 2u) x3 = float4(history[0].z, history[1].z, history[2].z, history[3].z);
    else if (t == 1u) x3 = float4(history[0].y, history[1].y, history[2].y, history[3].y);
    else x3 = float4(history[0].x, history[1].x, history[2].x, history[3].x);
    device const packed_ushort4 *taps = (device const packed_ushort4 *)(convWeight + channel * 4u);
    float4 out;
    for (uint e = 0; e < 4u; ++e) {
        const ushort4 bits = ushort4(taps[e]);
        const float4 w = float4(sp_bf16(bits.x), sp_bf16(bits.y), sp_bf16(bits.z), sp_bf16(bits.w));
        out[e] = sp_silu(dot(w, float4(x3[e], x2[e], x1[e], x0[e])));
    }
    const uint index = row * p.valueHeads + h;
    if (kind < 2u) {
        // L2 norms of q and k (q additionally scaled by 1/sqrt(headDim)).
        const float squares = simd_sum(dot(out, out));
        const float inv = kind == 0u ? rsqrt(squares + 1e-6f) * rsqrt(float(d)) : rsqrt(squares + 1e-6f);
        device float *target = kind == 0u ? gq : gk;
        const float4 stored = Normalised ? out * inv : out;
        const uint copies = Shared ? 1u : perKey;
        for (uint copy = 0; copy < copies; ++copy) {
            *(device packed_float4 *)(target + (index + copy) * d + jc) = stored;
            if (!Normalised && lane == 0u) gnorm[(index + copy) * 2u + kind] = inv;
        }
    } else {
        *(device packed_float4 *)(gv + index * d + jc) = out;
        if (lane == 0u) {
            const float negA = -exp(sp_bf16(aLog[h]));
            const float x = a[index] * inv0 + sp_bf16(dtBias[h]);
            const float softplus = x > 20.0f ? x : log(1.0f + exp(x));
            const float decay = exp(negA * softplus), beta = sp_sigmoid(b[index] * inv0);
            a[index] = decay;
            b[index] = beta;
            if (Normalised) *(device packed_float2 *)(gnorm + index * 2u) = float2(decay, beta);
        }
    }
}

#define SP_GDN_PREPARE_B(Name, Normalised, Shared)                                                \
    kernel void Name(SP_GDN_ARGS,                                                                 \
                     uint3 group [[threadgroup_position_in_grid]],                                \
                     uint lane [[thread_index_in_simdgroup]],                                     \
                     uint sg [[simdgroup_index_in_threadgroup]])                                  \
    {                                                                                             \
        sp_gdn_prepare_b<Normalised, Shared>(mixed, a, b, convWeight, aLog, dtBias, convState,    \
                                             runs, p, rowRead, gq, gk, gv, gnorm, rowInv, group,  \
                                             lane, sg);                                           \
    }

SP_GDN_PREPARE_B(sp_gdn_prepare_b1, false, false)
SP_GDN_PREPARE_B(sp_gdn_prepare_bn1, true, false)
SP_GDN_PREPARE_B(sp_gdn_prepare_bs1, false, true)
SP_GDN_PREPARE_B(sp_gdn_prepare_bns1, true, true)
#undef SP_GDN_PREPARE_B

// ---------------------------------------------------------------------------------------------
// The recurrence. Threadgroup grid: (valueHeads * 8, runs), as sp_gdn_chains, with its 128
// threads (Parts 8) or with 64 or 128 (Parts 4, where only two simdgroups have work), and in its
// orientation: a lane holds consecutive key indices of one state column, and a column's
// sum over the key index is xor-shuffles between the lanes that share it.
//
// Parts is the lanes to a column: 8 is sp_gdn_chains' layout (16 state elements a lane, three
// shuffle steps a sum, 32 simdgroups a head); 4 holds 32 elements a lane (two shuffle steps, 16
// simdgroups a head, so of a threadgroup of 128 threads two simdgroups return at once; with 64
// threads they are not started). With 32 elements the per-row work that does not grow with the
// state a lane holds (the scalar loads, delta, the shuffles, the output store) is spread over
// twice the elements.
//
// The sums run in one float4 accumulator per lane, a multiply-add per element, and the row's
// addresses are carried as pointers.
//
// Normalised: the operands sp_gdn_prepare_bn1 leaves. A row then reads one float2 of gates and
// v where sp_gdn_chains reads five scalars, and has no norm to apply:
//   false   delta = (v - S.k * decay * invK) * beta * invK;  out = S'.q * invQ
//   true    delta = (v - S.k * decay) * beta;                out = S'.q
// The journal entry of a speculative row holds the k and delta its update was made with, so
// sp_gdn_commit's replay is the same expression either way.
//
// Shared: q and k (and, when they are not normalised, their norms) are read at the key head's
// first value head rather than at this value head. The three value heads of a key head have the
// same q and k, so every prepare kernel leaves them there; sp_gdn_prepare_bs1 and _bns1 leave
// them nowhere else. The three heads' chains then read the same rows of q and k.
template <uint Parts, bool Normalised, bool Shared>
inline void sp_gdn_chain_b(device const float *a, device const float *b, device float *state,
                           device float *core, device const uint *runs, constant SpGdnParams &p,
                           device const float *gq, device const float *gk, device const float *gv,
                           device const float *gnorm, device float *journal, device const uint *runInfo,
                           uint3 group, uint lane, uint sgLocal)
{
    constexpr uint PerGroup = Parts / 2u;          // simdgroups of a head in each of its 8 threadgroups
    constexpr uint Columns = 32u / Parts;          // columns a simdgroup owns
    constexpr uint Quads = 32u / Parts;            // float4s of state a lane holds (headDim 128)
    const uint h = group.x / 8u;
    if (h >= p.valueHeads || group.y >= p.runs || sgLocal >= PerGroup) return;
    const uint sg = (group.x % 8u) * PerGroup + sgLocal;
    const uint part = lane % Parts, j = sg * Columns + lane / Parts;
    const uint slot = runs[group.y * 3u], start = runs[group.y * 3u + 1u], length = runs[group.y * 3u + 2u];
    const uint d = p.headDim;
    const uint key = part * (Quads * 4u);
    const bool speculative = runInfo[group.y * 2u] != 0u;
    // Journal entries accepted from the slot's last speculative run are already in the stored
    // state (sp_gdn_commit runs in the stage before).
    device packed_float4 *stored = (device packed_float4 *)(state + slot * p.valueHeads * d * d + (h * d + j) * d + key);
    float4 s[Quads];
    for (uint c = 0; c < Quads; ++c) s[c] = float4(stored[c]);
    device float *entry = journal + (slot * SP_GDN_JOURNAL_ROWS * p.valueHeads + h) * SP_GDN_JOURNAL_WIDTH;
    const uint rowStride = p.valueHeads * d;
    uint index = start * p.valueHeads + h;
    // From this value head back to the one whose q and k are read.
    const uint back = Shared ? h % (p.valueHeads / p.keyHeads) : 0u;
    device const packed_float4 *kp = (device const packed_float4 *)(gk + (index - back) * d + key);
    device const packed_float4 *qp = (device const packed_float4 *)(gq + (index - back) * d + key);
    device const float *vp = gv + index * d + j;
    device float *op = core + index * d + j;
    for (uint t = 0; t < length; ++t) {
        float decay, beta, invQ = 1.0f, invK = 1.0f;
        if (Normalised) {
            const float2 gates = float2(*(device const packed_float2 *)(gnorm + index * 2u));
            decay = gates.x; beta = gates.y;
        } else {
            invQ = gnorm[(index - back) * 2u]; invK = gnorm[(index - back) * 2u + 1u]; decay = a[index]; beta = b[index];
        }
        float4 k[Quads];
        float4 sum = float4(0.0f);
        for (uint c = 0; c < Quads; ++c) { k[c] = float4(kp[c]); sum = fma(s[c], k[c], sum); }
        float mem = (sum.x + sum.y) + (sum.z + sum.w);
        for (uint step = 1u; step < Parts; step <<= 1u) mem += simd_shuffle_xor(mem, ushort(step));
        const float v = *vp;
        const float delta = Normalised ? (v - mem * decay) * beta : (v - mem * decay * invK) * beta * invK;
        sum = float4(0.0f);
        for (uint c = 0; c < Quads; ++c) {
            s[c] = sp_gdn_update(s[c], decay, k[c], delta);
            sum = fma(s[c], float4(qp[c]), sum);
        }
        // (Here, before the output's sum and store: see the note at the top of the file.)
        if (speculative) {
            device float *line = entry + t * p.valueHeads * SP_GDN_JOURNAL_WIDTH;
            // Column 0's parts between them hold the whole of k.
            if (j == 0u) {
                for (uint c = 0; c < Quads; ++c) *(device packed_float4 *)(line + key + c * 4u) = k[c];
                if (part == 0u) line[2u * d] = decay;
            }
            if (part == 0u) line[d + j] = delta;
        }
        float o = (sum.x + sum.y) + (sum.z + sum.w);
        for (uint step = 1u; step < Parts; step <<= 1u) o += simd_shuffle_xor(o, ushort(step));
        if (part == 0u) *op = Normalised ? o : o * invQ;
        index += p.valueHeads;
        kp += rowStride / 4u; qp += rowStride / 4u; vp += rowStride; op += rowStride;
    }
    if (!speculative && length != 0u) {
        for (uint c = 0; c < Quads; ++c) stored[c] = s[c];
    }
}

#define SP_GDN_CHAIN_B(Name, Parts, Normalised, Shared)                                           \
    kernel void Name(SP_GDN_ARGS,                                                                 \
                     uint3 group [[threadgroup_position_in_grid]],                                \
                     uint lane [[thread_index_in_simdgroup]],                                     \
                     uint sgLocal [[simdgroup_index_in_threadgroup]])                             \
    {                                                                                             \
        sp_gdn_chain_b<Parts, Normalised, Shared>(a, b, state, core, runs, p, gq, gk, gv, gnorm,  \
                                                  journal, runInfo, group, lane, sgLocal);        \
    }

// After sp_gdn_prepare or sp_gdn_prepare_b1.
SP_GDN_CHAIN_B(sp_gdn_chains_b1, 8, false, false)   // 16 elements a lane
SP_GDN_CHAIN_B(sp_gdn_chains_b2, 4, false, false)   // 32 elements a lane
// After sp_gdn_prepare_bn1.
SP_GDN_CHAIN_B(sp_gdn_chains_bn1, 8, true, false)   // 16 elements a lane
SP_GDN_CHAIN_B(sp_gdn_chains_bn2, 4, true, false)   // 32 elements a lane
// After sp_gdn_prepare, sp_gdn_prepare_b1 or sp_gdn_prepare_bs1.
SP_GDN_CHAIN_B(sp_gdn_chains_bs2, 4, false, true)   // 32 elements a lane
// After sp_gdn_prepare_bn1 or sp_gdn_prepare_bns1.
SP_GDN_CHAIN_B(sp_gdn_chains_bns2, 4, true, true)   // 32 elements a lane
#undef SP_GDN_CHAIN_B

// ---------------------------------------------------------------------------------------------
// History: sp_gdn_history walks every row of a run in every thread to find the rows that write
// a unit (two loads a row). An ordinary run writes one unit, at its last row (all of its rows
// name the slot's unit); only a speculative run, of at most sixteen rows, writes one per row.
// Threadgroup grid: (valueHeads, runs), 384 threads, as sp_gdn_history.
kernel void sp_gdn_history_b1(SP_GDN_ARGS,
                              uint3 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]])
{
    const uint h = group.x;
    if (h >= p.valueHeads || group.y >= p.runs || tid >= 384u) return;
    const uint start = runs[group.y * 3u + 1u], length = runs[group.y * 3u + 2u];
    if (length == 0u) return;
    const bool speculative = runInfo[group.y * 2u] != 0u;
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
    for (uint t = speculative ? 0u : length - 1u; t < length; ++t) {
        const uint row = start + t;
        const uint mb = row * p.channels + channel;
        const float x1 = mixed[mb] * rowInv[row];
        const float x2 = t >= 1u ? mixed[mb - p.channels] * rowInv[row - 1u] : history.z;
        const float x3 = t >= 2u ? mixed[mb - 2u * p.channels] * rowInv[row - 2u] : (t == 1u ? history.z : history.y);
        const uint conv = rowWrite[row] * convPerUnit + convInUnit;
        convState[conv] = x3; convState[conv + 1u] = x2; convState[conv + 2u] = x1;
    }
}
