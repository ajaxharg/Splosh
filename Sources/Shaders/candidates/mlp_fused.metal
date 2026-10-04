#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// mlp_fused.metal — the MLP's up-projection with the SiLU product in its epilogue.
//
// The shipped wide path runs three dispatches: the gate GEMM and the up GEMM, each storing
// rows x 17408 fp32 values, then sp_silu_mul_na, which reads both arrays and writes the down
// projection's operand (bf16 values and their per-64 sums). Here the up GEMM never stores its
// result: once the gate GEMM has finished, each thread of the up GEMM reads the gate values for
// the elements it holds, forms silu(gate) * up in registers, and the accelerator stores the
// tile as bf16. Per element this drops one fp32 store and one fp32 load (the up array) and the
// bf16 store moves from threads to the accelerator.
//
// Variants of the whole-tile kernel, to be timed against each other:
// - the per-64 sums either read back from the tile just stored (it is in cache; one bf16 load
//   per element) or taken from the values still in registers, as the operand-emitting GEMM does
//   (two shuffles per four values and a four-way add through threadgroup memory);
// - the gate either fp32, as the shipped gate GEMM stores it, or bf16, stored through the
//   accelerator by sp_mlp_gate_bf16_m32n128s4 below (half the traffic on the one array left).
//
// With an fp32 gate the arithmetic is that of sp_na_wide followed by sp_silu_mul_na. The fp32
// product is not always the same to its last bit (the compiler is free to order the two scale
// multiplications differently in a different kernel), so on a step of whole tiles up to about
// one value in 100,000 rounds to the neighbouring bf16 and the rest are identical. On a step
// with a partial tile it is a few in 10,000: there the shipped path runs sp_na_tiled, not
// sp_na_wide, over every tile. Sums taken in registers add in a different order
// from simd_sum and differ by fp32 rounding. A bf16 gate is rounded before SiLU, which the
// shipped path does not do: the product then differs by what a bf16 step of the gate moves it.
// (Tests/SploshOracleTests/MlpFusedTests.swift measures all of these.)
//
// Weights are in the tiled layout (see engine_na.metal): packed [tile of 128 rows][quant
// group][row in tile][32 bytes of codes], sidecars [tile][group][row in tile].

// As SpNaParams, with the residual flag's slot naming the first row tile the dispatch covers.
struct SpMlpFusedParams {
    uint rows;
    uint outDim;
    uint inner;
    uint groups;
    uint firstTile;
    uint outStride;
};

#define SP_MLP_ROW_BLOCK 4u

inline float sp_mlp_silu(float x) { return x / (1.0f + exp(-x)); }
inline float4 sp_mlp_silu(float4 x) { return x / (1.0f + exp(-x)); }

// Whole: every row of the tile is in the step, and the accelerator stores the tile. Otherwise
// (the last, partial tile of a step) the live rows are written from the threads, so nothing
// lands on rows the step does not own. The two are separate kernels: what a kernel carries
// decides how many of its threadgroups run at once (see sp_na_wide). RegisterSums and a bfloat
// Gate apply to the whole-tile kernel only.
template <ushort M, ushort N, ushort Simdgroups, bool Whole, bool RegisterSums, typename Gate>
inline void sp_mlp_up_silu(device uchar *packed, device bfloat *scales, device bfloat *biases,
                           device bfloat *a, device const Gate *gate, device bfloat *outB,
                           device const float *sums, constant SpMlpFusedParams &p,
                           device float *outSums, device const float *rowInv,
                           uint3 group, uint lane, uint simdgroup, threadgroup float *sumStage)
{
    const uint tile = group.y;
    const uint n0 = tile * N;
    const uint r0 = (p.firstTile + group.z * SP_MLP_ROW_BLOCK + group.x) * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, false);
    matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
    auto a0 = activations.template slice<64, M>(0, 0);
    device uchar *tileWeights = packed + ulong(tile) * p.groups * (N * 32u);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first(
        tileWeights, dextents<int, 2>{64, N}, array<int, 2>{1, 64});
    auto b0 = first.template slice<64, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
    for (uint g = 0; g < p.groups; ++g) {
        auto as = activations.template slice<64, M>(int(g * 64u), 0);
        tensor<device uint4b_format, dextents<int, 2>, tensor_inline> weights(
            tileWeights + ulong(g) * (N * 32u), dextents<int, 2>{64, N}, array<int, 2>{1, 64});
        auto bs = weights.template slice<64, N>(0, 0);
        auto partial = operation.template get_destination_cooperative_tensor<decltype(as), decltype(bs), float>();
        operation.run(as, bs, partial);
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            const auto index = accumulated.get_multidimensional_index(i);
            const ulong parameter = (ulong(tile) * p.groups + g) * N + index[0];
            accumulated[i] += partial[i] * float(scales[parameter])
                + sums[(r0 + uint(index[1])) * p.groups + g] * float(biases[parameter]);
        }
    }

    // silu(gate) * up. Both projections carry the deferred scale of the norm that fed them
    // (1 when it was applied), as in sp_silu_mul_na.
    if constexpr (Whole) {
        // A destination of the same operation and operands has the same element partition,
        // so element i of both is the same (column, row).
        auto product = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), bfloat>();
        // For this shape the accelerator gives every thread a regular block (sp_na_tiled checks
        // it per thread; here the test does, by comparing every value): columns c0 ..< c0 + 4
        // and c0 + 64 ..< c0 + 68, rows rb, rb + 8, rb + 16, rb + 24, in element order
        // [row pair][column run][row][column]. So a thread loads eight runs of four gate values
        // and four row scales, not one of each per element.
        const auto origin = accumulated.get_multidimensional_index(ushort(0));
        const uint c0 = uint(origin[0]), rb = uint(origin[1]);
        device const float *rowScale = rowInv + r0 + rb;
        const float4 rowScales = float4(rowScale[0], rowScale[8], rowScale[16], rowScale[24]);
#pragma unroll
        for (ushort block = 0; block < 8; ++block) {
            const uint quarter = (block & 1) + 2 * (block >> 2), side = (block >> 1) & 1;
            const uint rowInTile = rb + quarter * 8u;
            const uint position = (r0 + rowInTile) * p.outStride + n0 + c0 + side * 64u;
            const float scale = rowScales[quarter];
            const float4 gateValues = float4(*(device const vec<Gate, 4> *)(gate + position));
            const float4 up = float4(accumulated[block * 4], accumulated[block * 4 + 1],
                                     accumulated[block * 4 + 2], accumulated[block * 4 + 3]);
            const bfloat4 rounded = bfloat4(sp_mlp_silu(gateValues * scale) * (up * scale));
#pragma unroll
            for (ushort e = 0; e < 4; ++e) product[block * 4 + e] = rounded[e];
            if constexpr (RegisterSums) {
                // The sixteen runs of one (row, 64-column group) sit in the four lanes of each
                // simdgroup that share this thread's rows (lane bits 0 and 3 select the run):
                // two xor-shuffles here, then a sum across simdgroups through threadgroup
                // memory (as the operand-emitting GEMM, sp_na_tiled's Mode 7).
                const float4 back = float4(rounded);
                float sum = back.x + back.y + back.z + back.w;
                sum += simd_shuffle_xor(sum, ushort(1)); sum += simd_shuffle_xor(sum, ushort(8));
                if ((lane & 9u) == 0u) sumStage[(rowInTile * 2u + side) * Simdgroups + simdgroup] = sum;
            }
        }
        auto destination = tensor(outB + ulong(r0) * p.outStride, dextents<int, 2>{int(p.outStride), M},
                                  array<int, 2>{1, int(p.outStride)});
        product.store(destination.template slice<N, M>(int(n0), 0));
        if constexpr (RegisterSums) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdgroup == 0u && (lane & 9u) == 0u) {
                for (uint quarter = 0; quarter < 4u; ++quarter) {
                    const uint rowInTile = rb + quarter * 8u;
                    for (uint side = 0; side < 2u; ++side) {
                        float total = 0.0f;
                        for (uint s = 0; s < Simdgroups; ++s) total += sumStage[(rowInTile * 2u + side) * Simdgroups + s];
                        outSums[(r0 + rowInTile) * (p.outDim / 64u) + n0 / 64u + side] = total;
                    }
                }
            }
            return;
        }
    } else {
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (!accumulated.is_valid_element(i)) continue;
            const auto index = accumulated.get_multidimensional_index(i);
            const uint row = r0 + uint(index[1]);
            if (row >= p.rows) continue;
            const float scale = rowInv[row];
            const uint position = row * p.outStride + n0 + uint(index[0]);
            outB[position] = bfloat(sp_mlp_silu(float(gate[position]) * scale) * (accumulated[i] * scale));
        }
    }

    // Per-64 sums of the rounded values, read back from the tile once every thread has stored:
    // M rows x (N / 64) groups, shared out over the simdgroups, each summed across its lanes
    // in the order sp_silu_mul_na uses, so a sum is the shipped one bit for bit when its values
    // are.
    threadgroup_barrier(mem_flags::mem_device);
    constexpr uint groupsPerTile = N / 64;
    for (uint task = simdgroup; task < M * groupsPerTile; task += Simdgroups) {
        const uint row = r0 + task / groupsPerTile;
        const uint side = task % groupsPerTile;
        if constexpr (!Whole) { if (row >= p.rows) continue; }
        const uint origin = row * p.outStride + n0 + side * 64u + lane;
        const float total = simd_sum(float(outB[origin]) + float(outB[origin + 32u]));
        if (lane == 0u) outSums[row * (p.outDim / 64u) + n0 / 64u + side] = total;
    }
}

#define SP_MLP_UP_SILU_ARGUMENTS(GATE)                                                            \
                 device uchar *packed [[buffer(0)]],                                              \
                 device bfloat *scales [[buffer(1)]],                                             \
                 device bfloat *biases [[buffer(2)]],                                             \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device const GATE *gate [[buffer(4)]],                                           \
                 device bfloat *outB [[buffer(5)]],                                               \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpMlpFusedParams &p [[buffer(7)]],                                      \
                 device float *outSums [[buffer(8)]],                                             \
                 device const float *rowInv [[buffer(9)]],                                        \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint lane [[thread_index_in_simdgroup]],                                         \
                 uint simdgroup [[simdgroup_index_in_threadgroup]]

// Sums read back from the stored tile.
#define SP_DEFINE_MLP_UP_SILU(NAME, WHOLE, GATE)                                                  \
kernel void NAME(SP_MLP_UP_SILU_ARGUMENTS(GATE))                                                  \
{                                                                                                 \
    sp_mlp_up_silu<32, 128, 4, WHOLE, false, GATE>(packed, scales, biases, a, gate, outB, sums,   \
                                                   p, outSums, rowInv, group, lane, simdgroup,    \
                                                   nullptr);                                      \
}

// Sums from registers: one partial per (row in tile, 64-column group, simdgroup).
#define SP_DEFINE_MLP_UP_SILU_REGISTERS(NAME, GATE)                                               \
kernel void NAME(SP_MLP_UP_SILU_ARGUMENTS(GATE))                                                  \
{                                                                                                 \
    threadgroup float sumStage[32 * 2 * 4];                                                       \
    sp_mlp_up_silu<32, 128, 4, true, true, GATE>(packed, scales, biases, a, gate, outB, sums, p,  \
                                                 outSums, rowInv, group, lane, simdgroup,         \
                                                 sumStage);                                       \
}

// Threadgroup grid: (row tiles in a block, up to 4; outDim / 128; row blocks), 128 threads per
// group; row tile = firstTile + z * 4 + x. `rows` bounds the tiles: the whole-tile kernels are
// given a multiple of 32, the partial one the step's row count with firstTile at its last tile.
SP_DEFINE_MLP_UP_SILU(sp_mlp_up_silu_m32n128s4, true, float)
SP_DEFINE_MLP_UP_SILU(sp_mlp_up_silu_m32n128s4_partial, false, float)
SP_DEFINE_MLP_UP_SILU_REGISTERS(sp_mlp_up_silu_m32n128s4_regsums, float)
SP_DEFINE_MLP_UP_SILU(sp_mlp_up_silu_m32n128s4_bgate, true, bfloat)
SP_DEFINE_MLP_UP_SILU_REGISTERS(sp_mlp_up_silu_m32n128s4_bgate_regsums, bfloat)

// The gate GEMM of the bf16-gate variants: sp_na_wide (whole 32-row tiles, no residual) with
// its result rounded to bf16 and stored by the accelerator. Buffers as the wide GEMM's, the
// output at index 4 being bf16. Threadgroup grid as above.
kernel void sp_mlp_gate_bf16_m32n128s4(device uchar *packed [[buffer(0)]],
                                       device bfloat *scales [[buffer(1)]],
                                       device bfloat *biases [[buffer(2)]],
                                       device bfloat *a [[buffer(3)]],
                                       device bfloat *out [[buffer(4)]],
                                       device const float *sums [[buffer(6)]],
                                       constant SpMlpFusedParams &p [[buffer(7)]],
                                       uint3 group [[threadgroup_position_in_grid]])
{
    constexpr ushort M = 32, N = 128;
    const uint tile = group.y;
    const uint n0 = tile * N;
    const uint r0 = (p.firstTile + group.z * SP_MLP_ROW_BLOCK + group.x) * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto a0 = activations.slice<64, M>(0, 0);
    device uchar *tileWeights = packed + ulong(tile) * p.groups * (N * 32u);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first(
        tileWeights, dextents<int, 2>{64, N}, array<int, 2>{1, 64});
    auto b0 = first.slice<64, N>(0, 0);
    auto accumulated = operation.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
    for (uint g = 0; g < p.groups; ++g) {
        auto as = activations.slice<64, M>(int(g * 64u), 0);
        tensor<device uint4b_format, dextents<int, 2>, tensor_inline> weights(
            tileWeights + ulong(g) * (N * 32u), dextents<int, 2>{64, N}, array<int, 2>{1, 64});
        auto bs = weights.slice<64, N>(0, 0);
        auto partial = operation.get_destination_cooperative_tensor<decltype(as), decltype(bs), float>();
        operation.run(as, bs, partial);
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            const auto index = accumulated.get_multidimensional_index(i);
            const ulong parameter = (ulong(tile) * p.groups + g) * N + index[0];
            accumulated[i] += partial[i] * float(scales[parameter])
                + sums[(r0 + uint(index[1])) * p.groups + g] * float(biases[parameter]);
        }
    }
    auto rounded = operation.get_destination_cooperative_tensor<decltype(a0), decltype(b0), bfloat>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) rounded[i] = bfloat(accumulated[i]);
    auto destination = tensor(out + ulong(r0) * p.outStride, dextents<int, 2>{int(p.outStride), M},
                              array<int, 2>{1, int(p.outStride)});
    rounded.store(destination.slice<N, M>(int(n0), 0));
}
