#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// engine_na.metal — q4 GEMM on the M5 GPU neural accelerators.
//
// `matmul2d` multiplies bf16 activations against raw 4-bit weight codes in hardware, so the
// weights are never dequantised in the shader. The affine part is applied per quant group:
//
//   sum_k a_k (s q_k + b) = s * (sum_k a_k q_k) + b * (sum_k a_k)
//
// The first factor is the hardware matmul over one 64-wide group; the second needs only the
// per-group activation sums, which `sp_na_prepare` produces along with the bf16 activations.
//
// The weight tensor is addressed in place in the MLX row-major artifact: row stride is the full
// logical row (a multiple of 128 bytes), and each group is a 64-element slice of it.

inline float sp_na_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline float4 sp_na_bf16x4(device const ushort *p) {
    return as_type<float4>(uint4(ushort4(*(device const packed_ushort4 *)p)) << 16);
}

// Operand of the GEMM that follows a residual projection, emitted by that projection: the new
// hidden state times the next RMSNorm's weight, as bf16, with its per-64 sums, plus the
// per-64 sums of squares the norm's per-row scale is computed from. The scale itself is applied
// by whatever consumes the following GEMM's output (a GEMM is linear in its input), so the
// normalisation needs no stage of its own.
struct SpNaEmit {
    device const ushort *weight;
    device bfloat *values;
    device float *sums;
    device float *squares;
};

struct SpNaPrepareParams { uint rows; uint groups; uint inner; };

// fp32 activations -> bf16 activations + per-group sums of the rounded values.
// Threadgroup grid: (groups, rows), 32 threads per group.
kernel void sp_na_prepare(device const float *input [[buffer(0)]],
                          device bfloat *output [[buffer(1)]],
                          device float *sums [[buffer(2)]],
                          constant SpNaPrepareParams &p [[buffer(3)]],
                          uint3 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= p.groups || group.y >= p.rows) return;
    const uint base = group.y * p.inner + group.x * 64u + lane;
    const bfloat v0 = bfloat(input[base]);
    const bfloat v1 = bfloat(input[base + 32u]);
    output[base] = v0;
    output[base + 32u] = v1;
    const float total = simd_sum(float(v0) + float(v1));
    if (lane == 0u) sums[group.y * p.groups + group.x] = total;
}

struct SpNaParams {
    uint rows;
    uint outDim;
    uint inner;
    uint groups;
    uint hasResidual;
    uint outStride;
};

template <ushort M, ushort N, ushort Simdgroups, bool Relaxed, class Act>
inline void sp_na_tile(device uchar *packed, device const ushort *scales, device const ushort *biases,
                       device Act *a, device float *out, device const float *residual,
                       device const float *sums, constant SpNaParams &p, uint3 group)
{
    const uint n0 = group.x * N;
    const uint r0 = group.y * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;

    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> weights(
        packed + ulong(n0) * (p.inner / 2u), dextents<int, 2>{int(p.inner), N},
        array<int, 2>{1, int(p.inner)});

    constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, Relaxed);
    matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
    auto a0 = activations.template slice<64, M>(0, 0);
    auto b0 = weights.template slice<64, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;

    for (uint g = 0; g < p.groups; ++g) {
        auto as = activations.template slice<64, M>(int(g * 64u), 0);
        auto bs = weights.template slice<64, N>(int(g * 64u), 0);
        auto partial = operation.template get_destination_cooperative_tensor<decltype(as), decltype(bs), float>();
        operation.run(as, bs, partial);
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (!accumulated.is_valid_element(i)) continue;
            const auto index = accumulated.get_multidimensional_index(i);
            const uint parameter = (n0 + uint(index[0])) * p.groups + g;
            accumulated[i] += partial[i] * sp_na_bf16(scales[parameter])
                + sums[(r0 + uint(index[1])) * p.groups + g] * sp_na_bf16(biases[parameter]);
        }
    }
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        if (!accumulated.is_valid_element(i)) continue;
        const auto index = accumulated.get_multidimensional_index(i);
        const uint row = r0 + uint(index[1]);
        if (row >= p.rows) continue;
        const uint position = row * p.outStride + n0 + uint(index[0]);
        out[position] = p.hasResidual != 0u ? accumulated[i] + residual[position] : accumulated[i];
    }
}

#define SP_DEFINE_NA(NAME, M, N, SG)                                                              \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]])                                    \
{                                                                                                 \
    sp_na_tile<M, N, SG, false, bfloat>(packed, scales, biases, a, out, residual, sums, p, group);               \
}

// Threadgroup grid: (outDim / N, ceil(rows / M)), SG * 32 threads per group.
SP_DEFINE_NA(sp_gemm_q4_na_m32n128s8, 32, 128, 8)
SP_DEFINE_NA(sp_gemm_q4_na_m32n128s4, 32, 128, 4)
SP_DEFINE_NA(sp_gemm_q4_na_m32n64s4, 32, 64, 4)
SP_DEFINE_NA(sp_gemm_q4_na_m16n128s4, 16, 128, 4)
SP_DEFINE_NA(sp_gemm_q4_na_m8n128s4, 8, 128, 4)
SP_DEFINE_NA(sp_gemm_q4_na_m64n64s4, 64, 64, 4)
SP_DEFINE_NA(sp_gemm_q4_na_m64n128s8, 64, 128, 8)
SP_DEFINE_NA(sp_gemm_q4_na_m64n128s4, 64, 128, 4)
SP_DEFINE_NA(sp_gemm_q4_na_m128n64s8, 128, 64, 8)
SP_DEFINE_NA(sp_gemm_q4_na_m128n32s4, 128, 32, 4)

#define SP_DEFINE_NA_X(NAME, M, N, SG, RELAXED, ACT)                                              \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device ACT *a [[buffer(3)]],                                                     \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]])                                    \
{                                                                                                 \
    sp_na_tile<M, N, SG, RELAXED, ACT>(packed, scales, biases, a, out, residual, sums, p, group); \
}
SP_DEFINE_NA_X(sp_gemm_q4_na_m32n128s4_relaxed, 32, 128, 4, true, bfloat)
SP_DEFINE_NA_X(sp_gemm_q4_na_m32n128s4_half, 32, 128, 4, false, half)
SP_DEFINE_NA_X(sp_gemm_q4_na_m32n128s4_halfrelaxed, 32, 128, 4, true, half)

// fp32 -> fp16 variant of sp_na_prepare.
kernel void sp_na_prepare_half(device const float *input [[buffer(0)]],
                               device half *output [[buffer(1)]],
                               device float *sums [[buffer(2)]],
                               constant SpNaPrepareParams &p [[buffer(3)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= p.groups || group.y >= p.rows) return;
    const uint base = group.y * p.inner + group.x * 64u + lane;
    const half v0 = half(input[base]);
    const half v1 = half(input[base + 32u]);
    output[base] = v0;
    output[base + 32u] = v1;
    const float total = simd_sum(float(v0) + float(v1));
    if (lane == 0u) sums[group.y * p.groups + group.x] = total;
}

// ---------------------------------------------------------------------------------------------
// Tiled weight layout.
//
// Weights re-ordered as [tile of 128 rows][quant group][row in tile][64 codes], so the block
// one accelerator matmul consumes (128 rows x 64 codes = 4 KiB) is contiguous, and so are the
// 128 scales and biases its epilogue needs. Sidecars are [tile][group][row].

#define SP_NA_ROW_BLOCK 4u

// `Code` and `RowBytes` are the weight's code and the bytes 64 of them take: the 4-bit format
// and 32 for q4, `uchar` and 64 for the 8-bit pack (see the q8 kernels further down).
template <ushort M, ushort Simdgroups, ushort Mode = 0, ushort N = 128, class Code = uint4b_format, ushort RowBytes = 32>
inline void sp_na_tiled(device uchar *packed, device const ushort *scales, device const ushort *biases,
                        device bfloat *a, device float *out, device const float *residual,
                        device const float *sums, constant SpNaParams &p, uint3 group,
                        threadgroup float *biasStage = nullptr, uint __thread_index = 0,
                        SpNaEmit emit = SpNaEmit(), threadgroup float *emitStage = nullptr)
{
    // Grid is (row tiles in a block, weight tiles, row blocks). Row tiles vary fastest, so
    // consecutive threadgroups reuse one weight tile from cache; a block is SP_NA_ROW_BLOCK row
    // tiles, small enough that its activations stay in cache while every weight tile passes
    // over them. A step wider than one block streams the weights once per block, within the
    // same dispatch.
    const uint tile = group.y;
    const uint n0 = tile * N;
    const uint r0 = (group.z * SP_NA_ROW_BLOCK + group.x) * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;

    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    // Mode 6 (timing probe): every tile reads the first tile's weights, so nothing streams from memory.
    device uchar *tileWeights = packed + (Mode == 6 ? 0ul : ulong(tile) * p.groups * (N * uint(RowBytes)));
    constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, false);
    matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
    auto a0 = activations.template slice<64, M>(0, 0);
    tensor<device Code, dextents<int, 2>, tensor_inline> first(
        tileWeights, dextents<int, 2>{64, N}, array<int, 2>{1, 64});
    auto b0 = first.template slice<64, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    const uint parameterBase = tile * p.groups * N;
    if constexpr (Mode == 3) {
        // Bias term as one accelerator matmul: [rows x groups] sums times [groups x N] biases.
        // Sums here are row-major [row][group]; tiled biases are [group][column].
        auto sumTensor = tensor((device float *)sums + ulong(r0) * p.groups,
                                dextents<int, 2>{int(p.groups), M}, array<int, 2>{1, int(p.groups)});
        auto biasTensor = tensor((device bfloat *)biases + parameterBase,
                                 dextents<int, 2>{N, int(p.groups)}, array<int, 2>{1, N});
        constexpr auto biasDescriptor = matmul2d_descriptor(M, N, dynamic_length_v<int>, false, false, false);
        matmul2d<biasDescriptor, execution_simdgroups<Simdgroups>> biasOperation;
        // The destination type is bound to its operand types and the element partition differs
        // between operand types, so the bias product is stored to threadgroup memory and read
        // back by explicit (column, row) index.
        auto biasTerm = biasOperation.template get_destination_cooperative_tensor<decltype(sumTensor), decltype(biasTensor), float>();
        biasOperation.run(sumTensor, biasTensor, biasTerm);
        auto staged = tensor(biasStage, dextents<int, 2>{N, M}, array<int, 2>{1, N});
        biasTerm.store(staged);
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (!accumulated.is_valid_element(i)) { accumulated[i] = 0.0f; continue; }
            const auto index = accumulated.get_multidimensional_index(i);
            accumulated[i] = biasStage[uint(index[1]) * N + uint(index[0])];
        }
    } else {
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
    }

    if constexpr (Mode == 5) {
        // Partition probe: each thread of threadgroup (0, 0) records its (column, row) pairs.
        const uint tid = __thread_index;
        if (group.x == 0u && group.y == 0u) {
            out[tid * 80u] = float(accumulated.get_capacity());
            for (ushort i = 0; i < accumulated.get_capacity() && i < 39; ++i) {
                const auto index = accumulated.get_multidimensional_index(i);
                out[tid * 80u + 1u + i * 2u] = accumulated.is_valid_element(i) ? float(index[0]) : -1.0f;
                out[tid * 80u + 2u + i * 2u] = float(index[1]);
            }
        }
        return;
    }
    device const float *tileSums = sums + ulong(group.z * SP_NA_ROW_BLOCK + group.x) * p.groups * M;
    // For the 32 x 128 tile on four simdgroups the accelerator gives every thread a regular block:
    // columns c0 ..< c0 + 4 and c0 + 64 ..< c0 + 68, rows rb, rb + 8, rb + 16, rb + 24, in element
    // order [row pair][column run][row][column]. When that holds (checked per thread) the
    // epilogue needs four vector loads of scales and biases and four sums per quant group,
    // instead of three scalar loads per element.
    bool regular = false;
    uint c0 = 0u, rb = 0u;
    if constexpr ((Mode == 0 || Mode == 7) && M == 32 && Simdgroups == 4 && N == 128) {
        const auto origin = accumulated.get_multidimensional_index(ushort(0));
        c0 = uint(origin[0]); rb = uint(origin[1]);
        regular = accumulated.get_capacity() == 32;
#pragma unroll
        for (ushort i = 0; i < 32; ++i) {
            const auto index = accumulated.get_multidimensional_index(regular ? i : ushort(0));
            const ushort block = i >> 2;
            regular = regular && accumulated.is_valid_element(i)
                && uint(index[0]) == c0 + (i & 3) + ((block >> 1) & 1) * 64u
                && uint(index[1]) == rb + ((block & 1) + 2 * (block >> 2)) * 8u;
        }
    }
    for (uint g = 0; g < p.groups; ++g) {
        auto as = activations.template slice<64, M>(int(g * 64u), 0);
        tensor<device Code, dextents<int, 2>, tensor_inline> weights(
            tileWeights + ulong(g) * (N * uint(RowBytes)), dextents<int, 2>{64, N}, array<int, 2>{1, 64});
        auto bs = weights.template slice<64, N>(0, 0);
        auto partial = operation.template get_destination_cooperative_tensor<decltype(as), decltype(bs), float>();
        operation.run(as, bs, partial);
        if (regular) {
            const uint pb = parameterBase + g * N + c0;
            const float4 scaleA = sp_na_bf16x4(scales + pb), scaleB = sp_na_bf16x4(scales + pb + 64u);
            const float4 biasA = sp_na_bf16x4(biases + pb), biasB = sp_na_bf16x4(biases + pb + 64u);
            device const float *rowSums = sums + (r0 + rb) * p.groups + g;
            const float4 sum = float4(rowSums[0], rowSums[8u * p.groups], rowSums[16u * p.groups], rowSums[24u * p.groups]);
#pragma unroll
            for (ushort block = 0; block < 8; ++block) {
                const float4 scale = ((block >> 1) & 1) != 0 ? scaleB : scaleA;
                const float4 bias = ((block >> 1) & 1) != 0 ? biasB : biasA;
                const float rowSum = sum[(block & 1) + 2 * (block >> 2)];
#pragma unroll
                for (ushort e = 0; e < 4; ++e) {
                    accumulated[block * 4 + e] += partial[block * 4 + e] * scale[e] + rowSum * bias[e];
                }
            }
            continue;
        }
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (!accumulated.is_valid_element(i)) continue;
            const auto index = accumulated.get_multidimensional_index(i);
            const uint parameter = parameterBase + g * N + uint(index[0]);
            if constexpr (Mode == 1 || Mode == 6) {          // timing probe: no epilogue at all
                accumulated[i] += partial[i];
            } else if constexpr (Mode == 2 || Mode == 3) {   // scale term only (Mode 3 adds bias by matmul)
                accumulated[i] += partial[i] * sp_na_bf16(scales[parameter]);
            } else if constexpr (Mode == 4) {   // timing probe: sums read in tile layout
                accumulated[i] += partial[i] * sp_na_bf16(scales[parameter])
                    + tileSums[g * M + uint(index[1])] * sp_na_bf16(biases[parameter]);
            } else {
                accumulated[i] += partial[i] * sp_na_bf16(scales[parameter])
                    + sums[(r0 + uint(index[1])) * p.groups + g] * sp_na_bf16(biases[parameter]);
            }
        }
    }
    if (regular) {
        const bool emitting = Mode == 7;
        const uint lane = __thread_index & 31u, sg = __thread_index >> 5;
        // A tile whose rows are all in the step is written by the accelerator's own store,
        // which costs next to nothing; writing it from the threads, a few values each, was
        // 17 ms of a 128-row step. A partial last tile is still written by hand, so nothing
        // lands on rows the step does not own (some outputs are sized to the rows in use).
        const bool whole = r0 + M <= p.rows;
        if constexpr (Mode != 7) {
            if (whole) {
                if (p.hasResidual != 0u) {
#pragma unroll
                    for (ushort block = 0; block < 8; ++block) {
                        const uint row = r0 + rb + ((block & 1) + 2 * (block >> 2)) * 8u;
                        const float4 prior = float4(*(device const packed_float4 *)(residual + row * p.outStride + n0 + c0 + ((block >> 1) & 1) * 64u));
#pragma unroll
                        for (ushort e = 0; e < 4; ++e) accumulated[block * 4 + e] += prior[e];
                    }
                }
                auto destination = tensor(out + ulong(r0) * p.outStride, dextents<int, 2>{int(p.outStride), M},
                                          array<int, 2>{1, int(p.outStride)});
                accumulated.store(destination.template slice<N, M>(int(n0), 0));
                return;
            }
        }
        auto rounded16 = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), bfloat>();
#pragma unroll
        for (ushort block = 0; block < 8; ++block) {
            const uint rowInTile = rb + ((block & 1) + 2 * (block >> 2)) * 8u;
            const uint row = r0 + rowInTile;
            const bool live = row < p.rows;
            if (!emitting && !live) continue;
            const uint side = (block >> 1) & 1;
            const uint position = row * p.outStride + n0 + c0 + side * 64u;
            float4 value = float4(accumulated[block * 4], accumulated[block * 4 + 1],
                                  accumulated[block * 4 + 2], accumulated[block * 4 + 3]);
            if (p.hasResidual != 0u && live) value += float4(*(device const packed_float4 *)(residual + position));
            if (whole) {
#pragma unroll
                for (ushort e = 0; e < 4; ++e) accumulated[block * 4 + e] = value[e];
            } else if (live) {
                *(device packed_float4 *)(out + position) = value;
            }
            if (emitting) {
                // The sixteen quads of one (row, 64-column group) sit in the four lanes of each
                // simdgroup that share this thread's rows (lane bits 0 and 3 select the column
                // run): two xor-shuffles here, then a four-way sum across simdgroups through
                // threadgroup memory.
                const bfloat4 rounded = bfloat4(value * sp_na_bf16x4(emit.weight + n0 + c0 + side * 64u));
                const float4 back = float4(rounded);
                float sum = back.x + back.y + back.z + back.w, square = dot(value, value);
                sum += simd_shuffle_xor(sum, ushort(1)); sum += simd_shuffle_xor(sum, ushort(8));
                square += simd_shuffle_xor(square, ushort(1)); square += simd_shuffle_xor(square, ushort(8));
                if (whole) {
#pragma unroll
                    for (ushort e = 0; e < 4; ++e) rounded16[block * 4 + e] = rounded[e];
                } else if (live) {
                    *(device packed_bfloat4 *)(emit.values + position) = rounded;
                }
                if ((lane & 9u) == 0u) {
                    emitStage[((rowInTile * 2u + side) * 4u + sg) * 2u] = sum;
                    emitStage[((rowInTile * 2u + side) * 4u + sg) * 2u + 1u] = square;
                }
            }
        }
        if (whole) {
            auto destination = tensor(out + ulong(r0) * p.outStride, dextents<int, 2>{int(p.outStride), M},
                                      array<int, 2>{1, int(p.outStride)});
            accumulated.store(destination.template slice<N, M>(int(n0), 0));
            if (emitting) {
                auto operand = tensor(emit.values + ulong(r0) * p.outStride, dextents<int, 2>{int(p.outStride), M},
                                      array<int, 2>{1, int(p.outStride)});
                rounded16.store(operand.template slice<N, M>(int(n0), 0));
            }
        }
        if (emitting) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0u && (lane & 9u) == 0u) {
                for (uint quarter = 0; quarter < 4u; ++quarter) {
                    const uint rowInTile = rb + quarter * 8u;
                    if (r0 + rowInTile >= p.rows) continue;
                    for (uint side = 0; side < 2u; ++side) {
                        float sum = 0.0f, square = 0.0f;
                        for (uint s = 0; s < 4u; ++s) {
                            sum += emitStage[((rowInTile * 2u + side) * 4u + s) * 2u];
                            square += emitStage[((rowInTile * 2u + side) * 4u + s) * 2u + 1u];
                        }
                        const uint index = (r0 + rowInTile) * (p.outDim / 64u) + n0 / 64u + side;
                        emit.sums[index] = sum;
                        emit.squares[index] = square;
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        if (!accumulated.is_valid_element(i)) continue;
        const auto index = accumulated.get_multidimensional_index(i);
        const uint row = r0 + uint(index[1]);
        if (row >= p.rows) continue;
        const uint position = row * p.outStride + n0 + uint(index[0]);
        if constexpr (Mode == 7) { out[position] = NAN; continue; }
        out[position] = p.hasResidual != 0u ? accumulated[i] + residual[position] : accumulated[i];
    }
}

#define SP_DEFINE_NAT(NAME, M, SG)                                                                \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]])                                    \
{                                                                                                 \
    sp_na_tiled<M, SG>(packed, scales, biases, a, out, residual, sums, p, group);                 \
}
// Mode 7: the regular path also emits the following GEMM's operand (see SpNaEmit). A thread
// whose block is not regular poisons its outputs rather than leaving the operand unwritten.
kernel void sp_gemm_q4_nat_m32n128s4_emit(device uchar *packed [[buffer(0)]],
                 device const ushort *scales [[buffer(1)]],
                 device const ushort *biases [[buffer(2)]],
                 device bfloat *a [[buffer(3)]],
                 device float *out [[buffer(4)]],
                 device const float *residual [[buffer(5)]],
                 device const float *sums [[buffer(6)]],
                 constant SpNaParams &p [[buffer(7)]],
                 device const ushort *emitWeight [[buffer(8)]],
                 device bfloat *emitValues [[buffer(9)]],
                 device float *emitSums [[buffer(10)]],
                 device float *emitSquares [[buffer(11)]],
                 uint3 group [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float emitStage[32 * 2 * 4 * 2];
    sp_na_tiled<32, 4, 7>(packed, scales, biases, a, out, residual, sums, p, group, nullptr, tid,
                          SpNaEmit{emitWeight, emitValues, emitSums, emitSquares}, emitStage);
}

SP_DEFINE_NAT(sp_gemm_q4_nat_m8n128s4, 8, 4)
SP_DEFINE_NAT(sp_gemm_q4_nat_m16n128s4, 16, 4)
// The wide tile for steps made of whole 32-row tiles, kept deliberately small: one loop over
// quant groups, the affine correction per element, the residual if the kernel is the residual
// one, and the tile stored by the accelerator. Nothing is optional at run time. The same
// arithmetic inside sp_na_tiled (probe modes, an operand-emitting epilogue and a hand-written
// store compiled in beside it) is 7% slower per step, and a version of this kernel with the
// residual behind a run-time flag and a fallback for partial tiles was 20% slower: what a
// kernel carries, not only what it executes, decides how many of its threadgroups the GPU
// keeps running at once. Steps with a partial last tile use sp_na_tiled.
template <ushort M, ushort N, ushort Simdgroups, bool AddResidual, class Code = uint4b_format, ushort RowBytes = 32>
inline void sp_na_wide(device uchar *packed, device bfloat *scales, device bfloat *biases,
                       device bfloat *a, device float *out, device const float *residual,
                       device const float *sums, constant SpNaParams &p, uint3 group)
{
    const uint tile = group.y;
    const uint n0 = tile * N;
    const uint r0 = (group.z * SP_NA_ROW_BLOCK + group.x) * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, false);
    matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
    auto a0 = activations.template slice<64, M>(0, 0);
    device uchar *tileWeights = packed + ulong(tile) * p.groups * (N * uint(RowBytes));
    tensor<device Code, dextents<int, 2>, tensor_inline> first(
        tileWeights, dextents<int, 2>{64, N}, array<int, 2>{1, 64});
    auto b0 = first.template slice<64, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
    for (uint g = 0; g < p.groups; ++g) {
        auto as = activations.template slice<64, M>(int(g * 64u), 0);
        tensor<device Code, dextents<int, 2>, tensor_inline> weights(
            tileWeights + ulong(g) * (N * uint(RowBytes)), dextents<int, 2>{64, N}, array<int, 2>{1, 64});
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
    if constexpr (AddResidual) {
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            const auto index = accumulated.get_multidimensional_index(i);
            accumulated[i] += residual[(r0 + uint(index[1])) * p.outStride + n0 + uint(index[0])];
        }
    }
    auto destination = tensor(out + ulong(r0) * p.outStride, dextents<int, 2>{int(p.outStride), M},
                              array<int, 2>{1, int(p.outStride)});
    accumulated.store(destination.template slice<N, M>(int(n0), 0));
}
#define SP_DEFINE_NAT_WHOLE(NAME, RESIDUAL)                                                       \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device bfloat *scales [[buffer(1)]],                                             \
                 device bfloat *biases [[buffer(2)]],                                             \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]])                                    \
{                                                                                                 \
    sp_na_wide<32, 128, 4, RESIDUAL>(packed, scales, biases, a, out, residual, sums, p, group);   \
}
SP_DEFINE_NAT_WHOLE(sp_gemm_q4_nat_m32n128s4_whole, false)
SP_DEFINE_NAT_WHOLE(sp_gemm_q4_nat_m32n128s4_whole_residual, true)
SP_DEFINE_NAT(sp_gemm_q4_nat_m32n128s4, 32, 4)
SP_DEFINE_NAT(sp_gemm_q4_nat_m32n128s8, 32, 8)
SP_DEFINE_NAT(sp_gemm_q4_nat_m64n128s8, 64, 8)

// Prepare for the tiled kernels: as sp_na_prepare, but sums are written [row tile][group][row
// in tile] so the sums one epilogue reads are contiguous. `inner` carries the tile height in
// its top byte-free companion `tileRows`.
struct SpNaPrepareTiledParams { uint rows; uint groups; uint inner; uint tileRows; };
kernel void sp_na_prepare_tiled(device const float *input [[buffer(0)]],
                                device bfloat *output [[buffer(1)]],
                                device float *sums [[buffer(2)]],
                                constant SpNaPrepareTiledParams &p [[buffer(3)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= p.groups || group.y >= p.rows) return;
    const uint base = group.y * p.inner + group.x * 64u + lane;
    const bfloat v0 = bfloat(input[base]);
    const bfloat v1 = bfloat(input[base + 32u]);
    output[base] = v0;
    output[base + 32u] = v1;
    const float total = simd_sum(float(v0) + float(v1));
    if (lane == 0u) {
        sums[((group.y / p.tileRows) * p.groups + group.x) * p.tileRows + group.y % p.tileRows] = total;
    }
}

// Timing probes for a different way round the q4 epilogue: the accelerator accumulates every
// quant group into one destination (multiply_accumulate), and between groups the destination
// is rescaled in place from the last group's scale to the next one's. One cooperative tensor
// instead of two, and one FMA per element and group instead of a multiply and an FMA.
// Mode bit 0: do the in-place rescale (with the ordinary scale and bias sidecars, so the values
// are not the GEMM's; only the time is of interest). Mode bit 1: real weights instead of every
// tile reading the first.
template <ushort M, ushort Simdgroups, ushort Mode, ushort N = 128>
inline void sp_na_acc(device uchar *packed, device const ushort *scales, device const ushort *biases,
                      device bfloat *a, device float *out, device const float *sums,
                      constant SpNaParams &p, uint3 group)
{
    const uint tile = group.y;
    const uint n0 = tile * N;
    const uint r0 = (group.z * SP_NA_ROW_BLOCK + group.x) * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    device uchar *tileWeights = packed + ((Mode & 2) != 0 ? ulong(tile) * p.groups * (N * 32u) : 0ul);
    constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
    auto a0 = activations.template slice<64, M>(0, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first(
        tileWeights, dextents<int, 2>{64, N}, array<int, 2>{1, 64});
    auto b0 = first.template slice<64, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
    const uint parameterBase = tile * p.groups * N;
    const auto origin = accumulated.get_multidimensional_index(ushort(0));
    const uint c0 = uint(origin[0]), rb = uint(origin[1]);
    for (uint g = 0; g < p.groups; ++g) {
        if ((Mode & 1) != 0) {
            const uint pb = parameterBase + g * N + c0;
            const float4 scaleA = sp_na_bf16x4(scales + pb), scaleB = sp_na_bf16x4(scales + pb + 64u);
            const float4 biasA = sp_na_bf16x4(biases + pb), biasB = sp_na_bf16x4(biases + pb + 64u);
            device const float *rowSums = sums + (r0 + rb) * p.groups + g;
            const float4 sum = float4(rowSums[0], rowSums[8u * p.groups], rowSums[16u * p.groups], rowSums[24u * p.groups]);
#pragma unroll
            for (ushort block = 0; block < 8; ++block) {
                const float4 scale = ((block >> 1) & 1) != 0 ? scaleB : scaleA;
                const float4 bias = ((block >> 1) & 1) != 0 ? biasB : biasA;
                const float rowSum = sum[(block & 1) + 2 * (block >> 2)];
#pragma unroll
                for (ushort e = 0; e < 4; ++e) {
                    accumulated[block * 4 + e] = accumulated[block * 4 + e] * scale[e] + rowSum * bias[e];
                }
            }
        }
        auto as = activations.template slice<64, M>(int(g * 64u), 0);
        tensor<device uint4b_format, dextents<int, 2>, tensor_inline> weights(
            tileWeights + ulong(g) * (N * 32u), dextents<int, 2>{64, N}, array<int, 2>{1, 64});
        auto bs = weights.template slice<64, N>(0, 0);
        operation.run(as, bs, accumulated);
    }
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        if (!accumulated.is_valid_element(i)) continue;
        const auto index = accumulated.get_multidimensional_index(i);
        const uint row = r0 + uint(index[1]);
        if (row >= p.rows) continue;
        out[row * p.outStride + n0 + uint(index[0])] = accumulated[i];
    }
}
// Timing probe: one threadgroup takes R row tiles through each weight chunk in turn, to see
// whether consecutive runs over the same weights cost less than runs over different ones.
template <ushort M, ushort Simdgroups, ushort R, ushort N = 128>
inline void sp_na_acc_rows(device uchar *packed, device bfloat *a, device float *out,
                           constant SpNaParams &p, uint3 group)
{
    const uint tile = group.y;
    const uint n0 = tile * N;
    if ((group.x % R) != 0u) return;
    const uint r0 = (group.z * SP_NA_ROW_BLOCK + group.x) * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    device uchar *tileWeights = packed;
    constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
    auto act0 = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M}, array<int, 2>{1, int(p.inner)});
    auto act1 = tensor(a + ulong(r0 + (R > 1 ? M : 0)) * p.inner, dextents<int, 2>{int(p.inner), M}, array<int, 2>{1, int(p.inner)});
    auto act2 = tensor(a + ulong(r0 + (R > 2 ? 2 * M : 0)) * p.inner, dextents<int, 2>{int(p.inner), M}, array<int, 2>{1, int(p.inner)});
    auto act3 = tensor(a + ulong(r0 + (R > 3 ? 3 * M : 0)) * p.inner, dextents<int, 2>{int(p.inner), M}, array<int, 2>{1, int(p.inner)});
    auto a0 = act0.template slice<64, M>(0, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first(
        tileWeights, dextents<int, 2>{64, N}, array<int, 2>{1, 64});
    auto b0 = first.template slice<64, N>(0, 0);
    auto acc0 = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto acc1 = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto acc2 = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto acc3 = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) { acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f; }
    for (uint g = 0; g < p.groups; ++g) {
        tensor<device uint4b_format, dextents<int, 2>, tensor_inline> weights(
            tileWeights + ulong(g) * (N * 32u), dextents<int, 2>{64, N}, array<int, 2>{1, 64});
        auto bs = weights.template slice<64, N>(0, 0);
        auto s0 = act0.template slice<64, M>(int(g * 64u), 0);
        auto s1 = act1.template slice<64, M>(int(g * 64u), 0);
        auto s2 = act2.template slice<64, M>(int(g * 64u), 0);
        auto s3 = act3.template slice<64, M>(int(g * 64u), 0);
        operation.run(s0, bs, acc0);
        if (R > 1) operation.run(s1, bs, acc1);
        if (R > 2) operation.run(s2, bs, acc2);
        if (R > 3) operation.run(s3, bs, acc3);
    }
#pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) {
        if (!acc0.is_valid_element(i)) continue;
        const auto index = acc0.get_multidimensional_index(i);
        const uint position = uint(index[1]) * p.outStride + n0 + uint(index[0]);
        out[r0 * p.outStride + position] = acc0[i];
        if (R > 1) out[(r0 + M) * p.outStride + position] = acc1[i];
        if (R > 2) out[(r0 + 2 * M) * p.outStride + position] = acc2[i];
        if (R > 3) out[(r0 + 3 * M) * p.outStride + position] = acc3[i];
    }
}
#define SP_DEFINE_NAT_ACC_ROWS(NAME, M, SG, R)                                                    \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]])                                    \
{                                                                                                 \
    sp_na_acc_rows<M, SG, R>(packed, a, out, p, group);                                           \
}
SP_DEFINE_NAT_ACC_ROWS(sp_gemm_q4_nat_m32n128s4_accrows1, 32, 4, 1)
SP_DEFINE_NAT_ACC_ROWS(sp_gemm_q4_nat_m32n128s4_accrows2, 32, 4, 2)
SP_DEFINE_NAT_ACC_ROWS(sp_gemm_q4_nat_m32n128s4_accrows4, 32, 4, 4)

#define SP_DEFINE_NAT_ACC(NAME, M, SG, MODE)                                                      \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]])                                    \
{                                                                                                 \
    sp_na_acc<M, SG, MODE>(packed, scales, biases, a, out, sums, p, group);                       \
}
SP_DEFINE_NAT_ACC(sp_gemm_q4_nat_m32n128s4_accalias, 32, 4, 0)
SP_DEFINE_NAT_ACC(sp_gemm_q4_nat_m32n128s4_accscalealias, 32, 4, 1)
SP_DEFINE_NAT_ACC(sp_gemm_q4_nat_m32n128s4_acc, 32, 4, 2)
SP_DEFINE_NAT_ACC(sp_gemm_q4_nat_m32n128s4_accscale, 32, 4, 3)

#define SP_DEFINE_NAT_MODE(NAME, M, SG, MODE)                                                     \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]])                                    \
{                                                                                                 \
    threadgroup float biasStage[MODE == 3 ? M * 128 : 1];                                         \
    sp_na_tiled<M, SG, MODE>(packed, scales, biases, a, out, residual, sums, p, group, biasStage);\
}
kernel void sp_gemm_q4_nat_m32n128s4_partition(device uchar *packed [[buffer(0)]],
                 device const ushort *scales [[buffer(1)]],
                 device const ushort *biases [[buffer(2)]],
                 device bfloat *a [[buffer(3)]],
                 device float *out [[buffer(4)]],
                 device const float *residual [[buffer(5)]],
                 device const float *sums [[buffer(6)]],
                 constant SpNaParams &p [[buffer(7)]],
                 uint3 group [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]])
{
    sp_na_tiled<32, 4, 5>(packed, scales, biases, a, out, residual, sums, p, group, nullptr, tid);
}
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m32n128s4_noepi, 32, 4, 1)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m32n128s4_alias, 32, 4, 6)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m16n128s4_alias, 16, 4, 6)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m32n128s2_alias, 32, 2, 6)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m32n128s8_alias, 32, 8, 6)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m64n128s4_alias, 64, 4, 6)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m64n128s8_alias, 64, 8, 6)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m64n128s16_alias, 64, 16, 6)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m128n128s8_alias, 128, 8, 6)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m128n128s16_alias, 128, 16, 6)
SP_DEFINE_NA(sp_gemm_q4_na_m32n128s2, 32, 128, 2)
SP_DEFINE_NA(sp_gemm_q4_na_m64n128s16, 64, 128, 16)
SP_DEFINE_NA(sp_gemm_q4_na_m128n128s8, 128, 128, 8)
SP_DEFINE_NA(sp_gemm_q4_na_m128n128s16, 128, 128, 16)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m32n128s4_tilesums, 32, 4, 4)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m32n128s4_scaleonly, 32, 4, 2)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m32n128s4_biasmm, 32, 4, 3)
SP_DEFINE_NAT_MODE(sp_gemm_q4_nat_m16n128s4_biasmm, 16, 4, 3)
SP_DEFINE_NA(sp_gemm_q4_na_m8n256s4, 8, 256, 4)
SP_DEFINE_NA(sp_gemm_q4_na_m8n256s8, 8, 256, 8)
SP_DEFINE_NA(sp_gemm_q4_na_m16n256s8, 16, 256, 8)
SP_DEFINE_NA(sp_gemm_q4_na_m8n512s8, 8, 512, 8)
SP_DEFINE_NA(sp_gemm_q4_na_m8n128s2, 8, 128, 2)
SP_DEFINE_NA(sp_gemm_q4_na_m8n64s2, 8, 64, 2)
SP_DEFINE_NA(sp_gemm_q4_na_m8n64s1, 8, 64, 1)
SP_DEFINE_NA(sp_gemm_q4_na_m8n32s1, 8, 32, 1)

// ---------------------------------------------------------------------------------------------
// Split-K GEMM for narrow passes (a speculative verify block, or small batches).
//
// With eight rows a tile-per-threadgroup kernel leaves the GPU mostly idle: each threadgroup
// walks its quant groups one after another and there are too few threadgroups to fill the
// cores. Here a threadgroup of P simdgroups computes one 8 x 32 tile; simdgroup s runs its own
// accelerator matmuls over quant groups [s * G/P, (s+1) * G/P). Partial sums meet in
// threadgroup memory and are reduced before the single write.
// Threadgroup grid: (outDim / 32, ceil(rows / 8)), P * 32 threads per group.

template <ushort P>
inline void sp_na_split(device uchar *packed, device const ushort *scales, device const ushort *biases,
                        device bfloat *a, device float *out, device const float *residual,
                        device const float *sums, constant SpNaParams &p, uint3 group,
                        uint sg, uint tid, threadgroup float *partials)
{
    constexpr ushort M = 8, N = 32;
    const uint n0 = group.x * N;
    const uint r0 = group.y * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    const uint per = p.groups / P;
    const uint first = sg * per;

    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> weights(
        packed + ulong(n0) * (p.inner / 2u), dextents<int, 2>{int(p.inner), N},
        array<int, 2>{1, int(p.inner)});
    constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, false);
    matmul2d<descriptor, execution_simdgroup> operation;
    auto a0 = activations.template slice<64, M>(0, 0);
    auto b0 = weights.template slice<64, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;

    for (uint g = first; g < first + per; ++g) {
        auto as = activations.template slice<64, M>(int(g * 64u), 0);
        auto bs = weights.template slice<64, N>(int(g * 64u), 0);
        auto partial = operation.template get_destination_cooperative_tensor<decltype(as), decltype(bs), float>();
        operation.run(as, bs, partial);
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (!accumulated.is_valid_element(i)) continue;
            const auto index = accumulated.get_multidimensional_index(i);
            const uint parameter = (n0 + uint(index[0])) * p.groups + g;
            accumulated[i] += partial[i] * sp_na_bf16(scales[parameter])
                + sums[(r0 + uint(index[1])) * p.groups + g] * sp_na_bf16(biases[parameter]);
        }
    }
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        if (!accumulated.is_valid_element(i)) continue;
        const auto index = accumulated.get_multidimensional_index(i);
        partials[(sg * M + uint(index[1])) * N + uint(index[0])] = accumulated[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint element = tid; element < uint(M) * N; element += uint(P) * 32u) {
        const uint m = element / N, n = element % N;
        if (r0 + m >= p.rows) continue;
        float total = 0.0f;
        for (uint s = 0; s < P; ++s) total += partials[(s * M + m) * N + n];
        const uint position = (r0 + m) * p.outStride + n0 + n;
        out[position] = p.hasResidual != 0u ? total + residual[position] : total;
    }
}

#define SP_DEFINE_SPLIT(NAME, P)                                                                  \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                      \
                 uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                                 \
    threadgroup float partials[P * 8 * 32];                                                       \
    sp_na_split<P>(packed, scales, biases, a, out, residual, sums, p, group, sg, tid, partials);  \
}
SP_DEFINE_SPLIT(sp_gemm_q4_split4, 4)
SP_DEFINE_SPLIT(sp_gemm_q4_split8, 8)
SP_DEFINE_SPLIT(sp_gemm_q4_split16, 16)

// Split-K over the tiled layout: an M x 32 tile's weights for one quant group are 1 KiB
// contiguous, and its 32 scales and 32 biases are one cache line each.
// Threadgroup grid: (outDim / 32, ceil(rows / M)), P * 32 threads per group.
template <ushort M, ushort P, ushort N = 32, ushort Mode = 0, bool Emit = false, class Code = uint4b_format, ushort RowBytes = 32>
inline void sp_na_split_tiled(device uchar *packed, device const ushort *scales, device const ushort *biases,
                              device bfloat *a, device float *out, device const float *residual,
                              device const float *sums, constant SpNaParams &p, uint3 group,
                              uint sg, uint tid, threadgroup float *partials, SpNaEmit emit = SpNaEmit())
{
    constexpr ushort StorageN = 128;
    const uint n0 = group.x * N;
    const uint r0 = group.y * M;
    if (n0 >= p.outDim || r0 >= p.rows) return;
    const uint per = p.groups / P;
    const uint first = sg * per;
    const uint tile = n0 / StorageN, inTile = n0 % StorageN;

    auto activations = tensor(a + ulong(r0) * p.inner, dextents<int, 2>{int(p.inner), M},
                              array<int, 2>{1, int(p.inner)});
    // Element (tile, group, row-in-tile) starts at ((tile * G + group) * 128 + row) * 32 bytes
    // (RowBytes: 64 for 8-bit codes).
    // Mode 6 (timing probe): every tile reads the first tile's weights.
    device uchar *tileWeights = packed + (Mode == 6 ? 0ul : (ulong(tile) * p.groups * StorageN + inTile) * uint(RowBytes));
    const uint parameterBase = tile * p.groups * StorageN + inTile;
    constexpr auto descriptor = matmul2d_descriptor(M, N, 64, false, true, false);
    matmul2d<descriptor, execution_simdgroup> operation;
    auto a0 = activations.template slice<64, M>(0, 0);
    tensor<device Code, dextents<int, 2>, tensor_inline> firstWeights(
        tileWeights, dextents<int, 2>{64, N}, array<int, 2>{1, 64});
    auto b0 = firstWeights.template slice<64, N>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;

    if constexpr (Mode == 5) {
        // Partition probe: each thread of the first simdgroup of threadgroup (0, 0) records its
        // (column, row) pairs.
        if (group.x == 0u && group.y == 0u && sg == 0u) {
            out[tid * 80u] = float(accumulated.get_capacity());
            for (ushort i = 0; i < accumulated.get_capacity() && i < 39; ++i) {
                const auto index = accumulated.get_multidimensional_index(i);
                out[tid * 80u + 1u + i * 2u] = accumulated.is_valid_element(i) ? float(index[0]) : -1.0f;
                out[tid * 80u + 2u + i * 2u] = float(index[1]);
            }
        }
        return;
    }
    // A single-simdgroup tile of 16 or 32 rows gives each thread a regular block, as the wide
    // tile does: columns c0 ..< c0 + 4 and c0 + N/2 ..< c0 + N/2 + 4, rows rb, rb + 8, (rb + 16,
    // rb + 24), in element order [row pair][column run][row][column]. Checked per thread.
    constexpr ushort Blocks = M / 4;
    bool regular = false;
    uint c0 = 0u, rb = 0u;
    if constexpr (Mode == 0 && (M == 16 || M == 32)) {
        const auto origin = accumulated.get_multidimensional_index(ushort(0));
        c0 = uint(origin[0]); rb = uint(origin[1]);
        regular = accumulated.get_capacity() == M;
#pragma unroll
        for (ushort i = 0; i < M; ++i) {
            const auto index = accumulated.get_multidimensional_index(regular ? i : ushort(0));
            const ushort block = i >> 2;
            regular = regular && accumulated.is_valid_element(i)
                && uint(index[0]) == c0 + (i & 3) + ((block >> 1) & 1) * (N / 2u)
                && uint(index[1]) == rb + ((block & 1) + 2 * (block >> 2)) * 8u;
        }
    }
    for (uint g = first; g < first + per; ++g) {
        auto as = activations.template slice<64, M>(int(g * 64u), 0);
        tensor<device Code, dextents<int, 2>, tensor_inline> weights(
            tileWeights + ulong(g) * (StorageN * uint(RowBytes)), dextents<int, 2>{64, N}, array<int, 2>{1, 64});
        auto bs = weights.template slice<64, N>(0, 0);
        auto partial = operation.template get_destination_cooperative_tensor<decltype(as), decltype(bs), float>();
        operation.run(as, bs, partial);
        if (regular) {
            const uint pb = parameterBase + g * StorageN + c0;
            const float4 scaleA = sp_na_bf16x4(scales + pb), scaleB = sp_na_bf16x4(scales + pb + N / 2u);
            const float4 biasA = sp_na_bf16x4(biases + pb), biasB = sp_na_bf16x4(biases + pb + N / 2u);
            device const float *rowSums = sums + (r0 + rb) * p.groups + g;
            const float4 sum = float4(rowSums[0], rowSums[8u * p.groups],
                                      M == 32 ? rowSums[16u * p.groups] : 0.0f, M == 32 ? rowSums[24u * p.groups] : 0.0f);
#pragma unroll
            for (ushort block = 0; block < Blocks; ++block) {
                const float4 scale = ((block >> 1) & 1) != 0 ? scaleB : scaleA;
                const float4 bias = ((block >> 1) & 1) != 0 ? biasB : biasA;
                const float rowSum = sum[(block & 1) + 2 * (block >> 2)];
#pragma unroll
                for (ushort e = 0; e < 4; ++e) {
                    accumulated[block * 4 + e] += partial[block * 4 + e] * scale[e] + rowSum * bias[e];
                }
            }
            continue;
        }
#pragma unroll
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (!accumulated.is_valid_element(i)) continue;
            const auto index = accumulated.get_multidimensional_index(i);
            const uint parameter = parameterBase + g * StorageN + uint(index[0]);
            if constexpr (Mode == 1 || Mode == 6) {      // timing probes: no epilogue
                accumulated[i] += partial[i];
            } else {
                accumulated[i] += partial[i] * sp_na_bf16(scales[parameter])
                    + sums[(r0 + uint(index[1])) * p.groups + g] * sp_na_bf16(biases[parameter]);
            }
        }
    }
    if (regular) {
#pragma unroll
        for (ushort block = 0; block < Blocks; ++block) {
            const uint row = rb + ((block & 1) + 2 * (block >> 2)) * 8u;
            const uint column = c0 + ((block >> 1) & 1) * (N / 2u);
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
        const bool live = r0 + m < p.rows;
        if constexpr (!Emit) { if (!live) continue; }
        float4 total = float4(0.0f);
        for (uint s = 0; s < P; ++s) total += float4(*(threadgroup packed_float4 *)(partials + (s * M + m) * N + n));
        const uint position = (r0 + m) * p.outStride + n0 + n;
        if (p.hasResidual != 0u && live) total += float4(*(device const packed_float4 *)(residual + position));
        if (live) *(device packed_float4 *)(out + position) = total;
        if constexpr (Emit) {
            // With N = 64 a row's sixteen quads fall on sixteen consecutive lanes, so the row's
            // sums are masked reductions over half a simdgroup.
            const bfloat4 rounded = bfloat4(total * sp_na_bf16x4(emit.weight + n0 + n));
            const float4 back = float4(rounded);
            const float mine = back.x + back.y + back.z + back.w, square = dot(total, total);
            const uint lane = tid & 31u;
            const float sumLow = simd_sum(lane < 16u ? mine : 0.0f), sumHigh = simd_sum(lane < 16u ? 0.0f : mine);
            const float squareLow = simd_sum(lane < 16u ? square : 0.0f), squareHigh = simd_sum(lane < 16u ? 0.0f : square);
            if (live) {
                *(device packed_bfloat4 *)(emit.values + position) = rounded;
                if ((lane & 15u) == 0u) {
                    const uint index = (r0 + m) * (p.outDim / 64u) + n0 / 64u;
                    emit.sums[index] = lane < 16u ? sumLow : sumHigh;
                    emit.squares[index] = lane < 16u ? squareLow : squareHigh;
                }
            }
        }
    }
}

#define SP_DEFINE_SPLIT_TILED(NAME, M, P)                                                         \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                      \
                 uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                                 \
    threadgroup float partials[P * M * 32];                                                       \
    sp_na_split_tiled<M, P>(packed, scales, biases, a, out, residual, sums, p, group, sg, tid, partials); \
}
SP_DEFINE_SPLIT_TILED(sp_gemm_q4_split_tiled1, 8, 1)
SP_DEFINE_SPLIT_TILED(sp_gemm_q4_split_tiled4, 8, 4)
SP_DEFINE_SPLIT_TILED(sp_gemm_q4_split_tiled8, 8, 8)
SP_DEFINE_SPLIT_TILED(sp_gemm_q4_split_tiled16, 8, 16)
SP_DEFINE_SPLIT_TILED(sp_gemm_q4_split_tiled_m16p4, 16, 4)
SP_DEFINE_SPLIT_TILED(sp_gemm_q4_split_tiled_m16p8, 16, 8)
SP_DEFINE_SPLIT_TILED(sp_gemm_q4_split_tiled_m32p2, 32, 2)
SP_DEFINE_SPLIT_TILED(sp_gemm_q4_split_tiled_m32p4, 32, 4)
SP_DEFINE_SPLIT_TILED(sp_gemm_q4_split_tiled_m32p8, 32, 8)

// The same with wider tiles: fewer, larger accelerator runs per byte of weights.
#define SP_DEFINE_SPLIT_TILED_N(NAME, M, P, N)                                                    \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                      \
                 uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                                 \
    threadgroup float partials[P * M * N];                                                        \
    sp_na_split_tiled<M, P, N>(packed, scales, biases, a, out, residual, sums, p, group, sg, tid, partials); \
}
#define SP_DEFINE_SPLIT_TILED_MODE(NAME, M, P, MODE)                                              \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                      \
                 uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                                 \
    threadgroup float partials[P * M * 32];                                                       \
    sp_na_split_tiled<M, P, 32, MODE>(packed, scales, biases, a, out, residual, sums, p, group, sg, tid, partials); \
}
SP_DEFINE_SPLIT_TILED_MODE(sp_gemm_q4_split_tiled8_noepi, 8, 8, 1)
SP_DEFINE_SPLIT_TILED_MODE(sp_gemm_q4_split_tiled_m16p8_partition, 16, 8, 5)
SP_DEFINE_SPLIT_TILED_MODE(sp_gemm_q4_split_tiled_m32p4_partition, 32, 4, 5)
SP_DEFINE_SPLIT_TILED_MODE(sp_gemm_q4_split_tiled_m16p8_noepi, 16, 8, 1)
SP_DEFINE_SPLIT_TILED_MODE(sp_gemm_q4_split_tiled_m16p8_alias, 16, 8, 6)
SP_DEFINE_SPLIT_TILED_MODE(sp_gemm_q4_split_tiled8_alias, 8, 8, 6)
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_m16n64p4, 16, 4, 64)

kernel void sp_gemm_q4_split_tiled_m16n64p4_emit(device uchar *packed [[buffer(0)]],
                 device const ushort *scales [[buffer(1)]],
                 device const ushort *biases [[buffer(2)]],
                 device bfloat *a [[buffer(3)]],
                 device float *out [[buffer(4)]],
                 device const float *residual [[buffer(5)]],
                 device const float *sums [[buffer(6)]],
                 constant SpNaParams &p [[buffer(7)]],
                 device const ushort *emitWeight [[buffer(8)]],
                 device bfloat *emitValues [[buffer(9)]],
                 device float *emitSums [[buffer(10)]],
                 device float *emitSquares [[buffer(11)]],
                 uint3 group [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[4 * 16 * 64];
    sp_na_split_tiled<16, 4, 64, 0, true>(packed, scales, biases, a, out, residual, sums, p, group, sg, tid, partials,
                                          SpNaEmit{emitWeight, emitValues, emitSums, emitSquares});
}
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_m16n64p8, 16, 8, 64)
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_m16n128p2, 16, 2, 128)
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_m16n128p4, 16, 4, 128)
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_m16n32p16, 16, 16, 32)
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_n64p8, 8, 8, 64)
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_n64p4, 8, 4, 64)
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_n64p16, 8, 16, 64)
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_n128p4, 8, 4, 128)
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_n128p8, 8, 8, 128)
SP_DEFINE_SPLIT_TILED_N(sp_gemm_q4_split_tiled_n128p2, 8, 2, 128)

// A fused gate/up/SiLU split-K kernel (both projections in each simdgroup) was tried twice, on
// 8-row and on 16-row tiles. It is correct and saves a stage, but two interleaved weight
// streams per simdgroup made an eight-row step about 30 ms slower, so it is not kept.

// ---------------------------------------------------------------------------------------------
// 8-bit weights (the MLX 8-bit pack).
//
// The same affine scheme with a byte per code, 0 ... 255: the accelerator multiplies bf16
// activations by the bytes as it does by the nibbles, and the epilogue is unchanged. The tiled
// layout is the q4 one with 64 bytes per (row, quant group) instead of 32; scales and biases
// are laid out identically. Only the kernels the engine reaches with nothing set in the
// environment exist in this form: the split-K tiles for steps of up to 64 rows, the wide tile
// with its whole-tile and operand-emitting forms. The lane kernels are in engine_q8.metal.

#define SP_Q8_ARGUMENTS(SIDECAR)                                                                  \
    device uchar *packed [[buffer(0)]],                                                           \
    device SIDECAR *scales [[buffer(1)]],                                                         \
    device SIDECAR *biases [[buffer(2)]],                                                         \
    device bfloat *a [[buffer(3)]],                                                               \
    device float *out [[buffer(4)]],                                                              \
    device const float *residual [[buffer(5)]],                                                   \
    device const float *sums [[buffer(6)]],                                                       \
    constant SpNaParams &p [[buffer(7)]]
#define SP_Q8_EMIT_ARGUMENTS                                                                      \
    device const ushort *emitWeight [[buffer(8)]],                                                \
    device bfloat *emitValues [[buffer(9)]],                                                      \
    device float *emitSums [[buffer(10)]],                                                        \
    device float *emitSquares [[buffer(11)]]

// Threadgroup grids and thread counts as for the q4 kernels of the same names.
kernel void sp_gemm_q8_nat_m32n128s4(SP_Q8_ARGUMENTS(const ushort),
                                     uint3 group [[threadgroup_position_in_grid]])
{
    sp_na_tiled<32, 4, 0, 128, uchar, 64>(packed, scales, biases, a, out, residual, sums, p, group);
}
kernel void sp_gemm_q8_nat_m32n128s4_emit(SP_Q8_ARGUMENTS(const ushort), SP_Q8_EMIT_ARGUMENTS,
                                          uint3 group [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float emitStage[32 * 2 * 4 * 2];
    sp_na_tiled<32, 4, 7, 128, uchar, 64>(packed, scales, biases, a, out, residual, sums, p, group, nullptr, tid,
                                          SpNaEmit{emitWeight, emitValues, emitSums, emitSquares}, emitStage);
}
kernel void sp_gemm_q8_nat_m32n128s4_whole(SP_Q8_ARGUMENTS(bfloat),
                                           uint3 group [[threadgroup_position_in_grid]])
{
    sp_na_wide<32, 128, 4, false, uchar, 64>(packed, scales, biases, a, out, residual, sums, p, group);
}
kernel void sp_gemm_q8_nat_m32n128s4_whole_residual(SP_Q8_ARGUMENTS(bfloat),
                                                    uint3 group [[threadgroup_position_in_grid]])
{
    sp_na_wide<32, 128, 4, true, uchar, 64>(packed, scales, biases, a, out, residual, sums, p, group);
}
kernel void sp_gemm_q8_split_tiled_m16n64p4(SP_Q8_ARGUMENTS(const ushort),
                                            uint3 group [[threadgroup_position_in_grid]],
                                            uint sg [[simdgroup_index_in_threadgroup]],
                                            uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[4 * 16 * 64];
    sp_na_split_tiled<16, 4, 64, 0, false, uchar, 64>(packed, scales, biases, a, out, residual, sums, p, group, sg, tid, partials);
}
kernel void sp_gemm_q8_split_tiled_m16n64p4_emit(SP_Q8_ARGUMENTS(const ushort), SP_Q8_EMIT_ARGUMENTS,
                                                 uint3 group [[threadgroup_position_in_grid]],
                                                 uint sg [[simdgroup_index_in_threadgroup]],
                                                 uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[4 * 16 * 64];
    sp_na_split_tiled<16, 4, 64, 0, true, uchar, 64>(packed, scales, biases, a, out, residual, sums, p, group, sg, tid, partials,
                                                     SpNaEmit{emitWeight, emitValues, emitSums, emitSquares});
}
kernel void sp_gemm_q8_split_tiled_m32p4(SP_Q8_ARGUMENTS(const ushort),
                                         uint3 group [[threadgroup_position_in_grid]],
                                         uint sg [[simdgroup_index_in_threadgroup]],
                                         uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float partials[4 * 32 * 32];
    sp_na_split_tiled<32, 4, 32, 0, false, uchar, 64>(packed, scales, biases, a, out, residual, sums, p, group, sg, tid, partials);
}

// =============================================================================================
// Attention on the neural accelerators (4-bit KV pool, "na" layout).
//
// Pool layout, per full-attention layer, for K and for V:
//   codes: [page][kvHead][token][128 bytes]  — 256 values, natural nibble order, four groups of 64
//   meta:  [page][kvHead][token][4 groups][scale, bias] fp16
// so the codes of 64 consecutive tokens form a format tensor with a 128-byte row stride.
//
// One 256-thread group handles (KV head, span of pages, block of up to four rows). The block's
// M = 24 queries (4 rows x the 6 heads sharing the KV head) are scored 64 tokens at a time:
//
//   scores  = Q[M x 64 dims] x Kcodes[64 dims x 64 tokens]      per quant group, then affine
//   values += (P .* vscale)[M x 64 tokens] x Vcodes[64 tokens x 64 dims]   per quant group
//
// with an online softmax across chunks. Each group leaves the same per-span partial as
// sp_attn_scan_q4, and sp_attn_merge combines them.

#define SPA_M 24
#define SPA_C 64

// Query preparation for the accelerator scan: RMSNorm, RoPE, round to fp16, per-64 sums.
// Output layout: [kvHead][row][6 heads][256] so a block's queries are contiguous.
// Threadgroup grid: (heads, rows), 32 threads per group.
struct SpaQPrepParams { uint rows; uint heads; uint kvHeads; uint rotary; uint rowCap; float eps; float theta; };

inline float spa_rope(float x, float paired, uint i, uint rotary, uint position, float theta) {
    const uint half_ = rotary >> 1;
    const uint slot = i < half_ ? i : i - half_;
    const float angle = float(position) * pow(theta, -(2.0f * float(slot)) / float(rotary));
    return x * cos(angle) + (i < half_ ? -paired : paired) * sin(angle);
}

kernel void sp_attn_q_prepare_na(device const float *qproj [[buffer(0)]],
                                 device const ushort *normWeight [[buffer(1)]],
                                 device const uint *rowPos [[buffer(2)]],
                                 device half *qB [[buffer(3)]],
                                 device float *qSums [[buffer(4)]],
                                 constant SpaQPrepParams &p [[buffer(5)]],
                                 device const uint *rowQ [[buffer(6)]],
                                 device const float *rowInv [[buffer(7)]],
                                 uint3 group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_threadgroup]])
{
    const uint head = group.x, row = group.y;
    if (head >= p.heads || row >= p.rows) return;
    const uint base = (row * p.heads + head) * 512u;
    // The projection carries the deferred scale of the norm that fed it (1 when it was applied);
    // folding it into this norm's own scale applies it to every element below.
    const float carried = rowInv[row];
    float partial = 0.0f;
    for (uint i = lane; i < 256u; i += 32u) {
        const float x = qproj[base + i] * carried;
        partial += x * x;
    }
    const float inv = rsqrt(simd_sum(partial) / 256.0f + p.eps) * carried;
    const uint position = rowPos[row];
    const uint half_ = p.rotary >> 1;
    const uint perKV = p.heads / p.kvHeads;
    // rowQ maps a step row to its slot in the query buffer (its block-padded position).
    const uint query = ((head / perKV) * p.rowCap + rowQ[row]) * perKV + head % perKV;
    for (uint g = 0; g < 4u; ++g) {
        float total = 0.0f;
        for (uint s = 0; s < 2u; ++s) {
            const uint i = g * 64u + s * 32u + lane;
            float x = qproj[base + i] * inv * sp_na_bf16(normWeight[i]);
            if (i < p.rotary) {
                const uint pi = i < half_ ? i + half_ : i - half_;
                x = spa_rope(x, qproj[base + pi] * inv * sp_na_bf16(normWeight[pi]), i, p.rotary, position, p.theta);
            }
            const half rounded = half(x);
            qB[query * 256u + i] = rounded;
            total += float(rounded);
        }
        const float sum = simd_sum(total);
        if (lane == 0u) qSums[query * 4u + g] = sum;
    }
}

// K (RMSNorm + RoPE) and V (verbatim), quantised into the "na" pool layout.
// Lane l owns bytes l, l+32, l+64, l+96 of the record, i.e. elements (2l, 2l+1) of each group.
// Threadgroup grid: (kvHeads, rows), 32 threads per group.
struct SpaKVParams { uint rows; uint kvHeads; uint rotary; uint maxPages; float eps; float theta; };

inline void spa_quantize(float e0, float e1, device uchar *codes, device half *meta, uint g, uint lane) {
    const float lo = simd_min(min(e0, e1));
    const float hi = simd_max(max(e0, e1));
    const half scaleH = half(hi > lo ? (hi - lo) / 15.0f : 1.0f);
    const half biasH = half(lo);
    const float scale = float(scaleH), bias = float(biasH);
    const uint c0 = uint(clamp(round((e0 - bias) / scale), 0.0f, 15.0f));
    const uint c1 = uint(clamp(round((e1 - bias) / scale), 0.0f, 15.0f));
    codes[g * 32u + lane] = uchar(c0 | (c1 << 4));
    if (lane == 0u) { meta[g * 2u] = scaleH; meta[g * 2u + 1u] = biasH; }
}

kernel void sp_attn_kv_store_na(device const float *kproj [[buffer(0)]],
                                device const float *vproj [[buffer(1)]],
                                device const ushort *normWeight [[buffer(2)]],
                                device const uint *rowSlot [[buffer(3)]],
                                device const uint *rowPos [[buffer(4)]],
                                device const uint *pageTable [[buffer(5)]],
                                device uchar *kCodes [[buffer(6)]],
                                device half *kMeta [[buffer(7)]],
                                device uchar *vCodes [[buffer(8)]],
                                device half *vMeta [[buffer(9)]],
                                constant SpaKVParams &p [[buffer(10)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]])
{
    const uint head = group.x, row = group.y;
    if (head >= p.kvHeads || row >= p.rows) return;
    const uint base = (row * p.kvHeads + head) * 256u;
    float partial = 0.0f;
    for (uint i = lane; i < 256u; i += 32u) {
        const float x = kproj[base + i];
        partial += x * x;
    }
    const float inv = rsqrt(simd_sum(partial) / 256.0f + p.eps);
    const uint position = rowPos[row];
    const uint page = pageTable[rowSlot[row] * p.maxPages + position / 256u];
    const uint token = (page * p.kvHeads + head) * 256u + position % 256u;
    const uint half_ = p.rotary >> 1;
    for (uint g = 0; g < 4u; ++g) {
        float ke[2], ve[2];
        for (uint s = 0; s < 2u; ++s) {
            const uint i = g * 64u + 2u * lane + s;
            float x = kproj[base + i] * inv * sp_na_bf16(normWeight[i]);
            if (i < p.rotary) {
                const uint pi = i < half_ ? i + half_ : i - half_;
                x = spa_rope(x, kproj[base + pi] * inv * sp_na_bf16(normWeight[pi]), i, p.rotary, position, p.theta);
            }
            ke[s] = x;
            ve[s] = vproj[base + i];
        }
        spa_quantize(ke[0], ke[1], kCodes + token * 128u, kMeta + token * 8u, g, lane);
        spa_quantize(ve[0], ve[1], vCodes + token * 128u, vMeta + token * 8u, g, lane);
    }
}

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

// Shape-parametrised scan: M queries per block (6 per row), C tokens per chunk. The weighted
// probabilities for the value matmul are staged one quant group at a time so threadgroup
// memory holds M x C scores (fp32) plus one M x C fp16 operand.
// Threadgroup grid: (kvHeads * spans, blocks), 256 threads per group.
template <ushort M, ushort C>
inline void spa_scan(device half *qB, device const float *qSums, device const uint *blocks,
                     device const uint *rowSlot, device const uint *rowPos, device const uint *pageTable,
                     device uchar *kCodes, device const half *kMeta, device uchar *vCodes, device const half *vMeta,
                     device float *partials, constant SpaScanParams &p, uint3 group, uint tid,
                     threadgroup float *scores, threadgroup half *weighted,
                     threadgroup float *runMax, threadgroup float *runSum, threadgroup float *rescale,
                     threadgroup float *valueBias)
{
    const uint kvHead = group.x % p.kvHeads;
    const uint span = group.x / p.kvHeads;
    if (span >= p.spans || group.y >= p.blocks) return;
    const uint r0 = blocks[group.y * 2u], count = blocks[group.y * 2u + 1u];
    const uint perKV = p.heads / p.kvHeads;
    const uint maxVisible = rowPos[r0 + count - 1u] + 1u;
    const uint pages = (maxVisible + 255u) / 256u;
    const uint firstPage = span * p.pagesPerSpan;
    const uint lastPage = min(pages, firstPage + p.pagesPerSpan);
    const uint tableBase = rowSlot[r0] * p.maxPages;
    const uint queryBase = (kvHead * p.rowCap + r0) * perKV;

    constexpr auto scoreDescriptor = matmul2d_descriptor(M, C, 64, false, true, false);
    matmul2d<scoreDescriptor, execution_simdgroups<8>> scoreOp;
    constexpr auto valueDescriptor = matmul2d_descriptor(M, 64, C, false, false, false);
    matmul2d<valueDescriptor, execution_simdgroups<8>> valueOp;

    auto queries = tensor(qB + ulong(queryBase) * 256u, dextents<int, 2>{256, M}, array<int, 2>{1, 256});
    auto q0 = queries.template slice<64, M>(0, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> k0(kCodes, dextents<int, 2>{64, C}, array<int, 2>{1, 256});
    auto k0s = k0.template slice<64, C>(0, 0);
    auto score = scoreOp.template get_destination_cooperative_tensor<decltype(q0), decltype(k0s), float>();

    auto ws = tensor(weighted, dextents<int, 2>{C, M}, array<int, 2>{1, C});
    auto wss = ws.template slice<C, M>(0, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> v0(vCodes, dextents<int, 2>{64, C}, array<int, 2>{1, 256});
    auto v0s = v0.template slice<64, C>(0, 0);
    auto acc0 = valueOp.template get_destination_cooperative_tensor<decltype(wss), decltype(v0s), float>();
    auto acc1 = valueOp.template get_destination_cooperative_tensor<decltype(wss), decltype(v0s), float>();
    auto acc2 = valueOp.template get_destination_cooperative_tensor<decltype(wss), decltype(v0s), float>();
    auto acc3 = valueOp.template get_destination_cooperative_tensor<decltype(wss), decltype(v0s), float>();
#pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) { acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f; }

    if (tid < M) { runMax[tid] = -INFINITY; runSum[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint chunksPerPage = 256u / C;
    for (uint pg = firstPage; pg < lastPage; ++pg) {
        const uint pageTokens = (pageTable[tableBase + pg] * p.kvHeads + kvHead) * 256u;
        for (uint chunk = 0; chunk < chunksPerPage; ++chunk) {
            const uint t0 = pg * 256u + chunk * C;
            if (t0 >= maxVisible) break;
            const uint tokenBase = pageTokens + chunk * C;

#pragma unroll
            for (ushort i = 0; i < score.get_capacity(); ++i) score[i] = 0.0f;
            for (uint g = 0; g < 4u; ++g) {
                auto qs = queries.template slice<64, M>(int(g * 64u), 0);
                tensor<device uint4b_format, dextents<int, 2>, tensor_inline> kt(
                    kCodes + ulong(tokenBase) * 128u + g * 32u, dextents<int, 2>{64, C}, array<int, 2>{1, 256});
                auto ks = kt.template slice<64, C>(0, 0);
                auto part = scoreOp.template get_destination_cooperative_tensor<decltype(qs), decltype(ks), float>();
                scoreOp.run(qs, ks, part);
#pragma unroll
                for (ushort i = 0; i < score.get_capacity(); ++i) {
                    if (!score.is_valid_element(i)) continue;
                    const auto index = score.get_multidimensional_index(i);
                    device const half *meta = kMeta + (tokenBase + uint(index[0])) * 8u + g * 2u;
                    score[i] += part[i] * float(meta[0]) + qSums[(queryBase + uint(index[1])) * 4u + g] * float(meta[1]);
                }
            }
#pragma unroll
            for (ushort i = 0; i < score.get_capacity(); ++i) {
                if (!score.is_valid_element(i)) continue;
                const auto index = score.get_multidimensional_index(i);
                const uint m = uint(index[1]);
                const uint rowInBlock = m / perKV;
                const bool visible = rowInBlock < count && t0 + uint(index[0]) <= rowPos[r0 + min(rowInBlock, count - 1u)];
                score[i] = visible ? score[i] * p.scale : -INFINITY;
                scores[m * C + uint(index[0])] = score[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid < M) {
                float chunkMax = -INFINITY;
                for (uint t = 0; t < C; ++t) chunkMax = max(chunkMax, scores[tid * C + t]);
                const float newMax = max(runMax[tid], chunkMax);
                rescale[tid] = newMax == -INFINITY ? 0.0f : exp(runMax[tid] - newMax);
                runMax[tid] = newMax;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
            for (ushort i = 0; i < acc0.get_capacity(); ++i) {
                if (!acc0.is_valid_element(i)) continue;
                const float factor = rescale[uint(acc0.get_multidimensional_index(i)[1])];
                acc0[i] *= factor; acc1[i] *= factor; acc2[i] *= factor; acc3[i] *= factor;
            }
            // Probabilities, kept in registers and published for the per-query sums.
#pragma unroll
            for (ushort i = 0; i < score.get_capacity(); ++i) {
                if (!score.is_valid_element(i)) continue;
                const auto index = score.get_multidimensional_index(i);
                const uint m = uint(index[1]);
                score[i] = score[i] == -INFINITY ? 0.0f : exp(score[i] - runMax[m]);
                scores[m * C + uint(index[0])] = score[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid < M) {
                float sum = 0.0f;
                float4 bias = float4(0.0f);
                for (uint t = 0; t < C; ++t) {
                    const float prob = scores[tid * C + t];
                    if (prob > 0.0f) {
                        device const half *meta = vMeta + (tokenBase + t) * 8u;
                        sum += prob;
                        bias += prob * float4(float(meta[1]), float(meta[3]), float(meta[5]), float(meta[7]));
                    }
                }
                runSum[tid] = runSum[tid] * rescale[tid] + sum;
                valueBias[tid * 4u] = bias.x; valueBias[tid * 4u + 1u] = bias.y;
                valueBias[tid * 4u + 2u] = bias.z; valueBias[tid * 4u + 3u] = bias.w;
            }

            for (uint g = 0; g < 4u; ++g) {
#pragma unroll
                for (ushort i = 0; i < score.get_capacity(); ++i) {
                    if (!score.is_valid_element(i)) continue;
                    const auto index = score.get_multidimensional_index(i);
                    const uint t = uint(index[0]);
                    weighted[uint(index[1]) * C + t] =
                        score[i] > 0.0f ? half(score[i] * float(vMeta[(tokenBase + t) * 8u + g * 2u])) : half(0.0f);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                tensor<device uint4b_format, dextents<int, 2>, tensor_inline> vt(
                    vCodes + ulong(tokenBase) * 128u + g * 32u, dextents<int, 2>{64, C}, array<int, 2>{1, 256});
                auto vs = vt.template slice<64, C>(0, 0);
                auto part = valueOp.template get_destination_cooperative_tensor<decltype(wss), decltype(vs), float>();
                valueOp.run(wss, vs, part);
#pragma unroll
                for (ushort i = 0; i < part.get_capacity(); ++i) {
                    if (g == 0u) acc0[i] += part[i];
                    else if (g == 1u) acc1[i] += part[i];
                    else if (g == 2u) acc2[i] += part[i];
                    else acc3[i] += part[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
#pragma unroll
            for (ushort i = 0; i < acc0.get_capacity(); ++i) {
                if (!acc0.is_valid_element(i)) continue;
                const uint m = uint(acc0.get_multidimensional_index(i)[1]);
                acc0[i] += valueBias[m * 4u]; acc1[i] += valueBias[m * 4u + 1u];
                acc2[i] += valueBias[m * 4u + 2u]; acc3[i] += valueBias[m * 4u + 3u];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

#pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) {
        if (!acc0.is_valid_element(i)) continue;
        const auto index = acc0.get_multidimensional_index(i);
        const uint m = uint(index[1]), d = uint(index[0]);
        if (m / perKV >= count) continue;
        const uint row = r0 + m / perKV, head = kvHead * perKV + m % perKV;
        device float *partial = partials + ((row * p.heads + head) * p.spans + span) * 258u + 2u;
        partial[d] = acc0[i]; partial[64u + d] = acc1[i]; partial[128u + d] = acc2[i]; partial[192u + d] = acc3[i];
    }
    if (tid < M && tid / perKV < count) {
        const uint row = r0 + tid / perKV, head = kvHead * perKV + tid % perKV;
        device float *partial = partials + ((row * p.heads + head) * p.spans + span) * 258u;
        partial[0] = runMax[tid];
        partial[1] = runSum[tid];
    }
}

#define SP_DEFINE_ATTN_SCAN(NAME, M, C)                                                           \
kernel void NAME(device half *qB [[buffer(0)]],                                                   \
                 device const float *qSums [[buffer(1)]],                                         \
                 device const uint *blocks [[buffer(2)]],                                         \
                 device const uint *rowSlot [[buffer(3)]],                                        \
                 device const uint *rowPos [[buffer(4)]],                                         \
                 device const uint *pageTable [[buffer(5)]],                                      \
                 device uchar *kCodes [[buffer(6)]],                                              \
                 device const half *kMeta [[buffer(7)]],                                          \
                 device uchar *vCodes [[buffer(8)]],                                              \
                 device const half *vMeta [[buffer(9)]],                                          \
                 device float *partials [[buffer(10)]],                                           \
                 constant SpaScanParams &p [[buffer(11)]],                                        \
                 uint3 group [[threadgroup_position_in_grid]],                                    \
                 uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                                 \
    threadgroup float scores[M * C];                                                              \
    threadgroup half weighted[M * C];                                                             \
    threadgroup float runMax[M];                                                                  \
    threadgroup float runSum[M];                                                                  \
    threadgroup float rescale[M];                                                                 \
    threadgroup float valueBias[M * 4];                                                           \
    spa_scan<M, C>(qB, qSums, blocks, rowSlot, rowPos, pageTable, kCodes, kMeta, vCodes, vMeta,   \
                   partials, p, group, tid, scores, weighted, runMax, runSum, rescale, valueBias);\
}
SP_DEFINE_ATTN_SCAN(sp_attn_scan_na_m24c64, 24, 64)
SP_DEFINE_ATTN_SCAN(sp_attn_scan_na_m24c128, 24, 128)
SP_DEFINE_ATTN_SCAN(sp_attn_scan_na_m48c64, 48, 64)
SP_DEFINE_ATTN_SCAN(sp_attn_scan_na_m48c128, 48, 128)
SP_DEFINE_ATTN_SCAN(sp_attn_scan_na_m24c256, 24, 256)

// Original fixed-shape scan (M = 24, C = 64, all four value groups staged at once).
// Threadgroup grid: (kvHeads * spans, blocks), 256 threads per group.
kernel void sp_attn_scan_na(device half *qB [[buffer(0)]],
                            device const float *qSums [[buffer(1)]],
                            device const uint *blocks [[buffer(2)]],
                            device const uint *rowSlot [[buffer(3)]],
                            device const uint *rowPos [[buffer(4)]],
                            device const uint *pageTable [[buffer(5)]],
                            device uchar *kCodes [[buffer(6)]],
                            device const half *kMeta [[buffer(7)]],
                            device uchar *vCodes [[buffer(8)]],
                            device const half *vMeta [[buffer(9)]],
                            device float *partials [[buffer(10)]],
                            constant SpaScanParams &p [[buffer(11)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float scores[SPA_M * SPA_C];
    threadgroup half weighted[4 * SPA_M * SPA_C];
    threadgroup float runMax[SPA_M];
    threadgroup float runSum[SPA_M];
    threadgroup float rescale[SPA_M];
    threadgroup float valueBias[SPA_M * 4];

    const uint kvHead = group.x % p.kvHeads;
    const uint span = group.x / p.kvHeads;
    if (span >= p.spans || group.y >= p.blocks) return;
    const uint r0 = blocks[group.y * 2u], count = blocks[group.y * 2u + 1u];
    const uint perKV = p.heads / p.kvHeads;
    const uint maxVisible = rowPos[r0 + count - 1u] + 1u;
    const uint pages = (maxVisible + 255u) / 256u;
    const uint firstPage = span * p.pagesPerSpan;
    const uint lastPage = min(pages, firstPage + p.pagesPerSpan);
    const uint tableBase = rowSlot[r0] * p.maxPages;
    const uint queryBase = (kvHead * p.rowCap + r0) * perKV;

    constexpr auto scoreDescriptor = matmul2d_descriptor(SPA_M, SPA_C, 64, false, true, false);
    matmul2d<scoreDescriptor, execution_simdgroups<8>> scoreOp;
    constexpr auto valueDescriptor = matmul2d_descriptor(SPA_M, 64, SPA_C, false, false, false);
    matmul2d<valueDescriptor, execution_simdgroups<8>> valueOp;

    auto queries = tensor(qB + ulong(queryBase) * 256u, dextents<int, 2>{256, SPA_M}, array<int, 2>{1, 256});
    auto q0 = queries.template slice<64, SPA_M>(0, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> k0(kCodes, dextents<int, 2>{64, SPA_C}, array<int, 2>{1, 256});
    auto k0s = k0.template slice<64, SPA_C>(0, 0);
    auto score = scoreOp.template get_destination_cooperative_tensor<decltype(q0), decltype(k0s), float>();

    auto w0 = tensor(weighted, dextents<int, 2>{SPA_C, SPA_M}, array<int, 2>{1, SPA_C});
    auto w0s = w0.template slice<SPA_C, SPA_M>(0, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> v0(vCodes, dextents<int, 2>{64, SPA_C}, array<int, 2>{1, 256});
    auto v0s = v0.template slice<64, SPA_C>(0, 0);
    auto acc0 = valueOp.template get_destination_cooperative_tensor<decltype(w0s), decltype(v0s), float>();
    auto acc1 = valueOp.template get_destination_cooperative_tensor<decltype(w0s), decltype(v0s), float>();
    auto acc2 = valueOp.template get_destination_cooperative_tensor<decltype(w0s), decltype(v0s), float>();
    auto acc3 = valueOp.template get_destination_cooperative_tensor<decltype(w0s), decltype(v0s), float>();
#pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) { acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f; }

    if (tid < SPA_M) { runMax[tid] = -INFINITY; runSum[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pg = firstPage; pg < lastPage; ++pg) {
        const uint pageTokens = (pageTable[tableBase + pg] * p.kvHeads + kvHead) * 256u;
        for (uint chunk = 0; chunk < 4u; ++chunk) {
            const uint t0 = pg * 256u + chunk * SPA_C;
            if (t0 >= maxVisible) break;
            const uint tokenBase = pageTokens + chunk * SPA_C;

            // Scores: one accelerator matmul per quant group, then the affine epilogue.
#pragma unroll
            for (ushort i = 0; i < score.get_capacity(); ++i) score[i] = 0.0f;
            for (uint g = 0; g < 4u; ++g) {
                auto qs = queries.template slice<64, SPA_M>(int(g * 64u), 0);
                tensor<device uint4b_format, dextents<int, 2>, tensor_inline> kt(
                    kCodes + ulong(tokenBase) * 128u + g * 32u, dextents<int, 2>{64, SPA_C}, array<int, 2>{1, 256});
                auto ks = kt.template slice<64, SPA_C>(0, 0);
                auto part = scoreOp.template get_destination_cooperative_tensor<decltype(qs), decltype(ks), float>();
                scoreOp.run(qs, ks, part);
#pragma unroll
                for (ushort i = 0; i < score.get_capacity(); ++i) {
                    if (!score.is_valid_element(i)) continue;
                    const auto index = score.get_multidimensional_index(i);
                    device const half *meta = kMeta + (tokenBase + uint(index[0])) * 8u + g * 2u;
                    score[i] += part[i] * float(meta[0]) + qSums[(queryBase + uint(index[1])) * 4u + g] * float(meta[1]);
                }
            }
            // Mask (causal, and rows past the end of the block) and publish.
#pragma unroll
            for (ushort i = 0; i < score.get_capacity(); ++i) {
                if (!score.is_valid_element(i)) continue;
                const auto index = score.get_multidimensional_index(i);
                const uint m = uint(index[1]);
                const uint rowInBlock = m / perKV;
                const bool visible = rowInBlock < count && t0 + uint(index[0]) <= rowPos[r0 + min(rowInBlock, count - 1u)];
                score[i] = visible ? score[i] * p.scale : -INFINITY;
                scores[m * SPA_C + uint(index[0])] = score[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid < SPA_M) {
                float chunkMax = -INFINITY;
                for (uint t = 0; t < SPA_C; ++t) chunkMax = max(chunkMax, scores[tid * SPA_C + t]);
                const float newMax = max(runMax[tid], chunkMax);
                // exp(-inf - -inf) is NaN; a query with nothing visible yet keeps a zero rescale.
                rescale[tid] = newMax == -INFINITY ? 0.0f : exp(runMax[tid] - newMax);
                runMax[tid] = newMax;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
            for (ushort i = 0; i < acc0.get_capacity(); ++i) {
                if (!acc0.is_valid_element(i)) continue;
                const float factor = rescale[uint(acc0.get_multidimensional_index(i)[1])];
                acc0[i] *= factor; acc1[i] *= factor; acc2[i] *= factor; acc3[i] *= factor;
            }
#pragma unroll
            for (ushort i = 0; i < score.get_capacity(); ++i) {
                if (!score.is_valid_element(i)) continue;
                const auto index = score.get_multidimensional_index(i);
                const uint m = uint(index[1]), t = uint(index[0]);
                const float prob = score[i] == -INFINITY ? 0.0f : exp(score[i] - runMax[m]);
                scores[m * SPA_C + t] = prob;
                device const half *meta = vMeta + (tokenBase + t) * 8u;
                for (uint g = 0; g < 4u; ++g) {
                    weighted[(g * SPA_M + m) * SPA_C + t] = prob > 0.0f ? half(prob * float(meta[g * 2u])) : half(0.0f);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid < SPA_M) {
                float sum = 0.0f;
                float4 bias = float4(0.0f);
                for (uint t = 0; t < SPA_C; ++t) {
                    const float prob = scores[tid * SPA_C + t];
                    if (prob > 0.0f) {
                        device const half *meta = vMeta + (tokenBase + t) * 8u;
                        sum += prob;
                        bias += prob * float4(float(meta[1]), float(meta[3]), float(meta[5]), float(meta[7]));
                    }
                }
                runSum[tid] = runSum[tid] * rescale[tid] + sum;
                valueBias[tid * 4u] = bias.x; valueBias[tid * 4u + 1u] = bias.y;
                valueBias[tid * 4u + 2u] = bias.z; valueBias[tid * 4u + 3u] = bias.w;
            }

            // Values: one accelerator matmul per quant group over this chunk's 64 tokens.
            for (uint g = 0; g < 4u; ++g) {
                auto ws = tensor(weighted + g * SPA_M * SPA_C, dextents<int, 2>{SPA_C, SPA_M}, array<int, 2>{1, SPA_C});
                auto wss = ws.template slice<SPA_C, SPA_M>(0, 0);
                tensor<device uint4b_format, dextents<int, 2>, tensor_inline> vt(
                    vCodes + ulong(tokenBase) * 128u + g * 32u, dextents<int, 2>{64, SPA_C}, array<int, 2>{1, 256});
                auto vs = vt.template slice<64, SPA_C>(0, 0);
                auto part = valueOp.template get_destination_cooperative_tensor<decltype(wss), decltype(vs), float>();
                valueOp.run(wss, vs, part);
#pragma unroll
                for (ushort i = 0; i < part.get_capacity(); ++i) {
                    if (g == 0u) acc0[i] += part[i];
                    else if (g == 1u) acc1[i] += part[i];
                    else if (g == 2u) acc2[i] += part[i];
                    else acc3[i] += part[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
            for (ushort i = 0; i < acc0.get_capacity(); ++i) {
                if (!acc0.is_valid_element(i)) continue;
                const uint m = uint(acc0.get_multidimensional_index(i)[1]);
                acc0[i] += valueBias[m * 4u]; acc1[i] += valueBias[m * 4u + 1u];
                acc2[i] += valueBias[m * 4u + 2u]; acc3[i] += valueBias[m * 4u + 3u];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Leave this span's partial for every real query of the block.
#pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) {
        if (!acc0.is_valid_element(i)) continue;
        const auto index = acc0.get_multidimensional_index(i);
        const uint m = uint(index[1]), d = uint(index[0]);
        if (m / perKV >= count) continue;
        const uint row = r0 + m / perKV, head = kvHead * perKV + m % perKV;
        device float *partial = partials + ((row * p.heads + head) * p.spans + span) * 258u + 2u;
        partial[d] = acc0[i]; partial[64u + d] = acc1[i]; partial[128u + d] = acc2[i]; partial[192u + d] = acc3[i];
    }
    if (tid < SPA_M && tid / perKV < count) {
        const uint row = r0 + tid / perKV, head = kvHead * perKV + tid % perKV;
        device float *partial = partials + ((row * p.heads + head) * p.spans + span) * 258u;
        partial[0] = runMax[tid];
        partial[1] = runSum[tid];
    }
}

// =============================================================================================
// Dense three-pass attention on the accelerators.
//
// The scan above interleaves matmuls with softmax bookkeeping and pays a barrier per step. Here
// the three parts are separate, and the two matmul passes are plain GEMM loops:
//
//   1. sp_attn_scores_na   scores[query][token] = Q x Kcodes (per quant group, affine, masked)
//   2. sp_attn_softmax     per query: max, sum, and the value operands
//                          weighted[g][query][token] = exp(score - max) * vscale[token][g]
//   3. sp_attn_values_na   out[query][dims of g] = sum_t weighted[g] x Vcodes, + bias, / sum,
//                          then the sigmoid gate; emits the following GEMM's operand
//
// Rows are grouped into blocks of up to eight consecutive rows of one slot; a block's 48
// queries per KV head form the M side of both matmuls. "qrow" = block * 8 + row in block.
// A step may process its blocks in several sub-batches to bound the scratch buffers.

#define SPD_M 48
#define SPD_ROWS 8

struct SpdParams {
    uint blocks;        // blocks in this sub-batch
    uint blockBase;     // first block of the sub-batch
    uint heads;
    uint kvHeads;
    uint maxPages;
    uint qrowCap;       // qrows addressable in the query buffer
    uint tcap;          // token capacity of a scratch row (multiple of 64)
    uint rowBase;       // first step row of the sub-batch
    uint rowCount;
    float scale;
};

// Threadgroup grid: (pages, blocks, kvHeads), 128 threads per group. Each group walks the four
// 64-token chunks of one KV page.
kernel void sp_attn_scores_na(device half *qB [[buffer(0)]],
                              device const float *qSums [[buffer(1)]],
                              device const uint *blocks [[buffer(2)]],
                              device const uint *rowSlot [[buffer(3)]],
                              device const uint *rowPos [[buffer(4)]],
                              device const uint *pageTable [[buffer(5)]],
                              device uchar *kCodes [[buffer(6)]],
                              device const half *kMeta [[buffer(7)]],
                              device float *scores [[buffer(8)]],
                              constant SpdParams &p [[buffer(9)]],
                              uint3 group [[threadgroup_position_in_grid]])
{
    if (group.y >= p.blocks || group.z >= p.kvHeads) return;
    const uint block = p.blockBase + group.y, kvHead = group.z;
    const uint r0 = blocks[block * 2u], count = blocks[block * 2u + 1u];
    const uint maxVisible = rowPos[r0 + count - 1u] + 1u;
    const uint pageStart = group.x * 256u;
    if (pageStart >= maxVisible) return;
    const uint perKV = p.heads / p.kvHeads;
    const uint pageTokens = (pageTable[rowSlot[r0] * p.maxPages + group.x] * p.kvHeads + kvHead) * 256u;
    const uint queryBase = (kvHead * p.qrowCap + block * SPD_ROWS) * perKV;
    const uint localBase = (kvHead * p.blocks * SPD_ROWS + group.y * SPD_ROWS) * perKV;

    constexpr auto descriptor = matmul2d_descriptor(SPD_M, 64, 64, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto queries = tensor(qB + ulong(queryBase) * 256u, dextents<int, 2>{256, SPD_M}, array<int, 2>{1, 256});
    auto q0 = queries.template slice<64, SPD_M>(0, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> k0(
        kCodes + ulong(pageTokens) * 128u, dextents<int, 2>{64, 64}, array<int, 2>{1, 256});
    auto k0s = k0.template slice<64, 64>(0, 0);
    auto score = operation.template get_destination_cooperative_tensor<decltype(q0), decltype(k0s), float>();

    const uint chunks = min(4u, (maxVisible - pageStart + 63u) / 64u);
    for (uint chunk = 0; chunk < chunks; ++chunk) {
        const uint t0 = pageStart + chunk * 64u;
        const uint tokenBase = pageTokens + chunk * 64u;
#pragma unroll
        for (ushort i = 0; i < score.get_capacity(); ++i) score[i] = 0.0f;
        for (uint g = 0; g < 4u; ++g) {
            auto qs = queries.template slice<64, SPD_M>(int(g * 64u), 0);
            tensor<device uint4b_format, dextents<int, 2>, tensor_inline> kt(
                kCodes + ulong(tokenBase) * 128u + g * 32u, dextents<int, 2>{64, 64}, array<int, 2>{1, 256});
            auto ks = kt.template slice<64, 64>(0, 0);
            auto part = operation.template get_destination_cooperative_tensor<decltype(qs), decltype(ks), float>();
            operation.run(qs, ks, part);
#pragma unroll
            for (ushort i = 0; i < score.get_capacity(); ++i) {
                if (!score.is_valid_element(i)) continue;
                const auto index = score.get_multidimensional_index(i);
                device const half *meta = kMeta + (tokenBase + uint(index[0])) * 8u + g * 2u;
                score[i] += part[i] * float(meta[0]) + qSums[(queryBase + uint(index[1])) * 4u + g] * float(meta[1]);
            }
        }
#pragma unroll
        for (ushort i = 0; i < score.get_capacity(); ++i) {
            if (!score.is_valid_element(i)) continue;
            const auto index = score.get_multidimensional_index(i);
            const uint m = uint(index[1]), token = t0 + uint(index[0]);
            const uint rowInBlock = m / perKV;
            if (rowInBlock >= count) continue;
            scores[ulong(localBase + m) * p.tcap + token] = token <= rowPos[r0 + rowInBlock] ? score[i] * p.scale : -INFINITY;
        }
    }
}

// Per query: softmax statistics and the value-matmul operands.
// stats[query] = (1 / sum, bias0, bias1, bias2, bias3, 0, 0, 0)
// Threadgroup grid: (heads, rows of the sub-batch), 32 threads per group.
kernel void sp_attn_softmax(device const float *scores [[buffer(0)]],
                            device const uint *rowQ [[buffer(1)]],
                            device const uint *blocks [[buffer(2)]],
                            device const uint *rowSlot [[buffer(3)]],
                            device const uint *rowPos [[buffer(4)]],
                            device const uint *pageTable [[buffer(5)]],
                            device const half *vMeta [[buffer(6)]],
                            device half *weighted [[buffer(7)]],
                            device float *stats [[buffer(8)]],
                            constant SpdParams &p [[buffer(9)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_threadgroup]])
{
    if (group.x >= p.heads || group.y >= p.rowCount) return;
    const uint head = group.x, row = p.rowBase + group.y;
    const uint perKV = p.heads / p.kvHeads;
    const uint kvHead = head / perKV;
    const uint qrow = rowQ[row];
    const uint block = qrow / SPD_ROWS;
    const uint local = (kvHead * p.blocks * SPD_ROWS + (qrow - p.blockBase * SPD_ROWS)) * perKV + head % perKV;
    const uint localQueries = p.kvHeads * p.blocks * SPD_ROWS * perKV;
    const uint visible = rowPos[row] + 1u;
    const uint r0 = blocks[block * 2u], count = blocks[block * 2u + 1u];
    const uint blockVisible = rowPos[r0 + count - 1u] + 1u;
    device const float *s = scores + ulong(local) * p.tcap;
    const uint tableBase = rowSlot[row] * p.maxPages;

    float top = -INFINITY;
    for (uint t = lane; t < visible; t += 32u) top = max(top, s[t]);
    top = simd_max(top);

    float sum = 0.0f;
    float4 bias = float4(0.0f);
    device half *w0 = weighted + ulong(local) * p.tcap;
    device half *w1 = weighted + ulong(localQueries + local) * p.tcap;
    device half *w2 = weighted + ulong(2u * localQueries + local) * p.tcap;
    device half *w3 = weighted + ulong(3u * localQueries + local) * p.tcap;
    // Tokens up to the block's longest row are written, as zero past this row's own position,
    // because the value matmul walks the whole block range. One page-table load per page; the
    // four (scale, bias) pairs of a token arrive as two packed loads.
    const uint padded = (blockVisible + 63u) & ~63u;
    for (uint pageStart = 0; pageStart < padded; pageStart += 256u) {
        device const packed_half4 *meta =
            (device const packed_half4 *)(vMeta + (pageTable[tableBase + pageStart / 256u] * p.kvHeads + kvHead) * 256u * 8u);
        const uint pageEnd = min(padded, pageStart + 256u);
        for (uint t = pageStart + lane; t < pageEnd; t += 32u) {
            float4 scaled = float4(0.0f);
            if (t < visible) {
                const float prob = exp(s[t] - top);
                const float4 m0 = float4(meta[(t - pageStart) * 2u]), m1 = float4(meta[(t - pageStart) * 2u + 1u]);
                sum += prob;
                bias += prob * float4(m0.y, m0.w, m1.y, m1.w);
                scaled = prob * float4(m0.x, m0.z, m1.x, m1.z);
            }
            w0[t] = half(scaled.x); w1[t] = half(scaled.y); w2[t] = half(scaled.z); w3[t] = half(scaled.w);
        }
    }
    const float total = simd_sum(sum);
    const float4 totalBias = float4(simd_sum(bias.x), simd_sum(bias.y), simd_sum(bias.z), simd_sum(bias.w));
    if (lane == 0u) {
        device float *out = stats + local * 8u;
        out[0] = 1.0f / total;
        out[1] = totalBias.x; out[2] = totalBias.y; out[3] = totalBias.z; out[4] = totalBias.w;
    }
}

// Values, phase 1. The token range of a block is cut into `segments`; each threadgroup sums one
// segment and leaves an unnormalised partial, so the long contraction over tokens runs across
// many groups instead of one.
// partials layout: [(kvHead * blocks + block) * 4 + g][segment][48 queries][64 dims]
// Threadgroup grid: (4 quant groups * segments, blocks, kvHeads), 128 threads per group.
struct SpdValueParams {
    uint segments;
    uint chunksPerSegment;
};

kernel void sp_attn_values_na(device half *weighted [[buffer(0)]],
                              device const uint *blocks [[buffer(1)]],
                              device const uint *rowSlot [[buffer(2)]],
                              device const uint *rowPos [[buffer(3)]],
                              device const uint *pageTable [[buffer(4)]],
                              device uchar *vCodes [[buffer(5)]],
                              device float *partials [[buffer(6)]],
                              constant SpdParams &p [[buffer(7)]],
                              constant SpdValueParams &vp [[buffer(8)]],
                              uint3 group [[threadgroup_position_in_grid]])
{
    const uint g = group.x % 4u, segment = group.x / 4u;
    if (segment >= vp.segments || group.y >= p.blocks || group.z >= p.kvHeads) return;
    const uint block = p.blockBase + group.y, kvHead = group.z;
    const uint r0 = blocks[block * 2u], count = blocks[block * 2u + 1u];
    const uint maxVisible = rowPos[r0 + count - 1u] + 1u;
    const uint chunks = (maxVisible + 63u) / 64u;
    const uint firstChunk = segment * vp.chunksPerSegment;
    const uint lastChunk = min(chunks, firstChunk + vp.chunksPerSegment);
    const uint perKV = p.heads / p.kvHeads;
    const uint localBase = (kvHead * p.blocks * SPD_ROWS + group.y * SPD_ROWS) * perKV;
    const uint localQueries = p.kvHeads * p.blocks * SPD_ROWS * perKV;
    const uint tableBase = rowSlot[r0] * p.maxPages;

    constexpr auto descriptor = matmul2d_descriptor(SPD_M, 64, 64, false, false, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto operands = tensor(weighted + ulong(g * localQueries + localBase) * p.tcap,
                           dextents<int, 2>{int(p.tcap), SPD_M}, array<int, 2>{1, int(p.tcap)});
    auto a0 = operands.template slice<64, SPD_M>(0, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> v0(vCodes, dextents<int, 2>{64, 64}, array<int, 2>{1, 256});
    auto v0s = v0.template slice<64, 64>(0, 0);
    auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(v0s), float>();
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;

    for (uint chunk = firstChunk; chunk < lastChunk; ++chunk) {
        const uint t0 = chunk * 64u;
        const uint tokenBase = (pageTable[tableBase + t0 / 256u] * p.kvHeads + kvHead) * 256u + t0 % 256u;
        auto as = operands.template slice<64, SPD_M>(int(t0), 0);
        tensor<device uint4b_format, dextents<int, 2>, tensor_inline> vt(
            vCodes + ulong(tokenBase) * 128u + g * 32u, dextents<int, 2>{64, 64}, array<int, 2>{1, 256});
        auto vs = vt.template slice<64, 64>(0, 0);
        auto part = operation.template get_destination_cooperative_tensor<decltype(as), decltype(vs), float>();
        operation.run(as, vs, part);
#pragma unroll
        for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] += part[i];
    }
    device float *target = partials + (ulong((kvHead * p.blocks + group.y) * 4u + g) * vp.segments + segment) * (SPD_M * 64u);
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto index = acc.get_multidimensional_index(i);
        target[uint(index[1]) * 64u + uint(index[0])] = acc[i];
    }
}

// Values, phase 2: sum the segments, add the bias term, normalise, gate, and emit the operand
// for the output projection. Thread d of a group owns dimension d for every query.
// Threadgroup grid: (4 quant groups, blocks, kvHeads), 64 threads per group.
kernel void sp_attn_values_merge(device const float *partials [[buffer(0)]],
                                 device const float *stats [[buffer(1)]],
                                 device const uint *blocks [[buffer(2)]],
                                 device const float *qproj [[buffer(3)]],
                                 device float *out [[buffer(4)]],
                                 device bfloat *outB [[buffer(5)]],
                                 device float *outSums [[buffer(6)]],
                                 constant SpdParams &p [[buffer(7)]],
                                 constant SpdValueParams &vp [[buffer(8)]],
                                 uint3 group [[threadgroup_position_in_grid]],
                                 uint d [[thread_index_in_threadgroup]])
{
    threadgroup float staged[64];
    if (group.x >= 4u || group.y >= p.blocks || group.z >= p.kvHeads) return;
    const uint g = group.x, block = p.blockBase + group.y, kvHead = group.z;
    const uint r0 = blocks[block * 2u], count = blocks[block * 2u + 1u];
    const uint perKV = p.heads / p.kvHeads;
    const uint localBase = (kvHead * p.blocks * SPD_ROWS + group.y * SPD_ROWS) * perKV;
    device const float *source = partials + ulong((kvHead * p.blocks + group.y) * 4u + g) * vp.segments * (SPD_M * 64u);
    const uint channel = g * 64u + d;
    for (uint m = 0; m < count * perKV; ++m) {
        float acc = 0.0f;
        for (uint s = 0; s < vp.segments; ++s) acc += source[(s * SPD_M + m) * 64u + d];
        const uint row = r0 + m / perKV, head = kvHead * perKV + m % perKV;
        device const float *stat = stats + (localBase + m) * 8u;
        const float gate = qproj[(row * p.heads + head) * 512u + 256u + channel];
        const float result = (acc + stat[1u + g]) * stat[0] / (1.0f + exp(-gate));
        const uint position = (row * p.heads + head) * 256u + channel;
        out[position] = result;
        const bfloat rounded = bfloat(result);
        outB[position] = rounded;
        staged[d] = float(rounded);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d == 0u) {
            float total = 0.0f;
            for (uint i = 0; i < 64u; ++i) total += staged[i];
            outSums[row * (p.heads * 4u) + head * 4u + g] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// =============================================================================================
// Fused attention over an int8 KV pool ("i8" layout).
//
//   codes: [page][kvHead][token][256 int8]        scale: [page][kvHead][token] fp32
//
// A K or V vector is symmetric int8 with one scale, value = scale * code, so there is no bias
// term anywhere: a score is one 256-wide accelerator matmul and one multiply, and the value
// scale folds into the probability operand. Scores, probabilities and the running output never
// leave the threadgroup; K and V are read once.
//
// One barrier per chunk. The usual online softmax needs the running row maximum before the
// exponentials (two more barriers); here the reference is the row maximum of the span's first
// chunk and is only moved when a later score exceeds it by more than `SPI_BUMP`, which is
// handled on a rare path. Probabilities are written to alternating buffers, so the next chunk
// may start while other simdgroups are still reading the previous one.

// Threadgroup grid: (kvHeads, rows), 32 threads per group. Lane l owns elements 8l ..< 8l + 8.
kernel void sp_attn_kv_store_i8(device const float *kproj [[buffer(0)]],
                                device const float *vproj [[buffer(1)]],
                                device const ushort *normWeight [[buffer(2)]],
                                device const uint *rowSlot [[buffer(3)]],
                                device const uint *rowPos [[buffer(4)]],
                                device const uint *pageTable [[buffer(5)]],
                                device int8_t *kCodes [[buffer(6)]],
                                device float *kScale [[buffer(7)]],
                                device int8_t *vCodes [[buffer(8)]],
                                device float *vScale [[buffer(9)]],
                                constant SpaKVParams &p [[buffer(10)]],
                                device const float *rowInv [[buffer(11)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]])
{
    const uint head = group.x, row = group.y;
    if (head >= p.kvHeads || row >= p.rows) return;
    const uint base = (row * p.kvHeads + head) * 256u;
    // Both projections carry the deferred scale of the norm that fed them (1 when it was applied).
    const float carried = rowInv[row];
    float partial = 0.0f;
    for (uint i = lane; i < 256u; i += 32u) {
        const float x = kproj[base + i] * carried;
        partial += x * x;
    }
    const float inv = rsqrt(simd_sum(partial) / 256.0f + p.eps) * carried;
    const uint position = rowPos[row];
    const uint page = pageTable[rowSlot[row] * p.maxPages + position / 256u];
    const uint token = (page * p.kvHeads + head) * 256u + position % 256u;
    const uint half_ = p.rotary >> 1;
    float ke[8], ve[8];
    float kHi = 0.0f, vHi = 0.0f;
    for (uint e = 0; e < 8u; ++e) {
        const uint i = lane * 8u + e;
        float x = kproj[base + i] * inv * sp_na_bf16(normWeight[i]);
        if (i < p.rotary) {
            const uint pi = i < half_ ? i + half_ : i - half_;
            x = spa_rope(x, kproj[base + pi] * inv * sp_na_bf16(normWeight[pi]), i, p.rotary, position, p.theta);
        }
        ke[e] = x; ve[e] = vproj[base + i] * carried;
        kHi = max(kHi, abs(x)); vHi = max(vHi, abs(ve[e]));
    }
    kHi = simd_max(kHi); vHi = simd_max(vHi);
    const float kS = kHi > 0.0f ? kHi / 127.0f : 1.0f;
    const float vS = vHi > 0.0f ? vHi / 127.0f : 1.0f;
    for (uint e = 0; e < 8u; ++e) {
        kCodes[ulong(token) * 256u + lane * 8u + e] = int8_t(clamp(rint(ke[e] / kS), -127.0f, 127.0f));
        vCodes[ulong(token) * 256u + lane * 8u + e] = int8_t(clamp(rint(ve[e] / vS), -127.0f, 127.0f));
    }
    if (lane == 0u) { kScale[token] = kS; vScale[token] = vS; }
}

// Scheduling probe with the accelerator: every threadgroup (256 threads) does `iterations`
// score-shaped matmuls, 48 queries by 64 keys of 256, from fixed operands, into a cooperative
// tensor, and reads the result. How a dispatch of them scales says what short threadgroups
// that use the accelerator cost each other.
struct SpSpinParams { uint iterations; uint barriers; };
kernel void sp_probe_matmul(device half *qB [[buffer(0)]], device int8_t *kCodes [[buffer(1)]],
                            device float *out [[buffer(2)]], constant SpSpinParams &p [[buffer(3)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]])
{
    // `barriers` is a set of bits here: 1 a threadgroup barrier each round, 2 the value matmul
    // (accumulating, its probabilities read from threadgroup memory), 4 the probabilities
    // written by the threads each round, 8 the accumulated values written to device memory at
    // the end, 16 the two cooperative tensors zeroed by the threads first.
    threadgroup bfloat weighted[48 * 64];
    // More threadgroup memory, touched so that it is really there: bits 32 and 64 of `barriers`
    // add 6 KiB and 18 KiB (does the memory a threadgroup holds limit how many run at once?).
    threadgroup float extraA[1536];
    threadgroup float extraB[4608];
    if ((p.barriers & 32u) != 0u) extraA[tid * 6u] = float(tid);
    if ((p.barriers & 64u) != 0u) extraB[tid * 18u] = float(tid);
    constexpr auto descriptor = matmul2d_descriptor(48, 64, 256, false, true, false);
    matmul2d<descriptor, execution_simdgroups<8>> op;
    constexpr auto valueDescriptor = matmul2d_descriptor(48, 256, 64, false, false, false,
                                                         matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<valueDescriptor, execution_simdgroups<8>> valueOp;
    auto queries = tensor(qB, dextents<int, 2>{256, 48}, array<int, 2>{1, 256});
    auto q0 = queries.slice<256, 48>(0, 0);
    auto keys = tensor(kCodes, dextents<int, 2>{256, 64}, array<int, 2>{1, 256});
    auto k0 = keys.slice<256, 64>(0, 0);
    auto score = op.get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    auto wt = tensor(weighted, dextents<int, 2>{64, 48}, array<int, 2>{1, 64});
    auto w0 = wt.slice<64, 48>(0, 0);
    auto running = valueOp.get_destination_cooperative_tensor<decltype(w0), decltype(k0), float>();
    if ((p.barriers & 16u) != 0u) {
        for (ushort e = 0; e < running.get_capacity(); ++e) running[e] = 0.0f;
    }
    float total = 0.0f;
    for (uint i = 0; i < p.iterations; ++i) {
        op.run(q0, k0, score);
        for (ushort e = 0; e < score.get_capacity(); ++e) {
            if (!score.is_valid_element(e)) continue;
            total += score[e];
            if ((p.barriers & 4u) != 0u) {
                const auto index = score.get_multidimensional_index(e);
                weighted[uint(index[1]) * 64u + uint(index[0])] = bfloat(score[e] * 1e-6f);
            }
        }
        if ((p.barriers & 1u) != 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((p.barriers & 2u) != 0u) valueOp.run(w0, k0, running);
    }
    if ((p.barriers & 8u) != 0u) {
        for (ushort e = 0; e < running.get_capacity(); ++e) {
            if (!running.is_valid_element(e)) continue;
            const auto index = running.get_multidimensional_index(e);
            out[((group.y * 64u + group.x) % 64u) * 1024u + (uint(index[1]) * 256u + uint(index[0])) % 1024u] = running[e];
        }
    }
    if ((p.barriers & 32u) != 0u) total += extraA[(tid * 6u + 6u) % 1536u];
    if ((p.barriers & 64u) != 0u) total += extraB[(tid * 18u + 18u) % 4608u];
    if (tid == 0u) out[(group.y * 4096u + group.x) % 65536u] = total;
}

// A second accelerator probe with another shape and operand type (32 rows by 128 outputs of 64,
// bf16 by bf16, like a GEMM tile): alternated with sp_probe_matmul it shows what it costs the
// GPU to go from one kind of matmul to another.
kernel void sp_probe_matmul_b(device half *qB [[buffer(0)]], device int8_t *kCodes [[buffer(1)]],
                              device float *out [[buffer(2)]], constant SpSpinParams &p [[buffer(3)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]])
{
    constexpr auto descriptor = matmul2d_descriptor(32, 128, 64, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> op;
    auto left = tensor(qB, dextents<int, 2>{64, 32}, array<int, 2>{1, 64});
    auto l0 = left.slice<64, 32>(0, 0);
    auto right = tensor(qB + 8192, dextents<int, 2>{64, 128}, array<int, 2>{1, 64});
    auto r0 = right.slice<64, 128>(0, 0);
    auto result = op.get_destination_cooperative_tensor<decltype(l0), decltype(r0), float>();
    float total = 0.0f;
    for (uint i = 0; i < p.iterations; ++i) {
        op.run(l0, r0, result);
        for (ushort e = 0; e < result.get_capacity(); ++e) { if (result.is_valid_element(e)) total += result[e]; }
    }
    if (tid == 0u) out[(group.y * 4096u + group.x) % 65536u] = total;
}

#define SPI_BUMP 30.0f
#define SPI_CLAMP 60.0f

// M fused queries per block (6 per row), C tokens per chunk, SG simdgroups, PT the probability
// operand type. Threadgroup grid: (kvHeads * spans, blocks), SG * 32 threads per group.
// SG simdgroups run the score matmul and SGV the value matmul (0: the same). The accelerator
// is fastest at 32 x 32 outputs per simdgroup, and the two products have different shapes:
// scores are M x C, values M x 256. The threadgroup has max(SG, SGV) simdgroups; only the
// first SG hold score elements.
template <ushort M, ushort C, ushort SG, class PT, ushort DIAG = 0, ushort CAPACITY = 0, ushort SGV = 0>
inline void spi_scan(device half *qB, device const uint *blocks, device const uint *rowSlot,
                     device const uint *rowPos, device const uint *pageTable,
                     device int8_t *kCodes, device const float *kScale,
                     device int8_t *vCodes, device const float *vScale,
                     device float *partials, constant SpaScanParams &p, uint3 group, uint tid,
                     threadgroup float *staged, threadgroup float *reference, threadgroup float *bump,
                     threadgroup atomic_uint *flags)
{
    // Two grids. (kvHeads * spans, blocks), or with blocksAcross (kvHeads * blocks, spans): the
    // GPU spreads a row of the grid over its cores and takes the rows in turn, so the wide
    // dimension is the one that runs in parallel.
    const uint kvHead = group.x % p.kvHeads;
    const uint span = p.blocksAcross != 0u ? group.y : group.x / p.kvHeads;
    const uint block = p.blocksAcross != 0u ? group.x / p.kvHeads : group.y;
    if (span >= p.spans || block >= p.blocks) return;
    if (DIAG == 21) return;                     // timing: what launching the threadgroups costs
    if (DIAG == 26 && span != 0u) return;       // timing: one span's threadgroups working among the idle rest
    if (DIAG == 30 && span + 1u != p.spans) return;     // timing: the newest span alone
    if (DIAG == 27 && group.y % 4u != 0u) return;   // timing: a quarter of the blocks working
    const uint r0 = blocks[block * 2u], count = blocks[block * 2u + 1u];
    const uint perKV = p.heads / p.kvHeads;
    const uint firstPos = rowPos[r0];
    const uint maxVisible = firstPos + count;
    const uint pages = (maxVisible + 255u) / 256u;
    const uint firstPage = span * p.pagesPerSpan;
    const uint lastPage = min(pages, firstPage + p.pagesPerSpan);
    const uint tableBase = rowSlot[r0] * p.maxPages;
    // DIAG 34: every threadgroup reads the same query tile (timing: is it the tiles' first use that costs?).
    const uint queryBase = DIAG == 34 ? 0u : (kvHead * p.rowCap + r0) * perKV;

    constexpr auto scoreDescriptor = matmul2d_descriptor(M, C, 256, false, true, DIAG == 9);
    matmul2d<scoreDescriptor, execution_simdgroups<SG>> scoreOp;
    constexpr auto valueDescriptor = matmul2d_descriptor(M, 256, C, false, false, DIAG == 9,
                                                         matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<valueDescriptor, execution_simdgroups<(SGV != 0 ? SGV : SG)>> valueOp;
    // Whether this thread takes part in the score matmul and so owns score elements.
    const bool scoring = tid / 32u < SG;

    auto queries = tensor(qB + ulong(queryBase) * 256u, dextents<int, 2>{256, M}, array<int, 2>{1, 256});
    auto q0 = queries.template slice<256, M>(0, 0);
    auto kProto = tensor(kCodes, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
    auto k0 = kProto.template slice<256, C>(0, 0);
    auto score = scoreOp.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    // Row sums of the probabilities, per element: plain registers (a second cooperative
    // tensor used as storage was tried first; DIAG 42 times the difference).
    auto sumsTensor = scoreOp.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();

    threadgroup PT *weightedA = reinterpret_cast<threadgroup PT *>(staged);
    threadgroup PT *weightedB = weightedA + uint(M) * C;
    auto wProto = tensor(weightedA, dextents<int, 2>{C, M}, array<int, 2>{1, C});
    auto w0 = wProto.template slice<C, M>(0, 0);
    auto vProto = tensor(vCodes, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
    auto v0 = vProto.template slice<256, C>(0, 0);
    auto running = valueOp.template get_destination_cooperative_tensor<decltype(w0), decltype(v0), float>();
#pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0.0f;
    if (DIAG == 42) {
#pragma unroll
        for (ushort i = 0; i < sumsTensor.get_capacity(); ++i) sumsTensor[i] = 0.0f;
    }

    // The element partition of a cooperative tensor is fixed, so each thread resolves its
    // (query, token) coordinates once and keeps its rows' references in registers.
    // The accelerator tiles queries in units of 32, so the score tensor is padded to that.
    constexpr ushort CAP = CAPACITY != 0 ? CAPACITY : ((M + 31) / 32 * 32 * C) / (SG * 32);
    float sums[CAP];
#pragma unroll
    for (ushort i = 0; i < CAP; ++i) sums[i] = 0.0f;
    ushort em[CAP], et[CAP];
    float eref[CAP];
    bool evalid[CAP];
    const ushort capacity = score.get_capacity();
#pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
        evalid[i] = scoring && capacity == CAP && score.is_valid_element(i);
        const auto index = score.get_multidimensional_index(evalid[i] ? i : ushort(0));
        et[i] = ushort(index[0]); em[i] = ushort(index[1]);
        eref[i] = -INFINITY;
    }

    // In the shipped shape a thread's sixteen score elements are four consecutive tokens for
    // each of four queries (element i is token et[0] + (i & 3)), so the scales of a chunk's
    // tokens could be two vector loads a thread instead of thirty-two scalar ones. Tried
    // (DIAG 39): no gain at any context, so the loads are not what the softmax costs.
    constexpr bool QUAD = DIAG == 39;
    bool quads = true;
#pragma unroll
    for (ushort i = 0; i < CAP; ++i) { if (evalid[i] && et[i] != ushort(et[0] + (i & 3))) quads = false; }
    const uint quadToken = et[0];
    if (tid < M) { reference[tid] = -INFINITY; bump[tid] = 0.0f; }
    // "Some row's scores rose past its reference in this chunk": a word of threadgroup memory
    // that any thread sets to 1 and all read after the barrier. Plain stores and loads: the
    // atomic kind cost a fifth of the scan at 36K context (every thread loads it every chunk),
    // and every writer writes the same value. DIAG 33 keeps the atomic form for comparison.
    threadgroup uint *raised = reinterpret_cast<threadgroup uint *>(flags);
    if (tid < 2u && DIAG != 32) { if (DIAG == 33) atomic_store_explicit(&flags[tid], 0u, memory_order_relaxed); else raised[tid] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint chunksPerPage = 256u / C;
    const bool fullBlock = count * perKV == M;
    uint parity = 0u;
    bool first = DIAG != 35 && DIAG != 37;
    if (DIAG == 35 || DIAG == 36) {
        const uint tokenBase = (pageTable[tableBase] * p.kvHeads + kvHead) * 256u;
        for (uint round = 0; round < 3u; ++round) {
            auto kt = tensor(kCodes + ulong(tokenBase) * 256u, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
            auto ks = kt.template slice<256, C>(0, 0);
            if (scoring) scoreOp.run(q0, ks, score);
            if (first) {
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) {
                    if (!evalid[i]) continue;
                    score[i] = score[i] * kScale[tokenBase + et[i]] * p.scale;
                    staged[uint(em[i]) * C + et[i]] = score[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < M) {
                    float top = -INFINITY;
                    for (uint t = 0; t < C; ++t) top = max(top, staged[tid * C + t]);
                    reference[tid] = top;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) eref[i] = reference[em[i]];
            }
            threadgroup PT *weighted = parity != 0u ? weightedB : weightedA;
#pragma unroll
            for (ushort i = 0; i < CAP; ++i) {
                if (!evalid[i]) continue;
                const uint t = et[i];
                const float s = (first ? score[i] : score[i] * kScale[tokenBase + t] * p.scale) - (DIAG == 35 ? 0.0f : eref[i]);
                const float probability = exp(min(s, (DIAG == 40 ? 70.0f : SPI_CLAMP)));
                { if (DIAG == 42) sumsTensor[i] += probability; else sums[i] += probability; }
                weighted[uint(em[i]) * C + t] = PT(probability * vScale[tokenBase + t]);
            }
            first = false;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto wt = tensor(weighted, dextents<int, 2>{C, M}, array<int, 2>{1, C});
            auto ws = wt.template slice<C, M>(0, 0);
            auto vt = tensor(vCodes + ulong(tokenBase) * 256u, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
            auto vs = vt.template slice<256, C>(0, 0);
            valueOp.run(ws, vs, running);
            parity ^= 1u;
        }
    }
    // DIAG 28 / 29: the whole walk 4 / 16 times over (timing: do longer threadgroups share the cores?).
    for (uint again = 0; again < (DIAG == 28 ? 4u : DIAG == 29 ? 16u : (DIAG == 35 || DIAG == 36) ? 0u : 1u); ++again)
    for (uint pg = firstPage; pg < (DIAG == 22 ? firstPage : lastPage); ++pg) {   // 22: set-up and final writes only
        const uint pageTokens = (pageTable[tableBase + pg] * p.kvHeads + kvHead) * 256u;
        for (uint chunk = 0; chunk < chunksPerPage; ++chunk) {
            const uint t0 = pg * 256u + chunk * C;
            if (t0 >= maxVisible) break;
            if (DIAG == 24 && t0 + C > rowPos[0]) break;        // timing: leave out the chunks that hold the step's own rows
            // DIAG 8 makes the chunk loop-invariant, and the compiler then lifts the whole score
            // product out of the loop: it times "no score matmul", not "no memory traffic".
            // For the latter, set aliasTokens (SPLOSH_ATTN_ALIAS).
            const uint tokenBase = DIAG == 8 ? kvHead * 256u
                : p.aliasTokens != 0u ? kvHead * 256u + (t0 % p.aliasTokens) * p.kvHeads : pageTokens + chunk * C;
            auto kt = tensor(kCodes + ulong(tokenBase) * 256u, dextents<int, 2>{256, C}, array<int, 2>{1, 256});
            auto ks = kt.template slice<256, C>(0, 0);
            // DIAG 17 / 18: no matmuls in the first / the last span (timing: which threadgroups are dear).
            const bool idle = (DIAG == 17 && span == 0u) || (DIAG == 18 && span + 1u == p.spans);
            if (DIAG != 2 && DIAG != 4 && DIAG != 6 && DIAG != 7 && !idle) { if (scoring) scoreOp.run(q0, ks, score); }
            else {
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) { if (scoring && capacity == CAP) score[i] = 0.0f; }
            }

            float4 kQuad = float4(0.0f), vQuad = float4(0.0f);
            if (QUAD) {
                kQuad = float4(*(device const packed_float4 *)(kScale + tokenBase + quadToken));
                vQuad = float4(*(device const packed_float4 *)(vScale + tokenBase + quadToken));
            }
            // Rows see tokens up to and including their own position.
            const bool causal = t0 + C > firstPos;
            const bool scaled = first;
            if (first) {
                // The span's first chunk fixes each row's reference: its maximum there.
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) {
                    if (!evalid[i]) continue;
                    const uint t = et[i], rowInBlock = uint(em[i]) / perKV;
                    const bool visible = rowInBlock < count && t0 + t <= firstPos + rowInBlock;
                    score[i] = visible ? score[i] * (QUAD ? kQuad[i & 3] : kScale[tokenBase + t]) * p.scale : -INFINITY;
                    staged[uint(em[i]) * C + t] = score[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < M) {
                    float top = -INFINITY;
                    for (uint t = 0; t < C; ++t) top = max(top, staged[tid * C + t]);
                    reference[tid] = top;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) eref[i] = reference[em[i]];
                first = false;
            }

            threadgroup PT *weighted = parity != 0u ? weightedB : weightedA;
            // Which of this thread's elements rose past their row's reference by more than the
            // margin: a bit each, gathered without a store in the loop (a conditional store per
            // element there cost a seventh of the scan), and acted on below, which is rare.
            uint over = 0u;
            if (DIAG == 3 || DIAG == 6 || DIAG == 7 || DIAG == 8) {
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) { if (evalid[i]) { if (DIAG == 42) sumsTensor[i] += score[i]; else sums[i] += score[i]; } }
            } else if (scaled || ((causal || !fullBlock) && DIAG != 16)) {
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) {
                    if (!evalid[i]) continue;
                    const uint t = et[i], m = em[i];
                    float sc = score[i];
                    if (!scaled) {
                        const uint rowInBlock = m / perKV;
                        const bool visible = rowInBlock < count && t0 + t <= firstPos + rowInBlock;
                        sc = visible ? sc * (QUAD ? kQuad[i & 3] : kScale[tokenBase + t]) * p.scale : -INFINITY;
                    }
                    float probability = 0.0f;
                    if (sc != -INFINITY) {
                        const float s = sc - eref[i];
                        probability = (DIAG == 31 && s < -60.0f) ? 0.0f : exp(min(s, (DIAG == 40 ? 70.0f : SPI_CLAMP)));
                        if (DIAG == 41) {
                            if (s > SPI_BUMP) { bump[m] = 1.0f; raised[parity] = 1u; }
                        } else {
                            over |= uint(s > (DIAG == 40 ? 50.0f : SPI_BUMP)) << i;
                        }
                    }
                    { if (DIAG == 42) sumsTensor[i] += probability; else sums[i] += probability; }
                    weighted[m * C + t] = probability > 0.0f ? PT(probability * (QUAD ? vQuad[i & 3] : vScale[tokenBase + t])) : PT(0.0f);
                }
            } else {
                // Every query sees every token of the chunk.
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) {
                    if (!evalid[i]) continue;
                    const uint t = et[i];
                    const float s = score[i] * (QUAD ? kQuad[i & 3] : kScale[tokenBase + t]) * p.scale - eref[i];
                    const float probability = (DIAG == 31 && s < -60.0f) ? 0.0f : exp(min(s, (DIAG == 40 ? 70.0f : SPI_CLAMP)));
                    if (DIAG == 41) {
                        if (s > SPI_BUMP) { bump[em[i]] = 1.0f; raised[parity] = 1u; }
                    } else {
                        over |= uint(s > (DIAG == 40 ? 50.0f : SPI_BUMP)) << i;
                    }
                    { if (DIAG == 42) sumsTensor[i] += probability; else sums[i] += probability; }
                    weighted[uint(em[i]) * C + t] = PT(probability * (QUAD ? vQuad[i & 3] : vScale[tokenBase + t]));
                }
            }
            if (over != 0u && DIAG != 15 && DIAG != 32) {
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) { if ((over >> i) & 1u) bump[em[i]] = 1.0f; }
                if (DIAG == 33) atomic_store_explicit(&flags[parity], 1u, memory_order_relaxed); else raised[parity] = 1u;
            }
            if (DIAG != 7) threadgroup_barrier(mem_flags::mem_threadgroup);

            const bool moved = DIAG != 32 && (DIAG == 33 ? atomic_load_explicit(&flags[parity], memory_order_relaxed) != 0u : raised[parity] != 0u);
            auto wt = tensor(weighted, dextents<int, 2>{C, M}, array<int, 2>{1, C});
            auto ws = wt.template slice<C, M>(0, 0);
            // DIAG 43: the same bytes read as if the values were stored dimension by dimension
            // (timing only: would the value matmul be faster on that layout?).
            auto vt = tensor(vCodes + ulong(tokenBase) * 256u, dextents<int, 2>{256, C},
                             DIAG == 43 ? array<int, 2>{int(C), 1} : array<int, 2>{1, 256});
            auto vs = vt.template slice<256, C>(0, 0);
            if (DIAG != 1 && DIAG != 4 && DIAG != 6 && DIAG != 7 && !idle) valueOp.run(ws, vs, running);

            if (moved) {
                // Rare: a row's scores rose well above its reference. Shift that row down.
                const float factor = exp(-(DIAG == 40 ? 50.0f : SPI_BUMP));
#pragma unroll
                for (ushort i = 0; i < running.get_capacity(); ++i) {
                    if (!running.is_valid_element(i)) continue;
                    if (bump[uint(running.get_multidimensional_index(i)[1])] != 0.0f) running[i] *= factor;
                }
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) {
                    if (evalid[i] && bump[em[i]] != 0.0f) { if (DIAG == 42) sumsTensor[i] *= factor; else sums[i] *= factor; }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < M && bump[tid] != 0.0f) { reference[tid] += (DIAG == 40 ? 50.0f : SPI_BUMP); bump[tid] = 0.0f; }
                if (tid == 0u) { if (DIAG == 33) atomic_store_explicit(&flags[parity], 0u, memory_order_relaxed); else raised[parity] = 0u; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
                for (ushort i = 0; i < CAP; ++i) eref[i] = reference[em[i]];
            }
            parity ^= 1u;
        }
    }

    if (DIAG == 38) {
        // Partition map: for each thread of the first threadgroup, its score elements' (token,
        // query) pairs and how many are valid, then its value elements' first and last (dim, query).
        if (group.x == 0u && group.y == 0u) {
            device float *row = partials + tid * 40u;
            for (ushort i = 0; i < CAP && i < 16; ++i) { row[i * 2u] = float(et[i]); row[i * 2u + 1u] = evalid[i] ? float(em[i]) : -1.0f; }
            const auto first_ = running.get_multidimensional_index(ushort(0));
            const auto last_ = running.get_multidimensional_index(ushort(running.get_capacity() - 1));
            row[32] = float(first_[0]); row[33] = float(first_[1]); row[34] = float(last_[0]); row[35] = float(last_[1]);
            row[36] = float(running.get_capacity());
        }
        return;
    }
    if (DIAG == 5) {
        // Partition probe: per thread, capacity and valid counts of the score and output tensors.
        ushort validScore = 0, validRunning = 0;
        for (ushort i = 0; i < capacity; ++i) validScore += score.is_valid_element(i) ? 1 : 0;
        for (ushort i = 0; i < running.get_capacity(); ++i) validRunning += running.is_valid_element(i) ? 1 : 0;
        if (group.x == 0u && group.y == 0u) {
            partials[tid * 4u] = float(capacity); partials[tid * 4u + 1u] = float(validScore);
            partials[tid * 4u + 2u] = float(running.get_capacity()); partials[tid * 4u + 3u] = float(validRunning);
        }
        return;
    }
    // Row sums: every simdgroup has finished reading the probability buffers before they are reused.
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
        if (evalid[i]) staged[uint(em[i]) * C + et[i]] = DIAG == 42 ? float(sumsTensor[i]) : sums[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!running.is_valid_element(i)) continue;
        const auto index = running.get_multidimensional_index(i);
        const uint m = uint(index[1]), d = uint(index[0]);
        if (m / perKV >= count) continue;
        const uint row = r0 + m / perKV, head = kvHead * perKV + m % perKV;
        partials[((row * p.heads + head) * p.spans + span) * 258u + 2u + d] = running[i];
    }
    if (tid < M && tid / perKV < count) {
        float total = 0.0f;
        for (uint t = 0; t < C; ++t) total += staged[tid * C + t];
        const uint row = r0 + tid / perKV, head = kvHead * perKV + tid % perKV;
        device float *partial = partials + ((row * p.heads + head) * p.spans + span) * 258u;
        // A score partition this kernel was not sized for poisons the output rather than
        // silently dropping scores (rows below M are always in the scoring simdgroups).
        partial[0] = capacity != CAP || (QUAD && !quads) ? NAN : reference[tid];
        partial[1] = total;
    }
}

#define SP_DEFINE_ATTN_SCAN_I8(NAME, M, C, SG, PT, DIAG)                                             \
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
    threadgroup float staged[M * C * (sizeof(PT) / 2)];                                           \
    threadgroup float reference[M];                                                               \
    threadgroup float bump[M];                                                                    \
    threadgroup atomic_uint flags[2];                                                             \
    spi_scan<M, C, SG, PT, DIAG>(qB, blocks, rowSlot, rowPos, pageTable, kCodes, kScale, vCodes,  \
                                 vScale, partials, p, group, tid, staged, reference, bump, flags);\
}
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8, 48, 64, 8, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c32s8d5, 48, 32, 8, bfloat, 5)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c32s4d5, 48, 32, 4, bfloat, 5)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c128s8, 48, 128, 8, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s4, 48, 64, 4, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m96c64s8, 96, 64, 8, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m96c32s8, 96, 32, 8, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m24c64s4, 24, 64, 4, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8f, 48, 64, 8, float, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d1, 48, 64, 8, bfloat, 1)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d2, 48, 64, 8, bfloat, 2)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d3, 48, 64, 8, bfloat, 3)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d4, 48, 64, 8, bfloat, 4)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d5, 48, 64, 8, bfloat, 5)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s4d5, 48, 64, 4, bfloat, 5)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m96c64s8d5, 96, 64, 8, bfloat, 5)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m64c64s8d5, 64, 64, 8, bfloat, 5)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m32c64s8d5, 32, 64, 8, bfloat, 5)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d6, 48, 64, 8, bfloat, 6)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d7, 48, 64, 8, bfloat, 7)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m96c64s8d1, 96, 64, 8, bfloat, 1)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m96c64s8d2, 96, 64, 8, bfloat, 2)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m96c64s8d3, 96, 64, 8, bfloat, 3)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m96c64s8d4, 96, 64, 8, bfloat, 4)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m96c64s8d6, 96, 64, 8, bfloat, 6)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m96c64s8d7, 96, 64, 8, bfloat, 7)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d8, 48, 64, 8, bfloat, 8)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d9, 48, 64, 8, bfloat, 9)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d24, 48, 64, 8, bfloat, 24)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d26, 48, 64, 8, bfloat, 26)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d30, 48, 64, 8, bfloat, 30)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d31, 48, 64, 8, bfloat, 31)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d32, 48, 64, 8, bfloat, 32)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d33, 48, 64, 8, bfloat, 33)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d34, 48, 64, 8, bfloat, 34)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d38, 48, 64, 8, bfloat, 38)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d39, 48, 64, 8, bfloat, 39)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d40, 48, 64, 8, bfloat, 40)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d41, 48, 64, 8, bfloat, 41)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d42, 48, 64, 8, bfloat, 42)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d43, 48, 64, 8, bfloat, 43)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d35, 48, 64, 8, bfloat, 35)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d36, 48, 64, 8, bfloat, 36)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d37, 48, 64, 8, bfloat, 37)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d28, 48, 64, 8, bfloat, 28)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d29, 48, 64, 8, bfloat, 29)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d27, 48, 64, 8, bfloat, 27)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d21, 48, 64, 8, bfloat, 21)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d22, 48, 64, 8, bfloat, 22)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d17, 48, 64, 8, bfloat, 17)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d18, 48, 64, 8, bfloat, 18)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d15, 48, 64, 8, bfloat, 15)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m48c64s8d16, 48, 64, 8, bfloat, 16)

#define SP_DEFINE_ATTN_SCAN_I8_CAP(NAME, M, C, SG, PT, DIAG, CAPACITY)                            \
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
    threadgroup float staged[M * C * (sizeof(PT) / 2)];                                           \
    threadgroup float reference[M];                                                               \
    threadgroup float bump[M];                                                                    \
    threadgroup atomic_uint flags[2];                                                             \
    spi_scan<M, C, SG, PT, DIAG, CAPACITY>(qB, blocks, rowSlot, rowPos, pageTable, kCodes, kScale,\
                                 vCodes, vScale, partials, p, group, tid, staged, reference, bump, flags);\
}
SP_DEFINE_ATTN_SCAN_I8_CAP(sp_attn_scan_i8_m48c64s6d5, 48, 64, 6, bfloat, 5, 16)
SP_DEFINE_ATTN_SCAN_I8_CAP(sp_attn_scan_i8_m48c64s3d5, 48, 64, 3, bfloat, 5, 32)
SP_DEFINE_ATTN_SCAN_I8_CAP(sp_attn_scan_i8_m48c64s12d5, 48, 64, 12, bfloat, 5, 8)
SP_DEFINE_ATTN_SCAN_I8_CAP(sp_attn_scan_i8_m48c64s6, 48, 64, 6, bfloat, 0, 16)
SP_DEFINE_ATTN_SCAN_I8_CAP(sp_attn_scan_i8_m48c64s3, 48, 64, 3, bfloat, 0, 32)
// The capacity is what the partition probe (the d5 variants, SPLOSH_DUMP_PARTIALS) reports for
// the shape, not what the formula gives: a kernel sized wrongly computes no scores, poisons its
// output, and looks fast. m48c64s12 was defined with 8 and m48c32s8 without, and both were.
SP_DEFINE_ATTN_SCAN_I8_CAP(sp_attn_scan_i8_m48c64s12, 48, 64, 12, bfloat, 0, 16)
SP_DEFINE_ATTN_SCAN_I8_CAP(sp_attn_scan_i8_m48c32s8, 48, 32, 8, bfloat, 0, 16)
SP_DEFINE_ATTN_SCAN_I8_CAP(sp_attn_scan_i8_m48c32s4, 48, 32, 4, bfloat, 0, 16)
// Ten rows (60 queries) per 64-query tile: the accelerator pads a 48-query tile to 64 anyway.
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m64c64s8, 64, 64, 8, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m64c32s8, 64, 32, 8, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m32c64s8, 32, 64, 8, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m32c64s4, 32, 64, 4, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m32c128s8, 32, 128, 8, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m32c256s8, 32, 256, 8, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m16c128s8, 16, 128, 8, bfloat, 0)
SP_DEFINE_ATTN_SCAN_I8(sp_attn_scan_i8_m16c256s8, 16, 256, 8, bfloat, 0)

// Timing probes: wider tiles than the stored 128-row tile (weights aliased, results meaningless).
#define SP_DEFINE_NAT_PROBE(NAME, M, SG, N)                                                       \
kernel void NAME(device uchar *packed [[buffer(0)]],                                              \
                 device const ushort *scales [[buffer(1)]],                                       \
                 device const ushort *biases [[buffer(2)]],                                       \
                 device bfloat *a [[buffer(3)]],                                                  \
                 device float *out [[buffer(4)]],                                                 \
                 device const float *residual [[buffer(5)]],                                      \
                 device const float *sums [[buffer(6)]],                                          \
                 constant SpNaParams &p [[buffer(7)]],                                            \
                 uint3 group [[threadgroup_position_in_grid]])                                    \
{                                                                                                 \
    threadgroup float biasStage[1];                                                               \
    sp_na_tiled<M, SG, 6, N>(packed, scales, biases, a, out, residual, sums, p, group, biasStage);\
}
SP_DEFINE_NAT_PROBE(sp_gemm_q4_nat_m32n256s8_alias, 32, 8, 256)
SP_DEFINE_NAT_PROBE(sp_gemm_q4_nat_m32n256s4_alias, 32, 4, 256)
SP_DEFINE_NAT_PROBE(sp_gemm_q4_nat_m32n64s2_alias, 32, 2, 64)
SP_DEFINE_NAT_PROBE(sp_gemm_q4_nat_m32n64s4_alias, 32, 4, 64)
SP_DEFINE_NAT_PROBE(sp_gemm_q4_nat_m16n256s4_alias, 16, 4, 256)
SP_DEFINE_NAT_PROBE(sp_gemm_q4_nat_m16n256s8_alias, 16, 8, 256)
SP_DEFINE_NA(sp_gemm_q4_na_m32n256s8, 32, 256, 8)
SP_DEFINE_NA(sp_gemm_q4_na_m32n256s4, 32, 256, 4)
SP_DEFINE_NA(sp_gemm_q4_na_m32n64s2, 32, 64, 2)
SP_DEFINE_NA(sp_gemm_q4_na_m16n256s4, 16, 256, 4)

#define SP_DEFINE_ATTN_SCAN_I8_SPLIT(NAME, M, C, SG, SGV)                                         \
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
    threadgroup float staged[M * C];                                                              \
    threadgroup float reference[M];                                                               \
    threadgroup float bump[M];                                                                    \
    threadgroup atomic_uint flags[2];                                                             \
    spi_scan<M, C, SG, bfloat, 0, 0, SGV>(qB, blocks, rowSlot, rowPos, pageTable, kCodes, kScale, \
                                 vCodes, vScale, partials, p, group, tid, staged, reference, bump, flags);\
}
SP_DEFINE_ATTN_SCAN_I8_SPLIT(sp_attn_scan_i8_m48c64s4v16, 48, 64, 4, 16)
SP_DEFINE_ATTN_SCAN_I8_SPLIT(sp_attn_scan_i8_m48c64s4v8, 48, 64, 4, 8)
SP_DEFINE_ATTN_SCAN_I8_SPLIT(sp_attn_scan_i8_m48c64s8v16, 48, 64, 8, 16)
SP_DEFINE_ATTN_SCAN_I8_SPLIT(sp_attn_scan_i8_m64c64s4v16, 64, 64, 4, 16)
SP_DEFINE_ATTN_SCAN_I8_SPLIT(sp_attn_scan_i8_m96c64s8v24, 96, 64, 8, 24)
SP_DEFINE_ATTN_SCAN_I8_SPLIT(sp_attn_scan_i8_m96c64s4v24, 96, 64, 4, 24)
SP_DEFINE_ATTN_SCAN_I8_SPLIT(sp_attn_scan_i8_m32c64s2v8, 32, 64, 2, 8)
SP_DEFINE_ATTN_SCAN_I8_SPLIT(sp_attn_scan_i8_m48c128s8v16, 48, 128, 8, 16)
SP_DEFINE_ATTN_SCAN_I8_SPLIT(sp_attn_scan_i8_m64c128s8v16, 64, 128, 8, 16)

// =============================================================================================
// Draft-model attention on the accelerators.
//
// The draft's block of eight rows attends, non-causally, to a ring of up to 2,304 context
// positions per slot (fp16 K and V, [slot * capacity + position % capacity][kvHead][128]).
// Eight rows x four query heads per KV head are exactly one 32-query tile. The ring is walked
// in 64-slot chunks of ring index, so a chunk never wraps; which position a slot currently
// holds, and whether a row may see it, is arithmetic.
//
// Scores are fp16 x fp16 on the accelerator, the softmax is the ordinary online one (running
// maximum per row, three barriers per chunk — there are at most 36 chunks, spread over spans),
// and the values are accumulated with bf16 probabilities x fp16 V.
// Threadgroup grid: (kvHeads * spans, requests), 256 threads per group.
// Partial per (row, head, span): [running max, running sum, 128 values], as the merge expects.
struct SpdNaParams {
    uint requests;
    uint heads;
    uint kvHeads;
    uint capacity;
    uint window;
    uint spans;
    uint rowCap;
    float scale;
};

kernel void sp_draft_attention_na(device half *qB [[buffer(0)]],
                                  device const uint *rowSlot [[buffer(1)]],
                                  device const uint *rowPos [[buffer(2)]],
                                  device const uint *rowLo [[buffer(3)]],
                                  device const uint *rowHi [[buffer(4)]],
                                  device half *kring [[buffer(5)]],
                                  device half *vring [[buffer(6)]],
                                  device float *partials [[buffer(7)]],
                                  constant SpdNaParams &p [[buffer(8)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]])
{
    constexpr ushort M = 32, C = 64, D = 128, SG = 8, CAP = (M * C) / (SG * 32);
    threadgroup float staged[M * C];
    threadgroup bfloat weighted[M * C];
    threadgroup float runMax[M];
    threadgroup float rescale[M];

    const uint kvHead = group.x % p.kvHeads, span = group.x / p.kvHeads, request = group.y;
    if (span >= p.spans || request >= p.requests) return;
    const uint perKV = p.heads / p.kvHeads;          // 4
    const uint rowsPerRequest = M / perKV;           // 8
    const uint r0 = request * rowsPerRequest;
    const uint slot = rowSlot[r0];
    const int hi = int(rowHi[r0]);
    const int capacity = int(p.capacity);
    const uint ringChunks = p.capacity / C;
    const uint perSpan = (ringChunks + p.spans - 1u) / p.spans;
    const uint firstChunk = span * perSpan, lastChunk = min(ringChunks, firstChunk + perSpan);

    // The oldest position any row of the block may see.
    int lowest = hi;
    for (uint r = 0; r < rowsPerRequest; ++r) {
        const int position = int(rowPos[r0 + r]);
        lowest = min(lowest, max(int(rowLo[r0 + r]), position + 1 - int(p.window)));
    }

    constexpr auto scoreDescriptor = matmul2d_descriptor(M, C, D, false, true, false);
    matmul2d<scoreDescriptor, execution_simdgroups<SG>> scoreOp;
    constexpr auto valueDescriptor = matmul2d_descriptor(M, D, C, false, false, false,
                                                         matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<valueDescriptor, execution_simdgroups<SG>> valueOp;

    auto queries = tensor(qB + ulong((kvHead * p.rowCap + r0) * perKV) * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    auto q0 = queries.template slice<D, M>(0, 0);
    const int ringStride = int(p.kvHeads * D);
    auto kProto = tensor(kring, dextents<int, 2>{D, C}, array<int, 2>{1, ringStride});
    auto k0 = kProto.template slice<D, C>(0, 0);
    auto score = scoreOp.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    auto sums = scoreOp.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    auto wProto = tensor(weighted, dextents<int, 2>{C, M}, array<int, 2>{1, C});
    auto w0 = wProto.template slice<C, M>(0, 0);
    auto vProto = tensor(vring, dextents<int, 2>{D, C}, array<int, 2>{1, ringStride});
    auto v0 = vProto.template slice<D, C>(0, 0);
    auto running = valueOp.template get_destination_cooperative_tensor<decltype(w0), decltype(v0), float>();
#pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0.0f;

    // Each thread's score coordinates, and the oldest position its rows may see.
    const ushort capacityCount = score.get_capacity();
    ushort em[CAP], et[CAP];
    int elo[CAP];
    bool evalid[CAP];
#pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
        evalid[i] = capacityCount == CAP && score.is_valid_element(i);
        const auto index = score.get_multidimensional_index(evalid[i] ? i : ushort(0));
        et[i] = ushort(index[0]); em[i] = ushort(index[1]);
        const uint row = r0 + uint(em[i]) / perKV;
        elo[i] = max(int(rowLo[row]), int(rowPos[row]) + 1 - int(p.window));
        sums[i] = 0.0f;
    }
    if (tid < M) runMax[tid] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int newestSlot = (hi - 1) % capacity;
    for (uint rc = firstChunk; rc < lastChunk; ++rc) {
        const int s0 = int(rc * C);
        // Slot s holds the newest position below `hi` with that residue.
        const int lastPosition = hi - 1 - ((hi - 1 - (s0 + int(C) - 1)) % capacity + capacity) % capacity;
        const bool holdsNewest = newestSlot >= s0 && newestSlot < s0 + int(C);
        if (hi <= 0 || (!holdsNewest && lastPosition < lowest)) continue;

        const ulong chunkBase = (ulong(slot) * p.capacity + rc * C) * p.kvHeads + kvHead;
        auto kt = tensor(kring + chunkBase * D, dextents<int, 2>{D, C}, array<int, 2>{1, ringStride});
        auto ks = kt.template slice<D, C>(0, 0);
        scoreOp.run(q0, ks, score);
#pragma unroll
        for (ushort i = 0; i < CAP; ++i) {
            if (!evalid[i]) continue;
            const int s = s0 + int(et[i]);
            const int position = hi - 1 - ((hi - 1 - s) % capacity + capacity) % capacity;
            score[i] = position >= elo[i] ? score[i] * p.scale : -INFINITY;
            staged[uint(em[i]) * C + et[i]] = score[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < M) {
            float top = -INFINITY;
            for (uint t = 0; t < C; ++t) top = max(top, staged[tid * C + t]);
            const float newMax = max(runMax[tid], top);
            rescale[tid] = newMax == -INFINITY ? 1.0f : exp(runMax[tid] - newMax);
            runMax[tid] = newMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            if (!running.is_valid_element(i)) continue;
            running[i] *= rescale[uint(running.get_multidimensional_index(i)[1])];
        }
#pragma unroll
        for (ushort i = 0; i < CAP; ++i) {
            if (!evalid[i]) continue;
            const float probability = score[i] == -INFINITY ? 0.0f : exp(score[i] - runMax[em[i]]);
            sums[i] = sums[i] * rescale[em[i]] + probability;
            weighted[uint(em[i]) * C + et[i]] = bfloat(probability);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto vt = tensor(vring + chunkBase * D, dextents<int, 2>{D, C}, array<int, 2>{1, ringStride});
        auto vs = vt.template slice<D, C>(0, 0);
        valueOp.run(w0, vs, running);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
        if (evalid[i]) staged[uint(em[i]) * C + et[i]] = sums[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!running.is_valid_element(i)) continue;
        const auto index = running.get_multidimensional_index(i);
        const uint m = uint(index[1]), d = uint(index[0]);
        const uint row = r0 + m / perKV, head = kvHead * perKV + m % perKV;
        partials[((row * p.heads + head) * p.spans + span) * (D + 2u) + 2u + d] = running[i];
    }
    if (tid < M) {
        float total = 0.0f;
        for (uint t = 0; t < C; ++t) total += staged[tid * C + t];
        const uint row = r0 + tid / perKV, head = kvHead * perKV + tid % perKV;
        device float *partial = partials + ((row * p.heads + head) * p.spans + span) * (D + 2u);
        // A score partition this kernel was not sized for poisons the output.
        partial[0] = capacityCount != CAP ? NAN : runMax[tid];
        partial[1] = total;
    }
}
