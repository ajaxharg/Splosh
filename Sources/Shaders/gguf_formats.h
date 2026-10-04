#pragma once

#include <metal_stdlib>

using namespace metal;

// gguf_formats.h — llama.cpp's quantised weight formats as the GGUF kernels read them.
//
// A GGUF weight keeps its codes at their native width. Its native blocks are re-ordered once,
// by SploshModel/GgufPlanes.swift, into three planes tiled like the engine's other weights:
//
//   plane0 [rows / 128][groups][128][Plane0 bytes]                the codes of one row's group
//   plane1 [rows / 128][groups][128][Plane1 bytes]                their high bits, where a code is split
//   meta   [rows / 128][groups / MetaGroups][128][MetaBytes]      one native block's scale fields
//
// with groups = inner / 32. Inside a group the 32 elements are in chunk order: chunk c (0..3)
// holds elements 4c..4c+3 and 16+4c..16+4c+3 as four pairs, pair p being elements e0 and e0 + 1
// with e0 = 16 (p >> 1) + 4c + 2 (p & 1), so element e is at slot 8c + 2p + (e & 1). A chunk
// decodes to two runs of four weights, which is what one store of the stage takes. The planes
// pack the slots as follows.
//
//   4-bit linear codes (Q4_K, the low bits of Q5_K and Q6_K): word c holds chunk c, pair p's e0
//     at bits 4p and its e1 at bits 16 + 4p.
//   Every other field is a little-endian bit string of the 32 slots: IQ4 indices (4 bits, so
//     byte p of word c is pair p), Q8_0 values (8), Q6_K high and Q3_K low bits (2), Q5_K fifth
//     and Q3_K high bits (1).
//   IQ3_S word c: bits 0..7 and 8..15 the low grid index bits of elements 4c..4c+3 and
//     16+4c..16+4c+3, 16..23 the sign bits of chunk c's slots, 24 and 25 the two ninth index
//     bits, 26..29 the group's scale.
//
// The plane encoding, the chunk order and the decoders below are adapted from Splash and its M5
// fork Splish (Apache-2.0): runtime/metal/abi/QuantFormat.h, kernels/common/quant_formats.h and
// kernels/common/gguf_staged.h. The formats, the arithmetic that makes a weight of a code and
// the two tables are llama.cpp's (MIT, Copyright (c) 2023-2026 The ggml authors):
// ggml-quants.c and ggml-common.h. The values are those of SploshModel/GgufQuant.swift.

#define SP_GGUF_TILE_ROWS 128u

// The sixteen values a 4-bit code of IQ4_NL and IQ4_XS indexes (llama.cpp's kvalues_iq4nl).
constant constexpr int8_t sp_gguf_iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// The IQ3_S grid by 9-bit index (llama.cpp's iq3s_grid): an entry holds the magnitudes of four
// consecutive elements in its bytes 0 to 3, byte 0 first.
constant constexpr uint sp_gguf_iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

// A format F states the sizes of its records (Plane0, Plane1 and MetaBytes, in bytes), the
// groups one meta record covers (MetaGroups: its native block), the entries of its pair table
// (PairEntries), the offset its linear codes are stored with (Zero), and how to read it:
//
//   load(plane0, plane1) -> Payload      the record of one (row, group)
//   loadMeta(meta) -> Meta               the record of one (row, native block)
//   chunk(Payload, c) -> Chunk           chunk c of the group
//   coef(Meta, j) -> SpGgufCoef          the scale and offset of group j of the block; with
//                                        ScaleInChunk, coef(Meta, Chunk): the scale is stored
//                                        with the codes, in every chunk of its group
//
// and one accessor for the elements of a chunk, by Kind:
//
//   SpGgufLinear     codes(Chunk) -> uint4, pair p in component p with e0 at bit 0 and e1 at
//                    bit 16; weight = s (code - Zero) + m
//   SpGgufCodebook   indices(Chunk) -> uint, byte p indexing pair p in the pair table of
//                    value(i); weight = s value
//   SpGgufInt8       values(Chunk) -> uint2, the signed bytes of pairs 0, 1 (x) and 2, 3 (y);
//                    weight = s byte
//   SpGgufGrid       grid(Chunk) -> uint2, the magnitudes of pairs 0, 1 (x) and 2, 3 (y), a
//                    byte each; signs(Chunk), bit 2p + i negating element i of pair p;
//                    weight = s signed magnitude
//
// The codebook and int8 formats name Scale, the narrowest type that holds s exactly: their
// product is then rounded to fp16 once whichever type it is made in.
enum SpGgufKind : ushort { SpGgufLinear, SpGgufCodebook, SpGgufInt8, SpGgufGrid };

// What scales a group: s.x and m.x its first sixteen elements (pairs 0 and 1 of each chunk),
// s.y and m.y the rest. The two are equal where one coefficient covers the group.
struct SpGgufCoef {
    float2 s;
    float2 m = float2(0.0f);
};

#define SP_GGUF_FORMAT(P0, P1, META, GROUPS, ZERO, KIND, IN_CHUNK)                                \
    enum : uint { Plane0 = P0, Plane1 = P1, MetaBytes = META, MetaGroups = GROUPS, Zero = ZERO,   \
                  PairEntries = KIND == SpGgufCodebook ? 256 : 1 };                               \
    static constexpr constant SpGgufKind Kind = KIND;                                             \
    static constexpr constant bool ScaleInChunk = IN_CHUNK

// The pair words of a word of 4-bit codes: pair p's e0 at bits 4p, its e1 at bits 16 + 4p.
inline uint4 sp_gguf_nibble_pairs(uint word) { return (uint4(word) >> uint4(0u, 4u, 8u, 12u)) & 0x000F000Fu; }

// Bit fields into pair-word order. The 1-bit fields of two chunk bytes (bit 2p + i of byte b)
// go to bits 8b + 2p (e0) and 16 + 8b + 2p (e1); the 2-bit fields of a chunk's halfword (bits
// 4p + 2i) go to bits 4p (e0) and 16 + 4p (e1).
inline uint sp_gguf_spread1(uint bits) { return (bits & 0x5555u) | ((bits & 0xAAAAu) << 15); }
inline uint sp_gguf_spread2(uint bits) { return (bits & 0x3333u) | ((bits & 0xCCCCu) << 14); }

// The header of a Q4_K or Q5_K block: d and dmin in header.x, then twelve bytes holding eight
// 6-bit scales and eight 6-bit minimums. Group j has s = d * scale and m = -dmin * minimum.
// The bytes are taken with shifts: j is not known at compile time, and Splash measured
// indexing a thread-local array by it at 8% of its eight-row kernel.
inline SpGgufCoef sp_gguf_k4_coef(uint4 header, ushort j)
{
    uint scale, minimum;
    if (j < 4) {
        const uint shift = 8u * j;
        scale = (header.y >> shift) & 63u;
        minimum = (header.z >> shift) & 63u;
    } else {
        const uint shift = 8u * (j - 4), nibbles = header.w >> shift;
        scale = (nibbles & 0xFu) | ((((header.y >> shift) >> 6) & 3u) << 4);
        minimum = ((nibbles >> 4) & 0xFu) | ((((header.z >> shift) >> 6) & 3u) << 4);
    }
    const float d = float(as_type<half>(ushort(header.x & 0xFFFFu)));
    const float dmin = float(as_type<half>(ushort(header.x >> 16)));
    return {float2(d * float(scale)), float2(-dmin * float(minimum))};
}

// Q4_K. plane0: the 4-bit codes, pair-packed. meta: the block's sixteen header bytes.
struct SpGgufQ4K {
    SP_GGUF_FORMAT(16, 0, 16, 8, 0, SpGgufLinear, false);
    struct Payload { uint4 a; };
    typedef uint Chunk;
    typedef uint4 Meta;
    static Payload load(device const uchar *p0, device const uchar *) { return {uint4(*(device const packed_uint4 *)p0)}; }
    static Meta loadMeta(device const uchar *m) { return uint4(*(device const packed_uint4 *)m); }
    static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
    static uint4 codes(Chunk q) { return sp_gguf_nibble_pairs(q); }
    static SpGgufCoef coef(Meta header, ushort j) { return sp_gguf_k4_coef(header, j); }
};

// Q5_K. plane0: the low four bits of each code, pair-packed. plane1: the fifth bits, byte c
// being chunk c. meta: as Q4_K.
struct SpGgufQ5K {
    SP_GGUF_FORMAT(16, 4, 16, 8, 0, SpGgufLinear, false);
    struct Payload { uint4 a; uint b; };
    typedef uint2 Chunk;
    typedef uint4 Meta;
    static Payload load(device const uchar *p0, device const uchar *p1)
    {
        return {uint4(*(device const packed_uint4 *)p0), *(device const uint *)p1};
    }
    static Meta loadMeta(device const uchar *m) { return uint4(*(device const packed_uint4 *)m); }
    static Chunk chunk(Payload w, ushort c)
    {
        return uint2(w.a[c], sp_gguf_spread1(w.b >> (16 * (c >> 1))) >> (8 * (c & 1)));
    }
    static uint4 codes(Chunk q)
    {
        return sp_gguf_nibble_pairs(q.x) | (((uint4(q.y) >> uint4(0u, 2u, 4u, 6u)) & 0x00010001u) << 4);
    }
    static SpGgufCoef coef(Meta header, ushort j) { return sp_gguf_k4_coef(header, j); }
};

// Q6_K. plane0: the low four bits, pair-packed. plane1: the high two bits, halfword c being
// chunk c. meta: the block's sixteen signed 8-bit scales, then d and two zero bytes. A weight
// is d * scale * (code - 32), with a scale for each sixteen elements.
struct SpGgufQ6K {
    SP_GGUF_FORMAT(16, 8, 20, 8, 32, SpGgufLinear, false);
    struct Payload { uint4 a; uint2 b; };
    typedef uint2 Chunk;
    struct Meta { uint4 scales; uint d; };
    static Payload load(device const uchar *p0, device const uchar *p1)
    {
        return {uint4(*(device const packed_uint4 *)p0), uint2(*(device const packed_uint2 *)p1)};
    }
    static Meta loadMeta(device const uchar *m)
    {
        return {uint4(*(device const packed_uint4 *)m), *(device const uint *)(m + 16)};
    }
    static Chunk chunk(Payload w, ushort c)
    {
        return uint2(w.a[c], sp_gguf_spread2(w.b[c >> 1] >> (16 * (c & 1))));
    }
    static uint4 codes(Chunk q)
    {
        return sp_gguf_nibble_pairs(q.x) | (((uint4(q.y) >> uint4(0u, 4u, 8u, 12u)) & 0x00030003u) << 4);
    }
    static SpGgufCoef coef(Meta header, ushort j)
    {
        const float d = float(as_type<half>(ushort(header.d & 0xFFFFu)));
        // Scale bytes 2j and 2j + 1, by shifts as in sp_gguf_k4_coef.
        const uint word = j < 2 ? header.scales.x : j < 4 ? header.scales.y : j < 6 ? header.scales.z : header.scales.w;
        const uint pair = word >> (16u * (j & 1));
        return {float2(d * float(as_type<char>(uchar(pair & 0xFFu))), d * float(as_type<char>(uchar((pair >> 8) & 0xFFu))))};
    }
};

// Q3_K. plane0: the low two bits of each code, halfword c being chunk c. plane1: the high
// bits, byte c being chunk c. meta: d, two zero bytes, then the block's twelve bytes of sixteen
// 6-bit scales. A weight is d * (scale - 32) * (code - 4), with a scale for each sixteen
// elements.
struct SpGgufQ3K {
    SP_GGUF_FORMAT(8, 4, 16, 8, 4, SpGgufLinear, false);
    struct Payload { uint2 a; uint b; };
    typedef uint2 Chunk;
    typedef uint4 Meta;
    static Payload load(device const uchar *p0, device const uchar *p1)
    {
        return {uint2(*(device const packed_uint2 *)p0), *(device const uint *)p1};
    }
    static Meta loadMeta(device const uchar *m) { return uint4(*(device const packed_uint4 *)m); }
    static Chunk chunk(Payload w, ushort c)
    {
        return uint2(sp_gguf_spread2(w.a[c >> 1] >> (16 * (c & 1))),
                     sp_gguf_spread1(w.b >> (16 * (c >> 1))) >> (8 * (c & 1)));
    }
    static uint4 codes(Chunk q)
    {
        return ((uint4(q.x) >> uint4(0u, 4u, 8u, 12u)) & 0x00030003u)
            | (((uint4(q.y) >> uint4(0u, 2u, 4u, 6u)) & 0x00010001u) << 2);
    }
    static SpGgufCoef coef(Meta header, ushort j)
    {
        const float d = float(as_type<half>(ushort(header.x & 0xFFFFu)));
        // The scale bytes as three words. Scales 4w..4w+3 have their low four bits in the low
        // (w < 2) or high nibbles of word w & 1 and their top two bits at bits 2w of the bytes
        // of the third word; `scales` holds the four of w = j / 2, a byte each.
        const uint low0 = header.y, low1 = header.z, high = header.w;
        uint scales;
        switch (j >> 1) {
            case 0: scales = (low0 & 0x0F0F0F0Fu) | ((high & 0x03030303u) << 4); break;
            case 1: scales = (low1 & 0x0F0F0F0Fu) | (((high >> 2) & 0x03030303u) << 4); break;
            case 2: scales = ((low0 >> 4) & 0x0F0F0F0Fu) | (((high >> 4) & 0x03030303u) << 4); break;
            default: scales = ((low1 >> 4) & 0x0F0F0F0Fu) | (((high >> 6) & 0x03030303u) << 4); break;
        }
        const uint pair = scales >> (16u * (j & 1));
        return {float2(d * float(int(pair & 0xFFu) - 32), d * float(int((pair >> 8) & 0xFFu) - 32))};
    }
};

// Q8_0. plane0: the signed bytes, bytes 8c..8c+7 being chunk c. meta: d, a record a group.
struct SpGgufQ80 {
    SP_GGUF_FORMAT(32, 0, 2, 1, 0, SpGgufInt8, false);
    struct Payload { uint4 a; uint4 b; };
    typedef uint2 Chunk;
    typedef ushort Meta;
    typedef half Scale;
    static Payload load(device const uchar *p0, device const uchar *)
    {
        return {uint4(*(device const packed_uint4 *)p0), uint4(*(device const packed_uint4 *)(p0 + 16))};
    }
    static Meta loadMeta(device const uchar *m) { return *(device const ushort *)m; }
    static Chunk chunk(Payload w, ushort c)
    {
        const uint4 words = c < 2 ? w.a : w.b;
        return (c & 1) != 0 ? words.zw : words.xy;
    }
    static uint2 values(Chunk q) { return q; }
    static SpGgufCoef coef(Meta header, ushort) { return {float2(float(as_type<half>(header)))}; }
};

// IQ4_NL. plane0: the 4-bit codebook indices. meta: d, a record a group.
struct SpGgufIQ4NL {
    SP_GGUF_FORMAT(16, 0, 2, 1, 0, SpGgufCodebook, false);
    struct Payload { uint4 a; };
    typedef uint Chunk;
    typedef ushort Meta;
    typedef half Scale;
    static half value(uint i) { return half(sp_gguf_iq4nl_values[i]); }
    static Payload load(device const uchar *p0, device const uchar *) { return {uint4(*(device const packed_uint4 *)p0)}; }
    static Meta loadMeta(device const uchar *m) { return *(device const ushort *)m; }
    static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
    static uint indices(Chunk q) { return q; }
    static SpGgufCoef coef(Meta header, ushort) { return {float2(float(as_type<half>(header)))}; }
};

// IQ4_XS. plane0: the 4-bit codebook indices. meta: the block's eight header bytes: d, sixteen
// bits of high scale bits, four bytes of low scale nibbles. A weight is d * (scale - 32) *
// value, with a 6-bit scale a group.
struct SpGgufIQ4XS {
    SP_GGUF_FORMAT(16, 0, 8, 8, 0, SpGgufCodebook, false);
    struct Payload { uint4 a; };
    typedef uint Chunk;
    typedef uint2 Meta;
    typedef float Scale;
    static half value(uint i) { return half(sp_gguf_iq4nl_values[i]); }
    static Payload load(device const uchar *p0, device const uchar *) { return {uint4(*(device const packed_uint4 *)p0)}; }
    static Meta loadMeta(device const uchar *m) { return uint2(*(device const packed_uint2 *)m); }
    static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
    static uint indices(Chunk q) { return q; }
    static SpGgufCoef coef(Meta header, ushort j)
    {
        const float d = float(as_type<half>(ushort(header.x & 0xFFFFu)));
        const uint high = header.x >> 16;
        const int scale = int((header.y >> (4u * j)) & 0xFu) | int(((high >> (2u * j)) & 3u) << 4);
        return {float2(d * float(scale - 32))};
    }
};

// IQ3_S. plane0: word c is chunk c: the two grid indices, the chunk's sign bits and the
// group's 4-bit scale. meta: d, a record a block. A weight is d * (1 + 2 scale) * the signed
// grid magnitude.
struct SpGgufIQ3S {
    SP_GGUF_FORMAT(16, 0, 2, 8, 0, SpGgufGrid, true);
    struct Payload { uint4 a; };
    typedef uint Chunk;
    typedef ushort Meta;
    static Payload load(device const uchar *p0, device const uchar *) { return {uint4(*(device const packed_uint4 *)p0)}; }
    static Meta loadMeta(device const uchar *m) { return *(device const ushort *)m; }
    static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
    static uint2 grid(Chunk q)
    {
        return uint2(sp_gguf_iq3s_grid[(q & 0xFFu) | ((q >> 16) & 0x100u)],
                     sp_gguf_iq3s_grid[((q >> 8) & 0xFFu) | ((q >> 17) & 0x100u)]);
    }
    static uint signs(Chunk q) { return (q >> 16) & 0xFFu; }
    static SpGgufCoef coef(Meta header, Chunk q)
    {
        return {float2(float(as_type<half>(header)) * float(1u + 2u * ((q >> 26) & 0xFu)))};
    }
};

#undef SP_GGUF_FORMAT

// Fills a codebook format's pair table: entry b is the values of the two codes of the index
// byte b, low nibble first. Every thread of the threadgroup calls it, with its index and the
// threadgroup's size. Other formats have no table and this is nothing.
template <class F>
inline void sp_gguf_pair_table(threadgroup half2 *table, uint tid, uint threads)
{
    if constexpr (F::Kind == SpGgufCodebook) {
        for (uint i = tid; i < uint(F::PairEntries); i += threads) table[i] = half2(F::value(i & 15u), F::value(i >> 4));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One pair of linear codes as weights.
template <class F>
inline half2 sp_gguf_linear(uint pair, float s, float m)
{
    // 0x6400 is fp16 1024, whose unit in the last place is 1: a code or-ed into its mantissa
    // is 1024 + code, exactly.
    const float2 code = float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024 + F::Zero)));
    if constexpr (F::Zero != 0) return half2(code * s);
    else return half2(fma(code, float2(s), float2(m)));
}

// One row's group of 32 as fp16 weights in element order, each rounded once: `w` is the
// group's record, `header` its block's and `j` the group's place in the block.
template <class F>
inline void sp_gguf_dequant32(typename F::Payload w, typename F::Meta header, ushort j,
                              threadgroup half2 *table, threadgroup half *dst)
{
    SpGgufCoef k;
    if constexpr (F::ScaleInChunk) k = F::coef(header, F::chunk(w, 0));
    else k = F::coef(header, j);
#pragma unroll
    for (ushort c = 0; c < 4; ++c) {
        const typename F::Chunk q = F::chunk(w, c);
        half4 low, high;
        if constexpr (F::Kind == SpGgufLinear) {
            const uint4 pairs = F::codes(q);
            low = half4(sp_gguf_linear<F>(pairs.x, k.s.x, k.m.x), sp_gguf_linear<F>(pairs.y, k.s.x, k.m.x));
            high = half4(sp_gguf_linear<F>(pairs.z, k.s.y, k.m.y), sp_gguf_linear<F>(pairs.w, k.s.y, k.m.y));
        } else if constexpr (F::Kind == SpGgufCodebook) {
            typedef typename F::Scale S;
            const uchar4 index = as_type<uchar4>(F::indices(q));
            low = half4(half2(vec<S, 2>(table[index.x]) * S(k.s.x)), half2(vec<S, 2>(table[index.y]) * S(k.s.x)));
            high = half4(half2(vec<S, 2>(table[index.z]) * S(k.s.y)), half2(vec<S, 2>(table[index.w]) * S(k.s.y)));
        } else if constexpr (F::Kind == SpGgufInt8) {
            typedef typename F::Scale S;
            const uint2 bytes = F::values(q);
            low = half4(vec<S, 4>(as_type<char4>(bytes.x)) * S(k.s.x));
            high = half4(vec<S, 4>(as_type<char4>(bytes.y)) * S(k.s.y));
        } else {
            const uint2 magnitudes = F::grid(q);
            const uint negative = F::signs(q);
            low = half4(float4(as_type<uchar4>(magnitudes.x)) * k.s.x);
            high = half4(float4(as_type<uchar4>(magnitudes.y)) * k.s.y);
            low = select(low, -low, bool4((negative & 1u) != 0u, (negative & 2u) != 0u, (negative & 4u) != 0u, (negative & 8u) != 0u));
            high = select(high, -high, bool4((negative & 16u) != 0u, (negative & 32u) != 0u, (negative & 64u) != 0u, (negative & 128u) != 0u));
        }
        *(threadgroup packed_half4 *)(dst + 4 * c) = low;
        *(threadgroup packed_half4 *)(dst + 16 + 4 * c) = high;
    }
}

// What scales one row's group: `w` is the group's record, `header` its block's and `j` the
// group's place in the block.
template <class F>
inline SpGgufCoef sp_gguf_coef(typename F::Payload w, typename F::Meta header, ushort j)
{
    if constexpr (F::ScaleInChunk) return F::coef(header, F::chunk(w, 0));
    else return F::coef(header, j);
}

// Chunk c of a group as fp32 weights: elements 4c..4c+3 in `low`, 16+4c..16+4c+3 in `high`.
struct SpGgufChunk32 {
    float4 low;
    float4 high;
};

// The two codes of a pair word, less the format's offset.
template <class F>
inline float2 sp_gguf_pair_codes(uint pair)
{
    return float2(int2(int(pair & 0xFFFFu), int(pair >> 16)) - int(F::Zero));
}

// A chunk's weights as the reference decoder makes them (SploshModel/GgufQuant.swift), bit for
// bit. A scale is an fp16 times one or two small integers and a weight is that times a code of
// at most seven bits, so every product has at most 24 significant bits and is exact in fp32
// however it is grouped. The one sum, the offset of Q4_K and Q5_K, is of two exact terms and
// rounds once, fused or not. No table: the codebook is read where it lies.
template <class F>
inline SpGgufChunk32 sp_gguf_chunk32(typename F::Chunk q, SpGgufCoef k)
{
    SpGgufChunk32 w;
    if constexpr (F::Kind == SpGgufLinear) {
        const uint4 pairs = F::codes(q);
        w.low = float4(sp_gguf_pair_codes<F>(pairs.x), sp_gguf_pair_codes<F>(pairs.y)) * k.s.x;
        w.high = float4(sp_gguf_pair_codes<F>(pairs.z), sp_gguf_pair_codes<F>(pairs.w)) * k.s.y;
        // Only a format with no code offset has a weight offset. The others must not add one,
        // zero though it is: a negative zero would come out positive.
        if constexpr (F::Zero == 0) {
            w.low += k.m.x;
            w.high += k.m.y;
        }
    } else if constexpr (F::Kind == SpGgufCodebook) {
        const uint4 index = uint4(as_type<uchar4>(F::indices(q)));
        w.low = float4(float(F::value(index.x & 15u)), float(F::value(index.x >> 4)),
                       float(F::value(index.y & 15u)), float(F::value(index.y >> 4))) * k.s.x;
        w.high = float4(float(F::value(index.z & 15u)), float(F::value(index.z >> 4)),
                        float(F::value(index.w & 15u)), float(F::value(index.w >> 4))) * k.s.y;
    } else if constexpr (F::Kind == SpGgufInt8) {
        const uint2 bytes = F::values(q);
        w.low = float4(as_type<char4>(bytes.x)) * k.s.x;
        w.high = float4(as_type<char4>(bytes.y)) * k.s.y;
    } else {
        const uint2 magnitudes = F::grid(q);
        const uint negative = F::signs(q);
        w.low = float4(as_type<uchar4>(magnitudes.x)) * k.s.x;
        w.high = float4(as_type<uchar4>(magnitudes.y)) * k.s.y;
        w.low = select(w.low, -w.low, bool4((negative & 1u) != 0u, (negative & 2u) != 0u, (negative & 4u) != 0u, (negative & 8u) != 0u));
        w.high = select(w.high, -w.high, bool4((negative & 16u) != 0u, (negative & 32u) != 0u, (negative & 64u) != 0u, (negative & 128u) != 0u));
    }
    return w;
}

// Where one weight row's records start: its group 0 in the code planes and its block 0 in the
// meta plane. The row's later records are a tile of records further on apiece.
struct SpGgufRow {
    device const uchar *plane0;
    device const uchar *plane1;
    device const uchar *meta;
};

template <class F>
inline SpGgufRow sp_gguf_row(device const uchar *plane0, device const uchar *plane1, device const uchar *meta,
                             uint n, uint groups)
{
    const ulong tile = n / SP_GGUF_TILE_ROWS, inTile = n % SP_GGUF_TILE_ROWS;
    const ulong record = tile * groups * SP_GGUF_TILE_ROWS + inTile;
    const ulong block = tile * (groups / F::MetaGroups) * SP_GGUF_TILE_ROWS + inTile;
    return {plane0 + record * F::Plane0, plane1 + record * F::Plane1, meta + block * F::MetaBytes};
}

// One thread's place in a weight row: the record of a group, and the record of the native
// block that group is in. A kernel moves it on a matmul ahead of staging, so that the loads
// are in flight while the accelerator runs.
template <class F>
struct SpGgufCursor {
    SpGgufRow row;
    typename F::Payload codes;
    typename F::Meta scales;
    uint block;
};

template <class F>
inline typename F::Payload sp_gguf_codes(SpGgufRow row, uint g)
{
    return F::load(row.plane0 + ulong(g) * (SP_GGUF_TILE_ROWS * F::Plane0),
                   row.plane1 + ulong(g) * (SP_GGUF_TILE_ROWS * F::Plane1));
}

template <class F>
inline typename F::Meta sp_gguf_scales(SpGgufRow row, uint block)
{
    return F::loadMeta(row.meta + ulong(block) * (SP_GGUF_TILE_ROWS * F::MetaBytes));
}

template <class F>
inline SpGgufCursor<F> sp_gguf_seek(SpGgufRow row, uint g)
{
    const uint block = g / F::MetaGroups;
    return {row, sp_gguf_codes<F>(row, g), sp_gguf_scales<F>(row, block), block};
}

// To group g: its codes, and its block's scale fields when the block is another.
template <class F>
inline void sp_gguf_advance(thread SpGgufCursor<F> &cursor, uint g)
{
    cursor.codes = sp_gguf_codes<F>(cursor.row, g);
    const uint block = g / F::MetaGroups;
    if (block != cursor.block) {
        cursor.scales = sp_gguf_scales<F>(cursor.row, block);
        cursor.block = block;
    }
}

// The group's weights, to `dst`.
template <class F>
inline void sp_gguf_stage(SpGgufCursor<F> cursor, uint g, threadgroup half2 *table, threadgroup half *dst)
{
    sp_gguf_dequant32<F>(cursor.codes, cursor.scales, ushort(g % F::MetaGroups), table, dst);
}
