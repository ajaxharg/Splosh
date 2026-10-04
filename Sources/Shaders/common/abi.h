#pragma once

// Sources/Shaders/common/abi.h — shared Metal parameter structs.
//
// Header only, deliberately: every `.metal` source under Sources/Shaders is linked into one
// `default.metallib`, so every function name must be unique across the whole set (rev4 §3.3).
// A header contributes no function name and cannot collide; a `.metal` file carrying definitions
// would. `copy.metal` is the only consumer at M0; later milestones add their structs alongside.
//
// `common/abi.h` and `common/quant.metal` must define no kernel, so neither contributes a name to
// the metallib and neither may appear in an exact-set assertion (rev4 §3.2).

#include <metal_stdlib>

// CopyParams — `copy.metal`'s third buffer binding (rev4 M0.7).
//
// The byte count is the only quantity the kernel needs: the copy is byte-granular and carries no
// alignment precondition, so the dispatch geometry holds all of the shape information.
struct CopyParams {
    uint byteCount;
};

static_assert(sizeof(CopyParams) == 4, "CopyParams is one u32; the Swift side binds 4 bytes");
