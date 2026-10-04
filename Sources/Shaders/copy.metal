#include <metal_stdlib>
#include "common/abi.h"

using namespace metal;

// copy — the M0 copy kernel (rev4 M0.7).
//
// The ABI is fixed, not incidental: M5.5's durable-cache staging copy reuses this exact binding
// set, so changing it is a spec change rather than a refactor.
//
//   [[buffer(0)]] src      device const uchar*
//   [[buffer(1)]] dst      device uchar*
//   [[buffer(2)]] params   constant CopyParams { u32 byteCount }
//
// Dispatch: threadgroup 256; threadgroups per grid = ceil(byteCount / (256 * 16)); each thread
// moves up to 16 bytes. Byte-granular, with no alignment requirement on either pointer or on the
// length, so an unaligned tail is the normal case rather than an error path.
kernel void copy(device const uchar  *src      [[buffer(0)]],
                 device uchar        *dst      [[buffer(1)]],
                 constant CopyParams &params   [[buffer(2)]],
                 uint                 gid      [[thread_position_in_grid]])
{
    const uint byteCount = params.byteCount;
    const uint base = gid * 16u;
    if (base >= byteCount) {
        return;
    }
    const uint end = min(base + 16u, byteCount);
    for (uint i = base; i < end; ++i) {
        dst[i] = src[i];
    }
}
