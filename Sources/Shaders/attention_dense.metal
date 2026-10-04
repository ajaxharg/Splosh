#include <metal_stdlib>
using namespace metal;

// Must match AttentionDense.Parameters byte-for-byte (8 uint32 + 2 float32 = 40 bytes).
struct AttentionDenseParams {
    uint headDim;
    uint heads;
    uint kvHeads;
    uint rowCount;
    uint tokenCount;
    uint queryPosition;
    uint startPosition;
    uint rotaryDim;
    float epsilon;
    float ropeTheta;
};

inline float dense_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

inline float dense_rope_position(uint slot, uint position) {
    // Ordinary text attention supplies T-only position IDs.  The mRoPE oracle's
    // H/W sections consequently leave every slot at the T position.
    return float(position);
}

inline float dense_rotated(device const float *x, uint i, uint d, uint rotary,
                           uint position, float theta, bool upper) {
    const uint mid = rotary / 2u;
    const uint slot = upper ? i - mid : i;
    const float exponent = (2.0f * float(slot)) / float(rotary);
    const float angle = dense_rope_position(slot, position) / pow(theta, exponent);
    const float c = cos(angle);
    const float s = sin(angle);
    const float paired = x[upper ? i - mid : i + mid];
    return x[i] * c + (upper ? paired : -paired) * s;
}

inline float dense_norm(device const float *x, device const float *weight,
                        uint base, uint i, uint d, float epsilon) {
    float sum = 0.0f;
    for (uint j = 0; j < d; ++j) {
        const float v = x[base + j];
        sum += v * v;
    }
    const float inv = rsqrt(sum / float(d) + epsilon);
    return x[base + i] * inv * (1.0f + weight[i]);
}

inline float dense_key(device const char *keys, device const float *scales,
                       uint token, uint kvHead, uint i, uint kh, uint d) {
    const uint index = (token * kh + kvHead) * d + i;
    return float(keys[index]) * scales[kvHead];
}

inline float dense_key_norm_rope(device const char *keys, device const float *scales,
                                 device const float *weight, uint token, uint kvHead,
                                 uint i, uint kh, uint d, uint rotary, uint position,
                                 float epsilon, float theta) {
    float sum = 0.0f;
    for (uint j = 0; j < d; ++j) {
        const float v = dense_key(keys, scales, token, kvHead, j, kh, d);
        sum += v * v;
    }
    const float inv = rsqrt(sum / float(d) + epsilon);
    const uint halfWidth = rotary / 2u;
    const float raw = dense_key(keys, scales, token, kvHead, i, kh, d) * inv * (1.0f + weight[i]);
    if (i >= rotary) return raw;
    const uint slot = i < halfWidth ? i : i - halfWidth;
    const float angle = float(position) / pow(theta, (2.0f * float(slot)) / float(rotary));
    const float c = cos(angle), s = sin(angle);
    const uint pairedIndex = i < halfWidth ? i + halfWidth : i - halfWidth;
    const float paired = dense_key(keys, scales, token, kvHead, pairedIndex, kh, d) * inv * (1.0f + weight[pairedIndex]);
    return raw * c + (i < halfWidth ? -paired : paired) * s;
}

inline float dense_query(device const float *q, device const float *normWeight,
                         uint base, uint i, uint d, uint rotary, uint position,
                         float epsilon, float theta) {
    const float raw = dense_norm(q, normWeight, base, i, d, epsilon);
    if (i >= rotary) return raw;
    const uint halfWidth = rotary / 2u;
    const uint slot = i < halfWidth ? i : i - halfWidth;
    const float angle = float(position) / pow(theta, (2.0f * float(slot)) / float(rotary));
    const float c = cos(angle), s = sin(angle);
    const uint pairedIndex = i < halfWidth ? i + halfWidth : i - halfWidth;
    const float paired = dense_norm(q, normWeight, base, pairedIndex, d, epsilon);
    return raw * c + (i < halfWidth ? -paired : paired) * s;
}

inline void dense_attention_row(device const float *qProjection,
                                device const float *qNorm, device const float *kNorm,
                                device const char *keys, device const char *values,
                                device const float *kScales, device const float *vScales,
                                device float *mixed, device const float *oProjection,
                                device float *output, constant AttentionDenseParams &p,
                                uint row) {
    const uint d = p.headDim, h = p.heads, kh = p.kvHeads, width = h * d;
    if (width > 8192u) return;
    float localMixed[8192];
    if (d == 0u || h == 0u || kh == 0u || h % kh != 0u || p.tokenCount == 0u ||
        p.rotaryDim == 0u || p.rotaryDim > d || (p.rotaryDim & 1u) != 0u ||
        !isfinite(p.epsilon) || p.epsilon <= 0.0f || !isfinite(p.ropeTheta) || p.ropeTheta <= 0.0f ||
        row >= p.rowCount) return;
    const uint queryPosition = p.startPosition + row;
    for (uint head = 0; head < h; ++head) {
        const uint qBase = row * (h * 2u * d) + head * 2u * d;
        const uint kvHead = head / (h / kh);
        float maxScore = -INFINITY;
        for (uint t = 0; t < p.tokenCount; ++t) {
            if (t > queryPosition) continue;
            float score = 0.0f;
            for (uint j = 0; j < d; ++j) {
                const float qv = dense_query(qProjection, qNorm, qBase, j, d, p.rotaryDim, queryPosition, p.epsilon, p.ropeTheta);
                const float kv = dense_key_norm_rope(keys, kScales, kNorm, t, kvHead, j, kh, d, p.rotaryDim, t, p.epsilon, p.ropeTheta);
                score += qv * kv;
            }
            maxScore = max(maxScore, score * rsqrt(float(d)));
        }
        float denom = 0.0f;
        for (uint t = 0; t < p.tokenCount; ++t) {
            if (t > queryPosition) continue;
            float score = 0.0f;
            for (uint k = 0; k < d; ++k) {
                const float qv = dense_query(qProjection, qNorm, qBase, k, d, p.rotaryDim, queryPosition, p.epsilon, p.ropeTheta);
                const float kv = dense_key_norm_rope(keys, kScales, kNorm, t, kvHead, k, kh, d, p.rotaryDim, t, p.epsilon, p.ropeTheta);
                score += qv * kv;
            }
            denom += exp(score * rsqrt(float(d)) - maxScore);
        }
        for (uint j = 0; j < d; ++j) {
            float numer = 0.0f;
            for (uint t = 0; t < p.tokenCount; ++t) {
                if (t > queryPosition) continue;
                float score = 0.0f;
                for (uint k = 0; k < d; ++k) {
                    const float qv = dense_query(qProjection, qNorm, qBase, k, d, p.rotaryDim, queryPosition, p.epsilon, p.ropeTheta);
                    const float kv = dense_key_norm_rope(keys, kScales, kNorm, t, kvHead, k, kh, d, p.rotaryDim, t, p.epsilon, p.ropeTheta);
                    score += qv * kv;
                }
                const float prob = exp(score * rsqrt(float(d)) - maxScore);
                numer += prob * float(values[(t * kh + kvHead) * d + j]) * vScales[kvHead];
            }
            const float gate = dense_sigmoid(qProjection[qBase + d + j]);
            localMixed[head * d + j] = (denom > 0.0f && isfinite(denom) ? numer / denom : 0.0f) * gate;
        }
    }
    const uint outBase = row * width;
    for (uint i = 0; i < width; ++i) {
        float result = localMixed[i];
        if (oProjection != nullptr) {
            result = 0.0f;
            for (uint j = 0; j < width; ++j) result += oProjection[i * width + j] * localMixed[j];
        }
        output[outBase + i] = result;
    }
}

kernel void attention_dense_decode(
    device const float *qProjection [[buffer(0)]], device const float *qNorm [[buffer(1)]],
    device const float *kNorm [[buffer(2)]], device const char *keys [[buffer(3)]],
    device const char *values [[buffer(4)]], device const float *kScales [[buffer(5)]],
    device const float *vScales [[buffer(6)]], device const float *oProjection [[buffer(7)]],
    device float *output [[buffer(8)]], constant AttentionDenseParams &p [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0u || p.rowCount == 0u) return;
    dense_attention_row(qProjection, qNorm, kNorm, keys, values, kScales, vScales,
                        output, oProjection, output, p, 0u);
}

kernel void attention_dense_prefill(
    device const float *qProjection [[buffer(0)]], device const float *qNorm [[buffer(1)]],
    device const float *kNorm [[buffer(2)]], device const char *keys [[buffer(3)]],
    device const char *values [[buffer(4)]], device const float *kScales [[buffer(5)]],
    device const float *vScales [[buffer(6)]], device const float *oProjection [[buffer(7)]],
    device float *output [[buffer(8)]], constant AttentionDenseParams &p [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.heads * p.headDim || gid.y >= p.rowCount) return;
    // One thread per row/channel is the public geometry; only lane zero performs
    // the row's reduction, while all lanes remain explicitly bounds checked.
    if (gid.x != 0u) return;
    dense_attention_row(qProjection, qNorm, kNorm, keys, values, kScales, vScales,
                        output, oProjection,
                        output, p, gid.y);
}
