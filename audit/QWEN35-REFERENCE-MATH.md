# Qwen3.5 reference math — extracted from the shipping implementation

**Source:** `huggingface/transformers@main`
`src/transformers/models/qwen3_5/modeling_qwen3_5.py` (88,586 bytes, fetched 2026-10-01).
Retrieved over the network; this file records the formulas the Splosh kernels must match.

This replaces guesswork. Every formula below is quoted from that file.

## 1. Norms — two different forms

### `Qwen3_5RMSNorm` (input_layernorm, post_attention_layernorm, final norm, q_norm, k_norm)

```python
weight = nn.Parameter(torch.zeros(dim))          # zero-initialised
output = x * rsqrt(x.pow(2).mean(-1) + eps)
output = output * (1.0 + weight.float())         # ← (1 + w), NOT w
```

`(1 + weight)` is correct. `Sources/Shaders/rmsnorm.metal` already matches.

### `Qwen3_5RMSNormGated` (GDN `linear_attn.norm`, 128-dim per value head)

```python
weight = nn.Parameter(torch.ones(hidden_size))   # one-initialised
variance = x.pow(2).mean(-1, keepdim=True)
x = x * rsqrt(variance + eps)
x = weight * x                                   # ← plain w, NOT (1 + w)
x = x * silu(gate)                               # norm BEFORE gate
```

**Two divergences from the current `gdn_gate.metal`:** it uses `(1 + weight)` where the
reference uses plain `weight`. Order (norm then gate) is already correct.

## 2. `l2norm` (GDN q/k only)

```python
inv_norm = rsqrt((x * x).sum(dim=-1) + eps)      # SUM, not mean; eps = 1e-6
return x * inv_norm
```

## 3. Gated DeltaNet (`linear_attention`, 48 of 64 layers)

Dimensions for this checkpoint: `num_k_heads=16`, `num_v_heads=48`, `head_k_dim=128`,
`head_v_dim=128`, `key_dim=2048`, `value_dim=6144`, `conv_dim = 2*2048+6144 = 10240`,
`conv_kernel_size=4`, conv is depthwise (`groups=conv_dim`) and **has no bias**.

```python
mixed_qkv = in_proj_qkv(x)                       # [T, 10240]
mixed_qkv = causal_conv1d(mixed_qkv)             # depthwise, k=4, left-pad 3
mixed_qkv = silu(mixed_qkv)                      # activation = hidden_act = "silu"
q, k, v = split(mixed_qkv, [2048, 2048, 6144])

z = in_proj_z(x)                                 # [T, 6144] — from x, NOT from conv output
b = in_proj_b(x)                                 # [T, 48]
a = in_proj_a(x)                                 # [T, 48]

beta = sigmoid(b)
g    = -exp(A_log) * softplus(a + dt_bias)       # log-space decay, <= 0

# v-head h reads k-head h // 3  (repeat_interleave by num_v_heads//num_k_heads = 3)
q = l2norm(q); k = l2norm(k)
q = q / sqrt(head_k_dim)                         # ALWAYS, = 1/sqrt(128)

# recurrence, state S is [48, 128(key), 128(value)]
S       = S * exp(g)
kv_mem  = sum_over_key(S * k)                    # [v_dim]
delta   = (v - kv_mem) * beta
S       = S + outer(k, delta)
o       = sum_over_key(S * q)                    # [v_dim]

o   = RMSNormGated(o, z)                         # per 128-dim head, §1
out = out_proj(o)                                # [6144] -> [5120]
```

`z`, `a` and `b` are projected from the **pre-convolution** hidden states.

Note the state carries `exp(g)` where `g <= 0`, so decay is in `(0, 1]`.

## 4. Full attention (`full_attention`, 16 of 64 layers)

`head_dim=256`, `num_attention_heads=24`, `num_key_value_heads=4`, groups = 6,
`scaling = head_dim ** -0.5 = 1/16`, `attention_bias = false`.

```python
# q_proj emits num_heads * head_dim * 2 = 12288; per head the 512 channels are [query | gate]
query, gate = chunk(q_proj(x).view(..., -1, head_dim * 2), 2, dim=-1)

query = q_norm(query)                            # RMSNorm over head_dim, (1 + w)
key   = k_norm(k_proj(x).view(..., -1, head_dim))
value = v_proj(x).view(..., -1, head_dim)

query, key = apply_rotary_pos_emb(query, key, cos, sin)   # after the norms

attn = softmax(query @ key.T * scaling + causal_mask) @ value
attn = attn * sigmoid(gate)                      # gate applied before o_proj
out  = o_proj(attn)                              # [6144] -> [5120]
```

## 5. RoPE

```python
dim      = head_dim * partial_rotary_factor = 256 * 0.25 = 64
inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2) / dim))   # 32 entries, theta = 1e7
cos      = cat(freqs, freqs)                                  # 64 entries, cos[j] == cos[j+32]
attention_scaling = 1.0                                       # "default" rope type
```

`rotate_half`, NeoX style, applied to the **first 64 channels only**; channels 64..255
pass through unchanged:

```python
x1, x2 = x_rot[..., :32], x_rot[..., 32:]
rotate_half(x_rot) = cat(-x2, x1)
out = x_rot * cos + rotate_half(x_rot) * sin
```

### mRoPE collapses for text

`mrope_section = [11, 11, 10]`, `mrope_interleaved = true`. The three grids (T, H, W) are
recomposed by `recomposition_frequencies`, but `Qwen3_5TextModel` builds text position ids as
`arange(seq) + past_seen_tokens` broadcast to all grids. **For text-only serving all three
position ids are equal, so mRoPE reduces exactly to standard RoPE.** Splosh serves text only;
this equivalence is why a single position scalar is sufficient, and it is the reason the
vision tower's 333 tensors are not loaded.

## 6. Block and model structure

```python
# Qwen3_5DecoderLayer
residual = x
x = mixer(input_layernorm(x))                    # linear_attn or self_attn
x = residual + x
residual = x
x = mlp(post_attention_layernorm(x))
x = residual + x

# Qwen3_5MLP
mlp(x) = down_proj(silu(gate_proj(x)) * up_proj(x))

# Qwen3_5TextModel
x = embed_tokens(input_ids)                      # plain lookup, NO scaling
... 64 layers ...
x = norm(x)                                      # RMSNorm, (1 + w)
logits = lm_head(x)                              # untied (tie_word_embeddings = false)
```

## 7. Checkpoint facts that constrain the loader

Tensor prefix is `language_model.model.layers.N.` — **not** `model.layers.N.`.

| Tensor | dtype | physical shape | logical shape |
|---|---|---|---|
| `embed_tokens.weight` | U32 | `[248320, 640]` | `[248320, 5120]` q4 |
| `lm_head.weight` | U32 | `[248320, 640]` | `[248320, 5120]` q4 |
| `linear_attn.in_proj_qkv.weight` | U32 | `[10240, 640]` | `[10240, 5120]` |
| `linear_attn.in_proj_z.weight` | U32 | `[6144, 640]` | `[6144, 5120]` |
| `linear_attn.in_proj_a/b.weight` | U32 | `[48, 640]` | `[48, 5120]` |
| `linear_attn.out_proj.weight` | U32 | `[5120, 768]` | `[5120, 6144]` |
| `linear_attn.conv1d.weight` | BF16 | `[10240, 4, 1]` | — |
| `linear_attn.A_log`, `dt_bias` | BF16 | `[48]` | — |
| `linear_attn.norm.weight` | BF16 | `[128]` | — |
| `self_attn.q_proj.weight` | U32 | `[12288, 640]` | `[12288, 5120]` |
| `self_attn.k_proj/v_proj.weight` | U32 | `[1024, 640]` | `[1024, 5120]` |
| `self_attn.o_proj.weight` | U32 | `[5120, 768]` | `[5120, 6144]` |
| `self_attn.q_norm/k_norm.weight` | BF16 | `[256]` | — |
| `mlp.gate_proj/up_proj.weight` | U32 | `[17408, 640]` | `[17408, 5120]` |
| `mlp.down_proj.weight` | U32 | `[5120, 2176]` | `[5120, 17408]` |
| `input_layernorm`, `post_attention_layernorm` | BF16 | `[5120]` | — |
| `model.norm.weight` | BF16 | `[5120]` | — |

Every q4 weight is MLX affine, group size 64, 4 bits: a U32 word holds eight logical
columns low-nibble-first, and `scales`/`biases` are BF16 `[rows, logicalCols/64]`.
Dequantised value for logical column `c` of row `r`:

```
group  = c / 64
nibble = (packed[r * strideWords + c / 8] >> ((c % 8) * 4)) & 0xF
value  = scales[r * groupsPerRow + group] * nibble + biases[r * groupsPerRow + group]
```

`vocab_size` is 248320 but `tokenizer.json` defines only 248044 vocab entries plus 33
added tokens (max id 248076). Ids 248077..248319 are unused padding rows in `lm_head`
and must be masked out before sampling.

## 8. The MLX artifact bakes `+1` into the norm weights

`mlx_lm/models/qwen3_5.py` `sanitize` rewrites `input_layernorm`, `post_attention_layernorm`,
`model.norm`, `q_norm` and `k_norm` as `v + 1.0` at conversion time and then evaluates them with
a plain `nn.RMSNorm` (`w * x * rsqrt(mean(x^2) + eps)`). The q4 pack Splosh serves is that
converted form: observed stored values are centred near 1 (for example layer 0
`input_layernorm` begins `1.0469, 0.9375, 0.9258`), not near 0.

**Every norm in the Splosh engine therefore applies the stored weight directly.** Applying
`(1 + w)` to this artifact, as §1 describes for the un-converted checkpoint, doubles the offset
and produces garbage tokens. The older `rmsnorm.metal` oracle kernel encodes the un-converted
form and must not be used against this artifact.
