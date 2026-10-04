# Splosh engine — performance research log

Measured on an Apple M5 Pro (20-core GPU, 51.84 GiB Metal working set), macOS 27.0, with the
Qwen3.8-27B MLX 4-bit pack (14.09 GiB of text tensors). Dates: 2026-10-01 and 2026-10-02.

Every number here was observed on this machine. Companion document:
`QWEN35-REFERENCE-MATH.md` (the model's formulas and tensor layout).

## 0. Read this first: how to measure on this machine

Timings move by 20–35% with the machine's state, which is more than most of the effects below.

- **Burst versus sustained.** The GPU runs about a third faster for the first second after being
  idle. Forty consecutive 128-row steps: 307 ms for the first, then a steady 380–390 ms. A
  five-iteration benchmark in a fresh process measures the burst; a real 20K-token prefill runs
  at the sustained rate. `SPLOSH_CTXBENCH_ITERATIONS` above 5 drops the first four iterations.
- **Heat soak.** After minutes of load the sustained rate itself drops again (the same step
  measured 288 ms, 343 ms and 385 ms at different times of one night). The Mac is in the
  "Automatic" energy mode; High Power mode was not tried (a system setting).
- **Other engines.** A second engine loaded in the same Metal working set, even idle, and the
  system's photo-analysis daemon both showed up as slower runs.
- **So:** compare alternatives interleaved, several rounds, and keep the best or the median
  of each. `tools/bench/step-bench` does this for engine steps (burst or sustained) and
  `tools/bench/server-ab` for two running servers. Single runs are not evidence.

## 1. Hardware limits

| Probe | Result |
|---|---|
| GPU read bandwidth (coalesced 16-byte loads, ordinary arithmetic) | ~290 GB/s |
| One pass over all weights at that rate | ~52 ms |
| Concurrent dispatch, no barrier | ~0.4 µs |
| Dependent stage inside a real step (measured by adding empty ones) | 40–60 µs |
| One encoder per dispatch / one command buffer per dispatch | ~38 µs / ~67 µs |
| Register-only FMA: dependent chain / eight independent chains | 5.7 / 15 T mul-add/s |
| `simdgroup_matrix` 8x8 multiply-accumulate | 5.4 T mul-add/s |
| Accelerator, q4 tile 32 x 128 x K 64 (bf16 x uint4b), no memory traffic | ~14 T mul-add/s |
| Accelerator, int8 attention tile 64 x 64 x K 256 (half x int8) | ~17 T mul-add/s burst, ~11 sustained |
| Cold start (page faults, pipeline warm-up) | ~1.4 s |

**Resources have to be kept resident, or the first step after an idle spell waits for them.**
With a step's command buffers committed within 4 ms, the GPU started the first of them 1.0-1.5 s
later, and the next two after further gaps, when the server had idled for 25 s: 1.5-2.3 s lost
on the first 128-row step of a request, 0.27 s on a narrow one, 0.3 s after only 2 s of idling,
nothing on the steps that follow (`SPLOSH_STEP_TIMING`). The driver makes a command buffer's
resources resident when it is scheduled. An `MTLResidencySet` holding the weights, scratch and
state, attached to the queue, cuts that to 0.28 s after 25 s and 0.09 s after 2 s. The KV pool
stays out of the set (residency commits all of it, and the waits were longer with it in);
what remains scales with the pool, 0.16 s with a 2 GiB pool. On the server this was the
difference between 300 and 425-445 tok/s for a 1.5K prompt arriving after a pause, which is
every turn of an agent that has been running a tool.

Consequences:

- One token per step cannot beat ~19 tok/s. Anything faster needs more tokens per step
  (speculative decoding) or more sessions per step (batching).
- The 128-row prefill GEMM is **bound by the accelerator's rate**, not by memory: with every
  tile aliased to one block of weights the step is only 9 ms faster (of ~280).
- With no stage boundaries, every GEMM in the model for 1, 8 or 16 rows takes the same
  ~51.5 ms: the weight stream. 32 rows take 62 ms, 64 rows 120 ms, 128 rows ~245 ms (1.9 ms a
  row, against 1.76 ms at the accelerator's rate). An eight-row verify step is 60.5 ms, so it
  is within 9 ms of one pass over the weights.
- **Ordinary arithmetic and accelerator work do not overlap.** A 26 ms block of FMA work
  dispatched in the same stage as the down-projection GEMM (no dependency between them) added
  22 ms to the step; in a stage of its own it added 27 ms. So the non-GEMM share of a step
  cannot be hidden behind the GEMMs by scheduling: total time is close to the sum of the work.
- Stages matter more than first assumed. An eight-row step has ~420 dependent stages; at
  40–60 µs each that is a quarter of the step. Removing the two normalisation stages per layer
  saved 6 ms of 66.

## 2. What the accelerators want

Metal Performance Primitives `matmul2d` runs on the M5 GPU's per-core neural accelerators.
Learned the hard way:

- **Row granule of 16.** An 8-row tile runs at about a quarter of the rate of a 16-row tile.
  An eight-row verify step is therefore faster through a 16 x 64 tile with eight unused rows
  (60 GEMM-ms) than through an 8 x 32 tile (80 ms).
- **Per-simdgroup tile of 32 x 32 outputs.** 32 x 128 on four simdgroups, 32 x 256 on eight and
  32 x 64 on two all give the same rate; 64 or 16 columns per simdgroup are two to three times
  slower.
- **Query tiles are padded to a multiple of 32.** A 48-query attention tile costs what a
  64-query one does. 64 real queries were slower anyway (see register pressure), and 32-query
  tiles gave the same total time, so the padding is left alone.
- **Register pressure.** More than about 48 accumulators per thread is slow: a 96 x 256 or
  64 x 256 output tensor held across the attention loop cost 1.5–3x.
- **Two weight streams in one simdgroup are slow.** A fused gate/up/SiLU kernel, tried on 8- and
  16-row tiles, was 30 ms slower per verify step than two GEMMs and a SiLU stage.
- **The destination tensor's partition is regular and usable.** For the 32 x 128 tile every
  thread owns columns `c0..c0+3` and `c0+64..c0+67` of rows `rb, rb+8, rb+16, rb+24`
  (`rb = ((lane >> 1) & 3) + 4 * (lane >> 4)`, column offset from lane bits 0 and 3); a
  single-simdgroup tile has the same shape with the second column run at `N/2`. Knowing that,
  the q4 epilogue is four vector loads and four sums per quant group instead of three scalar
  loads per element: 75 ms -> 15 ms per 128-row step. The layout is checked per thread at run
  time and falls back to (or, where it must not be silently wrong, poisons) the generic path.
- **Unused rows do not influence used ones** (`SPLOSH_PADCHECK`): outputs are bit-identical
  whether a tile's spare rows, or an attention tile's spare queries, hold 0 or 10,000.
- Cooperative tensors of different operand types do not share an element partition.
  `relaxed_precision` and fp16 instead of bf16 activations made no measurable difference.

Supported operand pairs worth knowing (from the SDK header): `bfloat/half x uint4b/int4b`,
`bfloat/half/float x int8/uint8`, `int8 x int8 -> int32`, `bfloat x bfloat`, and
`half x fp8_e4m3`. A symmetric `int4b` or an `fp8` KV cache would need no scale epilogue at all;
neither has been tried.

## 3. Weight layout and the q4 GEMM

MLX stores each q4 matrix row-major. The engine runs from a tiled layout
(`splosh convert --retile`, one resident copy):

```
packed:  [tile of 128 rows][quant group][row in tile][32 bytes of codes]
sidecar: [tile][quant group][row in tile]        (scales and biases, bf16)
```

Kernel by row count, as shipped:

| Rows | Kernel |
|---|---|
| 1–2 | split-K lane kernel, ordinary arithmetic (bandwidth-bound, ~51 ms) |
| 3–16 | split-K accelerator, 16 x 64 tile, four partitions |
| 17–64 | split-K accelerator, 32 x 32 tile, four partitions |
| 65+ | 32 x 128 accelerator tile on four simdgroups |

The q4 format costs an epilogue per (output, quant group): `scale * (a . codes) + bias * sum(a)`.
That cannot be made into a longer accelerator run, because the scale depends on both output
and group.

**What the kernel carries matters, not only what it runs.** Splash's prefill tile (Apache-2.0,
`runtime/metal/kernels/prefill/linear_q4.metal`), transcribed into this harness as a probe,
ran the 128-row GEMMs in 220 ms where `sp_na_tiled` took 238, with the same tile shape and
a plainer per-group loop. Bisecting the difference: not its bf16 output (221 with fp32), not
its sums layout (220 with this engine's), not its loop (220 with this engine's vector-load
loop dropped in). `sp_na_tiled` with its finished tile stored by the accelerator instead of
written from the threads: still 239. A new kernel with nothing but the loop, the residual and
the store: 221-224. The same kernel with the residual behind a run-time flag and a fallback
for partial tiles: 290. So whole-tile steps run `sp_na_wide`, in a plain and a residual
variant with nothing optional at run time, and take an ordinary norm stage instead of the
operand-emitting GEMM (which is the bigger kernel): 263 ms a 128-row step against 286. The
same treatment of the gated-delta kernel (journal and debug paths compiled out) changed
nothing, so this is not a general law; it is worth a probe wherever a kernel has grown.

## 4. Stages: normalisation without a stage

`RMSNorm(h) = h * w * inv(row)`, and a GEMM is linear in its input. So the residual projections
(mixer output, MLP down) also emit the operand of the GEMMs behind the next norm —
`bf16(h * w)` with its per-64 sums — plus per-64 sums of squares. The per-row scale is computed
from those alongside the GEMMs, and applied by whatever consumes the GEMM outputs (gated-delta
kernel, attention prepare/store/merge, the SiLU product). SiLU likewise emits the
down-projection's operand directly.

Stages per layer in an eight-row step: 9 -> 6 (gated-delta) and 8 (full attention).
Eight rows: 66.5 -> 60.5 ms. Teacher-forced loss unchanged. Not applied to 17–64 rows, whose
32-column tile cannot emit per-64 sums.

## 5. Gated-delta layers

One kernel, three phases in a 1024-thread threadgroup:

- A: conv + SiLU for q, k, v and the decay/beta terms for every row of the run at once (the
  conv has no recurrence — its history is earlier rows' raw input); L2 norms per row.
- B: the recurrence, state in registers. Simdgroup `s` owns state columns `4s..4s+3`; a lane
  holds 16 rows of one column; a sum over a column's eight parts is three xor-shuffles. **No
  barrier per token.**
- C: gated RMSNorm, the bf16 operand and its sums for every row at once.

About 67 ms -> 15–20 ms per 128-row step. (An earlier "89 ms" figure included a GEMM that the
skip switch also skipped.)

Speculative rows do not write states. A state is 150 MB per row across the model; a journal
entry — k, delta and decay per head, the ingredients of the row's rank-1 update — is 2.4 MB.
Accepted entries are replayed into the slot's single state at its next step. The replay and
the inline update share one explicitly fused expression so the replay is bit-identical.
Verify-shaped steps: 8 rows 70 -> 67 ms, 64 rows 184 -> 163 ms.

## 6. KV cache and attention

**Format.** Symmetric int8 with one scale per K or V vector (32.5 KiB per token; 1.6 GiB at
50K, 4.8 GiB at 150K). The 4-bit formats were tried first and are the wrong trade:

| KV format | Mean KL vs fp16 | Top-1 agreement | Notes |
|---|---|---|---|
| int8, one scale per vector | 0.0001 | 511/512 | shipped |
| q4, scale and bias per 64 values | 0.003–0.008 | 502–504/512 | needs a bias term per score |
| q4, one scale and bias per vector | — | — | failed the speculation self-test; removed |

(Teacher-forced, ~3.5K tokens each of prose and code, `SPLOSH_KVQUALITY`.) With no bias term a
score is one 256-wide accelerator matmul and one multiply, and the value scale folds into the
probability operand.

**Kernel.** One fused pass per 64-token chunk in a threadgroup that owns (KV head, 8 query rows
= 48 queries, span of history): Q x K on the accelerator, softmax in registers, P x V
accumulated in place. Nothing is written to device memory between phases. One threadgroup
barrier per chunk instead of three: the softmax reference is the row maximum of the span's
first chunk, moved on a rare path if a later score exceeds it by more than 30, and
probabilities alternate between two buffers so the next chunk can start while other
simdgroups are still reading the last. Probabilities are bf16, not fp16: at long context a
diffuse row's `p * scale` falls into fp16's subnormal range.

128-row prefill step / 8-row verify step, burst:

| Context | q4 three-pass (old) | int8 fused (now) |
|---|---|---|
| 1K | 406 / 97 ms | 285 / 60 ms |
| 16K | 571 / 113 ms | 329 / 65 ms |
| 50K | 986 / 148 ms | 420 / 73 ms |

Sustained, the same steps are 337 / 65 ms at 16K and 461 / 74 ms at 50K. Attention costs about
2.8 ms per 1K tokens of context per 128 rows (burst), 0.25 ms per 1K for an eight-row block. Span count (2–256) made no difference worth having.

**Where the time goes at 100K context** (128-row prefill step, sustained, 2026-10-02): of a
600 ms step about 320 ms is attention; the score product is ~135 ms of that, the value product
~173 ms (254 ms together), and the softmax, staging and barriers ~68 ms. Per multiply-add the
two products run at about 10 T/s against the weight GEMMs' 12-14 T/s. Reading the keys and
values is 7% of the step (`SPLOSH_ATTN_ALIAS=64`, which reads every chunk from the same 64
tokens, 602 ms against 650), not the 20% the old `d8` probe suggested: with a loop-invariant
chunk the compiler lifts the whole score product out of the loop, so `d8` times "no score
matmul". Tried at 100K and not kept: the blocks of one span dispatched together (645 ms against
605), 64 or 256 spans (636-800), loads that touch the next chunk's keys and values ahead of use
(660-673; a load blocks its thread, so it moves the wait and adds to it), 96-query tiles (659
at best, 1,200 at worst), 128-token chunks (885), relaxed precision (no change).

**The last layer.** Its output is read only for rows that want logits; nothing taps it for
the draft (the last tap is layer 61). For a prompt being evaluated it has only to store its
keys and values, so its queries, scan, merge, output projection and MLP are skipped for those
rows: one sixteenth of the attention and one sixty-fourth of the weights. Rows that want logits
come first in a step, so in a mixed step the layer is computed for a leading run of rows.
128-row step: 278 to 273 ms at 1K, 397 to 375 at 30K, 623 to 599 at 100K. Generation after a
6K prompt is token-identical with and without (`SPLOSH_FULL_LAST_LAYER=1`).

**Heat.** Over 7,168 rows of continuous prefill at 1K context (about 19 s) the rate is 350-390
tok/s whatever the step width (128, 256, 512 or 2,048 rows), against 465 for the first four
seconds. The machine was in the Automatic energy mode; it has a High Power mode
(`pmset -g cap`). A benchmark of a few steps measures the cool rate, and a real prompt gets
the warm one.

**Against Splish, part by part** (2026-10-02, High Power mode, this M5 Pro). Cold prompts,
each server alone: 6.4K tokens 448 tok/s against Splish's 499; 54.5K tokens 336-342 against
386 (Splash 380). Splish's own profiler (`decode-profile`, which replays each dispatch of a
2,048-row prefill command and sums GPU time per kernel) against the same split here, per 128
rows at short context:

| Part | Splish | Splosh |
|---|---|---|
| Weight GEMMs | 228 ms | 226 ms |
| Gated-delta layers | 12 ms | 16-21 ms |
| Attention (about 1K context) | 4 ms | 16-18 ms |
| SiLU and the down projection's operand | in the up GEMM | 5 ms |
| Norms and the rest | 2 ms | about 6 ms |
| Whole step | 247 ms | 270 ms |

The GEMMs are level (its per-shape benchmark, `q4-prefill-profile`, gives the same 13 T
mul-add/s). The difference is everything else, and most of it is the attention scan: it costs
about 1 ms a layer at any context, 17 ms a step at 64 tokens of context, 150 ms for a 512-row
step, growing with the square of the step's width. That is the chunks that hold the step's own
rows: about 130 us of core time each against 4 us for a chunk of older context. On a real
7,360-token prompt the scan is 20-34 ms a step, and without it the prefill runs at 473-494
tok/s. Why is not found. What the probes say (all step-by-step paired, ±1-2 ms): it is not the
masked branch (`d16`), the rescale path (`d15`), keys written a stage earlier
(`SPLOSH_ATTN_ALIAS`), the query prepare or the key store (left out: no change), the chunks
holding the step's own rows (`d24`: left out at 4K and 36K, no change), the first or the last
span (`d17`, `d18`), launching the threadgroups (`d21`, an empty kernel: 0.1 ms) or their set-up
and final writes (`d22`: 2.6 ms). It goes when the two matmuls do (12 of 19 ms) and with the
loop around them (5 ms). It does not grow with the number of spans, and it grows faster than
the step's width: 1.1, 2.5, 6.2, 19, 60 and 228 ms for 16 to 512 rows at 64 tokens of context.
A threadgroup with one chunk to do takes about 18 us of wall time whatever else is running,
where one with 35 chunks takes 160 us shared between sixteen cores (`d26`: one span's
threadgroups working among idle ones are as fast as when all work). Short threadgroups that
use the accelerator behave as if they ran one after another. Splish's scan has the same
structure and no such cost at 2,048 rows a command, but making our steps wider does not
help, because of the same growth with width.

More of what it is not, from a second round (probes `d28`-`d39`, `sp_probe_spin`,
`sp_probe_matmul`): the GPU not spreading small dispatches (64 spinning threadgroups run on
24-46 cores' worth in any grid shape, with or without barriers and threadgroup memory); the
accelerator being slow to start, to switch between kinds of matmul, or after a stage of
ordinary work (a 64-group dispatch of three score-and-value rounds, all the scan's
ingredients, takes 31 us alone and 45-60 us as an extra stage anywhere in a real step);
subnormal probabilities (`d31`); the atomic flag (`d33`: plain memory is no faster); the
scale loads (`d39`: two vector loads a thread instead of thirty-two, no gain); the query
tiles or key chunks being different per threadgroup (`d34`). The scan's own pieces add up
in place: set-up and final writes 2.6 ms, a plain three-round walk 4 ms more (`d35`), the
real walk's masks and loop control 7 ms more (`d37`), about 15 ms a step.

Two things the benchmark overstates. `SPLOSH_CTXBENCH` fills the context with synthetic keys,
and scores against them rise past the first chunk's reference often enough that the rescale
path runs all the time: 25 ms of the scan at 36K (`d32`). On a real prompt that path is
idle (445 tok/s with it and without). Attention figures from the benchmark at long context
are high by that much; the real-prompt figure is the one to trust (20-25 ms of scan a step
at 3.7K average context, where Splish's whole attention is about 12).

**What the fixed cost was: the benchmark.** `SPLOSH_CTXBENCH` places each iteration's rows
after the last one's, so a run "at 64 tokens of context" of 44 iterations of 136 rows ends at
6,000 tokens, and of 512-row steps at 16,000. The scan was never dearer near the step's own
rows; the context was simply not what the label said, and the "growth with the square of the
step's width" was the context growing with it. The scan costs about 3 ms of set-up, final
writes and merge a step, plus its slope. Short-context figures from the benchmark need the
iterations kept few, or the label read as the starting context.

**The rescale check, cheaper.** What does cost in the loop: the test for a score rising past
its row's reference was a conditional store per element (to the row's flag and to the
threadgroup's). Gathered as a bit per element and stored after the loop, which is rare, the
128-row step is 1.4 ms faster at 8K, 6.4 at 36K and 17.6 at 100K (`d41` is the old form).
The quality probe against fp16 gives the same figures to the last digit either way. With no
check at all (`d32`, wrong) another 5 ms would go at 36K.

**Against Splish again, with that understood.** Real prompts through the servers: Splosh 285
ms per 128 rows at 3K average context, 308 at 9K, 377 at 27K; Splish 256 at 3K and 331 at
27K. So about 27 ms of base and 0.7 ms per 1K of context of slope (3.8 against 3.1). The
base: gated-delta layers 4 ms, the SiLU pass 5, norms 3, the scan's fixed part 3, and the
rest in stage overhead (547 stages a 128-row step against 46 per 128 rows of a 2,048-row
command). The slope is the scan loop itself, where the two engines have the same structure
and theirs is a fifth cheaper per chunk; its tile run in this harness, as was done for the
GEMM, is the next thing to do.

A warning from this search. A scan kernel sized with the wrong score capacity computes no
scores, poisons its output with NaN, and is much faster: `m48c32s8` "saved" 122 ms at 100K
and passed the speculation self-test (which compares the engine with itself) before
`tools/serve-smoke` showed the model answering with token 0. The capacity is what the
partition probe reports (`d5` variants, `SPLOSH_DUMP_PARTIALS`), 16 for every 48-query shape
tried, not what the formula gives; `m48c64s12` had been defined with 8 and was as wrong.
Correctly sized, 32-token chunks, 12 simdgroups and 4 simdgroups are all slower than the
64-token, 8-simdgroup kernel at every context (5 to 150 ms). A timing is worth nothing
before the output has been checked against a reference.

Not the cause, by measurement: step width (128 to 2,048 rows over the same token count,
439-461 tok/s all), span count (1 to 16; fewer is a few ms better below 8K and worse at 100K),
the scheduler (under 1 ms a step), key and value memory traffic (7% at 100K).

The gated-delta kernel as three dispatches (`sp_gdn_prepare`, `sp_gdn_chains`,
`sp_gdn_finish`) is 4-5 ms a step faster: the recurrence in eight
threadgroups of four simdgroups per head fills twenty cores evenly where one of thirty-two
does not (17 to 11 ms). Token-identical on a single prompt, and right in steps that mix a
speculative generation with prompts being evaluated (updates accepted from a slot's last
speculative run are folded into its state by `sp_gdn_commit` in the stage before, since the
eight threadgroups of a head cannot replay them between them). On from 32 rows.

Probes added for this: `SPLOSH_DUMP_DISPATCHES=<rows>` lists a step's dispatches and stages;
`SPLOSH_CTXBENCH_AB=gdn|gdnmask|attnmask|spans` alternates two settings step by step inside
one run, which is the only comparison that survives whatever else the machine is doing (two
runs a minute apart differed by 15% while the screen was in use).

## 7. Speculative decoding (DFlash 2)

Draft: `incoai/Qwen3.8-27B-DFlash2`, the same draft Splash uses. Block of `[anchor, mask x 7]`
(`block_size` 8 in its config), five non-causal layers, a 2048-token window of the target's
hidden states at layers 5, 19, 33, 47 and 61, then a candidate selector.

Cycle at short context: verify 62-64 ms (8 or 16 rows) + draft 7.4 ms + context push 0.8 ms +
CPU ~1 ms.

**The draft pass was reading row-major weights.** Its linear layers are quantised at load and
were left in MLX's row-major order, so none of the tiled accelerator kernels applied to them
and every matmul went through the old row-major tile kernel. A stage-by-stage profile
(`SPLOSH_DRAFT_PROFILE`: the pass truncated after each stage in turn) put 11 of its 17 ms in
the five layers' MLP matmuls, which are the same size as the target's and there take a fifth
as long. Quantising straight into the tiled layout: 17.2 -> 7.4 ms, proposals identical. What
is left is the 248K-row output head (2.8 ms for 15 rows; 2.2 ms is its weight traffic) and
3 ms of MLP. Earlier attempts on this pass (fewer stages, other split-K shapes) changed
nothing because the kernels they varied were not the ones running.

**Blocks of sixteen.** A verify step of 9-16 rows costs 2 ms more than one of 8, so the
checkpoint's block of eight was the cap on easy text. Asked for `[anchor, mask x 15]` it
obliges, but its first seven drafts change too, for the worse on prose. Isolating the first
eight rows (they attend to each other and the context only; rows 9-16 see everything) makes
them exactly the trained block, and the second half a bonus that only counts when the first
seven drafts are all accepted. Paired comparison, the three shapes drafted from the same
anchors and scored against the finished greedy output (`SPLOSH_BLOCK_STUDY`), tokens per step:

| Prompt | 8 | 16 | 8 + 8 isolated |
|---|---|---|---|
| refactor a module | 6.45 | 10.00 | 10.20 |
| JSON document | 5.35 | 6.85 | 7.16 |
| copy records from a 6K prompt | 5.40 | 6.71 | 6.66 |
| how-to with commands | 4.22 | 4.51 | 4.75 |
| Python function | 4.01 | 4.80 | 4.55 |
| explanation | 3.79 | 3.67 | 3.93 |
| bash script | 3.71 | 3.88 | 3.92 |
| Swift struct | 3.63 | 3.70 | 3.86 |
| TypeScript class | 3.58 | 3.85 | 3.71 |
| history answer | 3.41 | 3.50 | 3.55 |
| SQL | 3.40 | 3.65 | 3.62 |
| essay | 3.13 | 3.19 | 3.25 |
| story | 2.66 | 2.66 | 2.76 |

The isolated shape is never worse than the block of eight (by construction) and is the default
for a lone session. With several sessions the second halves compete for rows: they are taken
in order of expected yield (how often the session's blocks are accepted whole, times what its
tails have added), free of charge while they fit the tile rows the step is paying for, and
across a tier boundary only when the estimate clears a margin. A waiting prompt's rows come
first.

**Copied drafts.** When the output is repeating text already in the context (a file being
rewritten, records being listed), the continuation of the earlier occurrence is a better
draft than the model's, and needs no draft pass. `LookupIndex` keys every three-token run;
the candidate with the longest shared run wins. Scored from the same anchors as above:

| Shared run | Copied, tokens/step | Drafted (8 + 8), tokens/step |
|---|---|---|
| 3-9 tokens | 1.3-5 | 6-11 |
| 10-19 | 1.2-9 | 6-13.5 |
| 20 or more | 15.2-15.8 | 7.4-14.7 |

Short runs are repeated structure with different contents (JSON keys, a signature) and lose
badly, so nothing under ten tokens is copied, and each session keeps its own running yield for
runs of 10-19 and of 20+, starting from these figures.

Through the server, one session, greedy, 400 tokens, interleaved against the same build with
both switched off (2026-10-02, machine in use by its owner, so absolute figures are low):

| Prompt | Block of 8 | 8 + 8 and copies |
|---|---|---|
| refactor a module | 106 tok/s | 214 |
| copy records from a 6K prompt | 82 | 162 |
| JSON document | 82 | 111 |
| SQL | 68 | 85 |
| Python function | 83 | 90 |
| how-to | 61 | 66 |
| TypeScript class | 57 | 59 |
| story | 28 | 30 |
| essay | 33 | 32 |

**Long context changes the sum.** Attention is the part of a verify step that grows with
context, and a second half doubles it:

| Context | 8-row verify step | 16-row verify step |
|---|---|---|
| 1K | 62.5 ms | 65.1 ms |
| 16K | 67.1 ms | 73.9 ms |
| 50K | 77.6 ms | 89.9 ms |
| 150K | 105.3 ms | 141 ms |

So the planner charges each session 0.29 ms per thousand tokens of context for its first
eight rows and 0.22 for its second eight, and at 50K a second half is drafted only where
blocks are usually accepted whole. Decode at 54.5K context, a follow-up turn on a restored
prefix, 300 tokens, interleaved against the block of eight:

| Task | Block of 8 | 8 + 8 and copies |
|---|---|---|
| list twelve records from the context | 61.7 tok/s | 90.3 |
| records as JSON | 66.1 | 92.9 |
| Python function | 47.3 | 55.1 |
| essay | 27.7 | 27.8 |

Other notes:

- Sampling: at temperature 0.7-1.0 code decodes at 76-78 tok/s against 81 greedy, prose is
  unchanged. A copied token is a certain proposal, so its acceptance test is the target's own
  probability of it.
- The draft's attention over a full 2,304-position window is ~0.5 ms on the accelerator
  (fp16 x fp16 scores, 32-query tiles, the ring walked in 64-slot chunks so a chunk never wraps).
- CPU work that used to cost 5-10 ms a cycle: arg-max and top-k over the 248K-entry logits row
  (now chunk maxima with vector code, then only the k chunks that can hold a top-k value), and
  page faults in the selector's file-mapped codebooks (now touched at load).
- Prefill pushes draft context only for the last 2,288 positions.

**Not worth building: a second verified branch.** With the first miss spread evenly over the
seven positions and the target's token being the draft's runner-up in about a third of misses,
one extra branch would help about 7% of steps.

**What "lossless" can and cannot be tested as.** A speculative block accepted whole gives
bit-identical logits to the same rows evaluated as an ordinary step (`SPLOSH_VERIFYCHECK`,
8 and 16 rows). A row evaluated after a different history of partial acceptances does not:
the state differs in the last bits, activations are rounded to bf16 at every matmul, and now
and then those bits decide a rounding. Over 200-240 positions that is a mean KL divergence of
0.00003-0.0001 and a worst of 0.001-0.013 (a position where two tokens are nearly tied), an
order of magnitude under what int8 KV costs, and identical for 8- and 16-row blocks. It also
means an occasional arg-max flip at a near-tie, so greedy output is not bit-reproducible
across block shapes or batch compositions, and a free-running token comparison is not a
sound test. `SPLOSH_SPECTEST` is therefore teacher-forced: the same tokens through blocks
corrupted at a rotating position, each kept row's distribution against a one-token-a-step
reference, bounds of 0.001 on mean KL and 0.1 on the worst. A deliberately dropped row of
state (`SPLOSH_SPECTEST_BREAK`) gives a mean of 0.3-0.6.

## 8. Batching

Verify-shaped steps, burst (hot: up to 20% more):

| Rows | Step |
|---|---|
| 8 | ~60 ms |
| 16 | ~62 ms |
| 32 | ~84 ms |
| 64 | ~165 ms |

Through the server, greedy, 240 tokens each, 16 slots (single runs, 2026-10-02):

| Sessions | Code, total tok/s | Essay, total tok/s |
|---|---|---|
| 1 | 89 | 30 |
| 2 | 167 | 58 |
| 4 | 118 | 74 |
| 8 | 140 | 86 |
| 16 | 134 | 82 |

Two sessions cost the same step as one. The 17–64 row range is the least efficient per row
(no deferred norms there); the dip at four code sessions is that range. (This table predates
the tiled draft weights, which take ~10 ms off every cycle.)

Rows come in tiers: up to 16 cost one figure, then tiles of 32 (24 rows cost what 32 do, 48
nearly what 64 do). Two things follow from that.

- **Prefill while other sessions decode.** It used to get eight rows a step, to protect the
  streams. Seen on a live server running a batch of agents, that is the wrong trade: one
  session decoding at 66K context, three prompts of 11-25K behind it sharing eight rows a
  step (about 75 tok/s between them, two of the three at zero), for minutes. Sessions only
  reached decode one at a time, so they never decoded together, which is where batching
  pays. The width of a mixed step is now chosen to make the most progress per millisecond,
  counting a generated token as `decodeWeight` prompt tokens (1 by default, which always
  gives a waiting prompt the full step; 8 keeps a stream at about 80% and gives prefill about
  65%; 20 is the old behaviour). The default was 3 at first; at 100K context that scored
  32-row and 128-row steps almost equally and chose the narrow one, 120 tok/s of prefill
  against about 180. Splash, for comparison, never mixes the two: it alternates a prefill
  command of up to 500 ms with a decode command, so prefill keeps about 80% of the GPU:

  | Rows in the step (one decoder) | Decoder keeps | Prefill gets |
  |---|---|---|
  | 16 | 100% | 27% of its full rate |
  | 32 | 80% | 65% |
  | 64 | 43% | 82% |
  | 128 | 25% | 100% |

  A stream plus three 3.1K prompts arriving behind it: the prompts' first tokens at 10, 21
  and 31 s in 128-row steps of ~370 ms, against roughly 30, 60 and 90 s at eight rows a step.
  While a prompt waits, second halves of double blocks are not drafted.
- Second halves of double blocks (section 7) are planned against the same tiers, after any
  waiting prompt. An earlier version corrected the tier costs from observed steps; a hot
  machine then made its 32-row steps look as dear as the 64-row step nobody had tried, and
  the plan oscillated. The costs are constants now: only their ratios matter.

Double blocks and copies against the block of eight, total tok/s over mixed prompts, 300
tokens each, interleaved: +0.6% at two sessions, +1.2% at three, -4.9% at four, +6.1% at
eight. That is the noise of the benchmark (sessions of unequal length, single runs); the gain
is for a lone session.

## 8a. Long context, end to end

Through the server, one session, a synthetic archive of numbered records with a question about
one in the middle (answered correctly each time), 2026-10-02, machine hot:

| Prompt | Cold prefill | Decode at that context | Follow-up turn (prefix in memory) |
|---|---|---|---|
| 54,482 tokens | 175 s (311 tok/s average) | 52–66 tok/s | 0.3 s to first token |
| 155,032 tokens | 770 s (201 tok/s average) | 37–45 tok/s | 1.3 s to first token |

After a server restart the 54K prefix came back from disk (1.88 GiB) in 0.6 s to first token.
The prefill step grows from ~380 ms at the start to ~670 ms at 85K; by 150K attention is over
half of it.

## 9. Disk prefix store

A prompt's KV pages, the recurrent state at its end and the draft's context ring are written
to `~/Library/Caches/Splosh/prefix-cache` when its session ends, and looked up when a prompt
arrives. A 5,905-token prompt: 14.0 s cold, 0.7 s for the follow-up turn after a server
restart. A growing conversation is not rewritten each turn (nothing is stored within 2,048
tokens or a sixteenth of an existing prefix; a longer prefix supersedes shorter ones). About
1.8 GiB per 50K tokens, 16 GiB budget by default, least-recently-used eviction.

**State checkpoints.** A stored prefix serves only prompts that extend it. Two conversations
with the same system prompt share a beginning, and what the second cannot get from the first
is not KV (that depends only on the tokens before it, so any prompt that begins the same way
has it) but the gated-delta state at the point where they part. So the scheduler keeps that
state at three kinds of position while a prompt is evaluated: where the server says the system
block ends, where a prompt was found to leave everything seen before (if that cost it more
than 512 tokens), and every 4,096 tokens up to 16K then every 16K. In memory a slot keeps one
(the shared prefix); on disk they are small files of their own (150 MB), neither superseding
nor superseded. A new prompt takes the longest checkpoint its beginning matches, with KV
copied from whichever slot (busy or not) or stored prompt covers it, into a free slot so
that an idle conversation is not thrown away to start another.

A 9.5K-token system prompt, new conversation each time: first 22 s; then 0.4 s, 0.4 s, two at
once 0.5 s each, and 0.7 s after a server restart, 9,498 tokens reused each time and every
answer right. The seven prompts in the store after a morning of one harness's agent runs
shared exactly 8,836 leading tokens and nothing more: that beginning had been evaluated seven
times, about half of all the tokens the server evaluated.

## 9b. Things a live batch of agents found

- A client that gives up while its prompt is being evaluated used to leave the session
  running: nothing is written to the connection during prefill, so nothing noticed. A session
  with no token yet now reports progress every two seconds; a streaming response writes it as
  an event-stream comment (`: evaluating prompt 896/9566`), a non-streaming one as whitespace
  ahead of its JSON. Either keeps an idle-timeout from firing and fails as a write when the
  client has gone, which cancels the session (5-6 s after the client disappears, measured).
  A non-streaming response that has started this way has already sent status 200, so a later
  failure arrives as an error body.
- "KV pool exhausted; no cached prefix left to evict" reached the harness as an error. The
  pool was a fixed 1,024 pages (262K tokens for all sessions together) and a session that ran
  out was failed. Three changes. The pool is committed only as it fills, so by default it is
  now sized to what the machine can hold with everything else loaded (about 570K tokens on a
  64 GB M5 Pro; `kvPages` still overrides). A prompt is admitted only if there is room for
  it and 4K tokens of generation, counting what running sessions were admitted for and have
  not taken yet; otherwise it stays queued, with progress events, until there is. And a
  session that needs a page mid-run sits steps out until one is free; if every session is in
  that position, the newest prompt still being evaluated goes back to the queue. Tested with
  a 40-page pool and five 3.1K prompts at once: three admitted, two queued, all five answered.
- Ctrl+C did not stop the server while requests were in flight: the HTTP layer waited for
  them and they waited on their sessions. The first signal now ends every session and the
  process leaves within two seconds; a second leaves at once.

- Two servers no longer fit side by side. With the weights, state and scratch held resident
  and a KV pool of 18 GiB of address space each, a second instance started for a test took
  the GPU out of memory ("Insufficient Memory" command-buffer errors) while the first was
  serving 125K-token sessions. Benchmarks that need two engines loaded need the other one
  stopped, or a small `kvPages`.
- What the dashboard shows now, because the questions kept being asked of it: each session's
  rate over the last few seconds beside its average since admission (the average includes
  time spent waiting, which is why one session read 60 tok/s while the server read 280);
  what a session is generating (reasoning, answer or tool call, from the server's output
  parser) and how many tokens of each; and combined rates over the time each kind of work was
  actually going on.
- An agent is one request per turn, so a history of finished requests was hundreds of rows for
  a handful of agents. The history is by conversation now (`ConversationLedger`): a request
  joins the conversation whose latest prompt it continues, and its figures are added to that
  row. "Continues" is compared on the prompt up to where it opens the reply, not the whole
  prompt: the prompt ends `<|im_start|>assistant\n<think>\n`, and the next turn renders that
  reply from history, where an empty or dropped thinking block tokenises differently, so no
  turn's prompt begins with the whole of the one before. Agents that share a system prompt but
  not a task stay separate. Checked with rendered and tokenised prompts in the unit tests, not
  yet against a live harness.
- The share of drafted tokens the model kept is shown for the server, each session and each
  conversation (accepted / drafted, copied drafts included). Every verify step yields one
  token whatever the drafts were, and that one is not counted as accepted: tokens per step is
  accepted drafts per step plus one, so the percentage reads lower than tokens per step suggests.
- A controlled stop: the first Ctrl+C lets requests in flight finish (up to `drainSeconds`),
  then every conversation's prompt-end context is written to the disk store whether or not
  something close to it is stored already, so the next server resumes each from disk.
- That was not enough to restart under a batch of agents, and the first try at it ended every
  agent at once. Two reasons. A stopped server refuses connections, and an agent between turns
  makes its next request within seconds. And a reply the scheduler cut short was written to
  the client as a normal completion: the endpoint's default for "how it ended" was `stop`, so
  a cut stream closed with `finish_reason: "stop"` and `[DONE]`, and an agent given half an
  answer with no tool call takes its work to be done. Now:
  - A cut reply is a failure. Streaming: the connection ends without its last chunk, and
    nothing is said first. Whole-body: 503, or the body left unfinished if the status had
    gone. A stream of events that simply stops is treated the same.
  - `splosh serve` is two processes (`ServeSupervisor`). The one started by hand binds the
    port and never loads the model; the engine is its child, serving on a private Unix-domain
    socket, and connections are passed through byte for byte. `splosh serve --restart`
    replaces the engine without closing the port: the old engine drains and saves, new
    connections are accepted and kept (the request is read, so the client's write completes)
    and handed to the new engine when it answers. The engine asks clients to use each
    connection once (`Connection: close`), so there are no idle connections to be closed
    under a request. A stop still ends both; an engine that dies is started again.
  - `tools/restart-smoke` checks this without the model (`serve --echo`): a reply in flight
    arrives whole, a 200 KB request sent mid-restart is answered by the new engine, nothing is
    refused. With the model it has not been run yet.
  - `splosh serve --takeover` starts in place of whatever server is on the port, an older
    one-process build included: that one is asked to stop, this one binds the port the moment
    the listener closes (a few milliseconds in which a connection is refused), keeps what
    arrives, and loads the model once the other has gone.
  - What the harness does (dsh 0.1.7, read from its installed source and its session logs),
    which decided the details:
    - It streams, through the OpenAI Node SDK with the SDK's retries off and its own on: five
      tries at 0.5, 1, 2, 4, 8 s, for a refused or broken connection, a stream that ends
      without a `finish_reason`, or a 5xx. An error frame in the stream is not retried: its
      message is classified by keyword and ours had none. So a cut stream is a dropped
      connection and nothing else.
    - 15 s of retries is less than a model load, which is why the agents that were between
      turns died of "Connection error." (20 turns in the logs), and why the port has to stay
      open rather than come back quickly.
    - `finish_reason: "stop"` with one reasoning delta and no tool call ends a turn as
      completed: 10 turns in the logs, in bursts at the two moments the server was stopped.
    - It gives up 300 s after the last content, reasoning or tool-call delta, the first
      included; response headers and comment lines do not count. A prompt that needs more
      than 300 s of evaluation (about 40K new tokens here) therefore fails whatever the
      server sends meanwhile. The setting is the harness's: `streamIdleTimeoutMs` on the
      provider.
  - So a restart does not wait long for requests in flight (`restartDrainSeconds`, 30): the
    ones waiting behind it wait as long, inside that 300 s. A request cut then has its
    context saved as far as it got, a prompt half evaluated included (the state is exported
    where the prompt stands; before, only a finished prompt's end was), is sent again by the
    harness, kept at the port, and carries on in the new engine.
  - The holder keeps what a client sent until the engine answers. An engine that closes a
    connection without a byte of answer (told to stop as the request arrived, or dead) has
    not served it, and the request goes to the next engine: `tools/restart-smoke` kills the
    engine with a request in hand and the client gets its whole reply.
  - A review of the first version found: an engine orphaned if the holder was killed while
    the model loaded (it checked for its parent only afterwards); `--restart` trusting a pid
    file, so a stale one could signal an unrelated process (it now requires the recorded
    process to be the one listening on the port); a restart asked for while an engine was
    starting silently dropped; a stop during a restart's drain counted by the engine as a
    second signal and cutting the requests; a new engine failing to start taking the port
    down with it (it is retried, connections still kept). All fixed.
  - What a restart cannot hide: a streamed reply that was already under way when it was cut
    (the client has part of it; the harness retries the turn), and a wait longer than the
    client's own timeout.

## 9c. What a day of agents actually evaluated

From the harness's own session logs (dsh 0.1.7, 122 requests in 16 conversations, 2026-10-02;
it records prompt, cached and completion tokens per request). Prompts: median 35,855 tokens,
90th percentile 157,144, largest 195,775. Tokens evaluated per request: median 2,202, 90th
percentile 15,351. 91% of prompt tokens were served from what the server held. Of the tokens
that were evaluated, on requests that followed another of the same conversation:

| What | Tokens | Share |
|---|---|---|
| New content (tool results, messages) | 417,272 | 80% |
| The previous reply, evaluated again | 48,370 | 9% |
| Earlier parts of the prompt, evaluated again | 54,925 | 11% |

A fifth of the work was work already done once, and at these lengths it is the dear kind
(200-340 tok/s). Three causes, each now dealt with:

- **The reply comes back as other tokens** (7 of the 10 replies evaluated again; the other 3
  were server restarts). Not whitespace or argument formatting: in all 84 clean cases and in
  these 10 the reply re-renders to the same text. The model had spelt a word its own way,
  `M|ol|to` where the tokeniser makes `M|olto`, `FL|ASH` for `FLASH`, once in a few thousand
  tokens, so long replies are the ones hit (median 2,958 tokens against 405). Token for token
  the next prompt parted from the slot inside the reply and the reply was evaluated again
  from the prompt's end. `ReplyAliases` remembers each reply as generated, with its bytes;
  a prompt that begins with the reply's prompt and carries on with the same bytes has that
  stretch replaced by the generated tokens, so it runs on from what the slot holds.
- **A new turn drops the last turn's thinking.** The harness sends the finished turn back
  without its reasoning. The prompt then renders the reply as `<think>\n\n</think>` where
  the last prompt ended `<think>\n`: it parts from that prompt at its final token, the
  checkpoint at the prompt's end is unusable, and the conversation fell back to a regular
  checkpoint (cached counts of exactly 12,288, 32,768 and 131,072 in the logs), 2,854 tokens
  short at the median and 22,620 at the 90th percentile. The slot's checkpoint is now taken
  at the prompt's stable point, where it opens the reply, four tokens before its end; the
  extra step is the four-row one that finishes the prompt.
- **More conversations than slots.** The one displaced was on disk only as of whatever prefix
  the store had last thought worth writing. It is now written whole, last reply included,
  when its slot is given to something else and when the server stops (`retire`), with the
  stable-point state beside it. The store's size, unless set, is a tenth of the free space
  between 16 and 96 GiB: a conversation at the 90th percentile is 5 GiB and sixteen of them
  do not fit in 16.

New agents starting under a shared system prompt were the other waste in the morning's logs
(nine first requests of about 10.5K tokens evaluated in full); the shared-prefix checkpoints
of section 9 had already fixed that by the afternoon (8,836 of 10.5K reused).

## 9a. A harness trial (opencode 1.18)

One real agent run against the server, bash and web access denied, in a scratch project:
it globbed, read a file, fixed a bug with an edit tool call and reported. The request log:

| Request | Prompt | Reused | Evaluated | Then |
|---|---|---|---|---|
| title, a side request | 610 tokens | 0 | 1.6 s | 336 tokens at 64 tok/s |
| first turn | 9,199 | 0 | 27.2 s (338 tok/s) | 54 tokens |
| after each tool result | 9,332–10,228 | all but 20–358 | 0.1–1.0 s | 73–200 tokens at 62–102 tok/s |
| a subagent | 6,523 | 0 | 18.7 s | 299 tokens at 100 tok/s |

So within a session every turn reuses its prefix, and the cost is at the start: the system
prompt and tool definitions are 9.2K tokens, evaluated once per session and again (a
different 6.5K) per subagent. A second session reused nothing, and would not have even with
a checkpoint at the end of the system block: this harness's system prompt is not the same
from one session to the next (its skill list comes out in a different order, at token 7,475
of 9,170). Reusing a prefix up to the point where two prompts part needs the recurrent state
at that point, which nothing keeps. A design that does (state checkpoints at regular
positions and at learned divergence points, KV taken from any prompt that covers the prefix)
is on the `prefix-checkpoints` branch: it builds and is untested.

## 10. Comparison with Splash 1.1.0

Both servers loaded for the whole run, same client (`tools/bench/server-ab`), greedy, thinking
off, requests interleaved, medians, machine hot. Three sessions on 2026-10-02: the second
after the draft-attention and deferred-norm work, the third after blocks of sixteen, copied
drafts and the tiled draft weights (section 7):

| | Splash | Splosh | Splosh / Splash |
|---|---|---|---|
| Short code prompt, decode (4, 6, 6 rounds) | 77.3, 78.9, 76.5 tok/s | 76.0, 74.5, 100.1 tok/s | 0.98, 0.94, 1.31 |
| 6K prompt, prefill (4 rounds each) | 364, 411, 410 tok/s | 324, 342, 328 tok/s | 0.89, 0.83, 0.80 |
| 6K prompt, decode of 260 tokens (repeating records from the prompt) | 67.3, 77.5, 80.3 tok/s | 64.5, 73.3, 148.3 tok/s | 0.96, 0.95, 1.85 |

So: decode is now ahead, by about 30% on new code and by more where the output repeats its
context. Prefill is 11–20% behind, and that has not moved. Within a round, whichever engine
runs second is on a hotter machine and loses about 5%.

**Against Splish v1.1** (a fork of Splash with tuned kernel choices for this chip and a copy
rule; it states that it does not change Splash's prefill), each engine alone, one request at a
time, same prompts, single runs:

| | Splish v1.1 | Splosh |
|---|---|---|
| Decode, copying records at 1.6K / 6.4K / 18K context | 111 / 106 / 100 tok/s | 189 / 147 / 110 |
| Decode, new code (three prompts) | 85 / 77 / 89 | 105 / 98 / 101 |
| Decode, essay | 22 | 35 |
| Decode, refactor a module | 108 | 211 |
| Prefill at 1.5K / 6K / 17K, two rounds in both orders, before the two fixes below | 494 / 511 / 453 | 305 / 425 / 389 |

The prefill row was taken before the small whole-tile kernels and the residency set (sections
3 and 1). The 1.5K figure was the idle wait, not the engine.

**Where the prefill gap is.** Each engine alone, restarted between measurements, prompts of
1.6K, 6.4K and 17.1K (time to first token):

| Prompt | Splash | Splosh |
|---|---|---|
| 1,566 | 3.08 s | 3.88 s |
| 6,365 | 12.40 s | 15.46 s |
| 17,102 | 39.41 s | 46.24 s |

Fitted as a per-token base plus a term that grows with context: Splash's base is 1.76 ms a
token (225 ms per 128 rows), Splosh's 2.29 (293 ms); the context term is 8.2 ms per 1K per 128
rows for Splash and 6.2 for Splosh. So Splosh's attention is the cheaper one and the whole gap
is in the base, which is nine-tenths matmul. Splash's base equals what the GEMMs here cost
with the q4 scale and bias work left out. GEMM-only steps, 128 rows:

| Kernel | Step |
|---|---|
| as shipped | 241 ms |
| no scale or bias, partial results summed | 230 |
| same, every tile reading one block of weights | 220 |
| accelerator accumulating across groups, nothing else | 211 (205 with no weight traffic) |

Splash's weight files are the same MLX q4 with a scale and bias per row and group (its layer
files are the size that needs), so it is doing that arithmetic for less than the 30 ms it
costs here. How is the open question.

Where the two differ in design: Splash evaluates 2,048 rows per prefill step and writes bf16
outputs with the SiLU gate fused into the up-projection; Splosh steps 128 rows (256 and 512
measured no faster) and keeps fp32 outputs. Its gated-delta recurrence has the same shape as
Splosh's. Splosh's non-GEMM share of a 128-row step at 3K context is about 16% (gated-delta
22 ms, attention 18 ms, the rest 6 ms, of ~297 ms); that share is the most likely home of the
prefill gap, since the GEMM itself runs at the accelerator's rate.

Not compared head to head: 50K and beyond, where Splosh's attention costs 2.8 ms per 1K tokens
of context per 128 rows.

For scale, the night's starting point on the same kind of prompt was 55 tok/s decode and
180 tok/s prefill at 15K. Earlier single-run Splash figures in this file's history (473 tok/s
prefill, 98.7 tok/s decode) were taken on a cooler machine and are not comparable with the
table.

## 11. Things that did not work

- Single-affine q4 KV; q4 KV generally (section 6).
- Three separate attention passes with scores in device memory: bound by scratch traffic.
- Fused gate/up/SiLU GEMM; bias term as a second accelerator matmul; `simdgroup_matrix`.
- 64- or 96-query attention tiles; 128-token chunks; relaxed precision; different simdgroup
  counts for the score and value matmuls (scores on four simdgroups put the softmax
  bookkeeping on half the threads and nearly doubled the step).
- Wider single-simdgroup tiles (8 x 64, 16 x 128) for verify steps.
- Fusing the draft pass's stages (convs emitting GEMM operands, the residual conv carrying the
  norm behind it, the merge emitting the output projection's operand: 19 -> 14 stages a layer):
  identical proposals, 1 ms slower. Its stages are not what the draft pass was waiting on;
  row-major weights were (section 7).
- A plain block of sixteen from the draft model: better on code, worse on prose (3.5 -> 3.15
  tokens a step), because the first seven drafts change. The isolated first half fixes that.
- Copying from shared runs shorter than ten tokens (section 7).
- The wide emitting GEMM for 17-64 row steps: 105 ms against 84 at 24-32 rows, no different
  at 48-64. Deferred norms for that range still need their own kernel.
- Fusing SiLU into the up-projection's epilogue: the separate SiLU stage is 4 ms of a 292 ms
  step, so there is nothing to win.
- Taller GEMM tiles, again (64 or 128 rows a tile, 8 or 16 simdgroups): 2-5x slower.
- Letting the accelerator accumulate across quant groups (`multiply_accumulate`) with the
  destination rescaled in place between groups: the accumulate alone is fast (see section
  10), but touching the destination between runs costs more than reading a separate partial
  result does. Several row tiles per threadgroup over one weight chunk: 1.3-2.3x slower.
- Relaxed precision, retested once the prefill GEMM was known to be accelerator-bound: no change.
- 256-, 512- or 1,024-row prefill steps: no gain, and slower past 512, even with the wide
  GEMM taking row tiles in cache-sized blocks of four within one dispatch. Every component
  scales linearly with rows, so the cost of a stage is not a fixed launch cost that a bigger
  step would amortise; it behaves like a tail proportional to the stage's own work.
- More attention spans for narrow steps (partials traffic cancels the shorter chains).
- Holding the gated-delta state in registers while keeping three barriers per token: only
  86 -> 61 ms; the barriers were the cost.
- Staging the recurrence's q, k, v and scalars in threadgroup memory, 16 tokens at a time
  (Splish does this): no change. Nor do the cross-lane shuffles or the output store cost
  anything measurable. The recurrence itself is 14 ms of a 128-row step and behaves as if
  about four simdgroups per core make progress at once; the phases around it are 8 ms.

## 12. Open questions, in the order they look worth attacking

1. Prefill: Splash's base cost is 225 ms per 128 rows against 293 here, and it is in the
   GEMM's per-group scale and bias (section 10). Gated-delta is 23 ms and attention 5 ms at
   no context, so they are not where the 68 ms is.
2. Deferred norms for 17–64 rows (needs a 64-column emit kernel that is not slower).
3. The draft pass is 7.4 ms, of which 2.8 is the output head for 15 rows. A head over a
   subset of the vocabulary, or fewer candidate rows, is what is left to try there.
4. Whether the draft is hurt by being quantised to q4 at load (the reference runs it in bf16).
5. Symmetric `int4b` or `fp8` KV on the accelerator, with the quality probe as the gate.
6. Attention at 100K and beyond (over half the prefill step), and a 50K head-to-head with
   Splash.
7. Sampling (temperature > 0): the rejection rule itself is now a pure function with a
   statistical test (400K draws against four kinds of draft: exact to five standard deviations,
   acceptance rate equal to the overlap). Not yet measured: how much acceptance falls at
   temperature 1 on real prompts.

**The scan rewritten on Splash's structure** (2026-10-02, `candidates/attn_scan_r.metal`, now the
default as `r48c64s8`). Scores stored whole to threadgroup memory, softmax by four lanes a row
with vector loads, a reference that moves only when a row's maximum passes it by 30. Paired
128-row steps against `m48c64s8`: 7 ms at 8K, 33 at 36K, 117 at 100K. Variants at 100K against
it: `r48c64s8q4` 16 ms slower, `r48c32s8` 70, `r48c32s8q2` 82; the agents' `b48c64s8` 13 and
`h48c64s8` 6 slower. Same answer on the 15K records prompt; quality probe level. Chosen in the
server by `attentionScan`. The gated-delta candidates (`gdn_b.metal`: prepare b1, chains b2,
history b1) are 3-4 ms a step in three timings and now the default; the fused SiLU
(`SPLOSH_MLP_FUSED`) gains nothing, because it swaps the SiLU stage for a second accelerator
stage and removes none. Through the servers, cold prompts: 6.4K Splosh 459, Splish 511, Splash
502 tok/s; 19.3K 411-435, 452, 456; 59K 375 (354 in an earlier run), 374, 370. A reading of
Splish's source part by part (`audit/splish-study-findings.txt`) found no single large fixed
cost left: bf16 where we move fp32 (1-2 ms), the a and b gate projections inside its packed
product (1-3), per-request costs in our server (draft pushes, the split at the reply opener,
the tokeniser, the first step after idling: 0.4-0.6 s a request), and the open question of
what a stage boundary costs at 128 rows (0-18 ms).
