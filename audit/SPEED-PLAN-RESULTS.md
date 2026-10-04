# Speed plan: what was done and what it measured

2026-10-03, overnight, with the live server stopped. Plan: `audit/SPEED-PLAN.md`. Every timing
here was taken on this Mac in High Power mode, interleaved or paired as
`splosh-benchmark-method` asks; single figures are marked as such. Nothing is committed.
`splosh serve --restart` picks it up.

## Item by item

| Item | Outcome | Measured |
|---|---|---|
| 0. Power mode | Already High Power (`powermode 2`); nothing to do | sustained/burst 128-row step: 8K 275/267 ms (465/479 tok/s), 36K 402/375 ms (318/341 tok/s); the 350-390 tok/s sustained figure of Automatic mode is gone |
| 1. Measurements | **Built**: per-request time outside steps, copied vs drafted steps, step time by tier and context | request log line, `/v1/stats` (`totals.finishedOverhead`, `totals.blocks`, `stepTiers`, per session and conversation), dashboard cards; `timings.outside_steps_ms` and `first_step_ms` in each response's usage |
| 2.1 Store copy | **Stopgap built**: skipped while any other session is active, waiting or arriving; KV and state exported with one copy, not two; restores read the mapped file and other slots straight into the pool | store copy 46-123 ms a turn at 31-39K (single copy). Incremental store **not built** (see below) |
| 2.2 State exports | **Built**: one copy, shared by marks at the same position | ~10-13 ms each, ~2 a turn |
| 2.3 Wake on arrival | **Built, measured, removed**: no help (first step 566 ms without, 596 with) | |
| 2.3' Keep-warm | **Built, on by default** (`SPLOSH_KEEPWARM`, seconds, 0 = off): a tiny pass over the KV pool every 0.5 s for 10 min after the last step | first step of a turn after 3 s idle: **556-623 ms -> 325-338 ms**; a 2.1K-token agent turn prefills at ~387 tok/s instead of ~368 |
| 2.4 Turn-mark step | **Not built**: the engine rejects a slot whose rows mix ordinary and speculative rows in one step, and its gated-delta kernels run a slot's rows as one run; needs kernel changes | the extra step is 74-77 ms at 35K for a lone session; in a batch it shares a step with others |
| 2.5 Lookup and ledger | **Built**: copy index kept with the slot and extended (`LookupIndex.extend`); ledger join measured <1 ms, left alone | lookup rebuild was 2-3 ms at 35K |
| 2.5 Draft context push | **Built, measured, reverted**: folding the push into the next draft pass gave nothing | push costs ~0.9 ms a step; decode 35.1/97.7/32.7 tok/s folded vs 36.2/96.4/33.5 not |
| 2.6 First long prompt slower | **Hypothesis refuted**: writing the KV pages from the CPU first changes nothing | 20K prompt, paired: 424.0/423.2, 418.3/412.8, 406.4/401.2 tok/s (untouched/touched); touching 680 MB took 0.02 s |
| 3.1 Draft quantiser | **Built, on by default**: per-group range by least squares over nine clipped ranges (`SPLOSH_DRAFT_QUANT=minmax` restores MLX's rule) | 13 prompts, same anchors: 8+8 block 4.645 -> **4.753 tokens a step** (+2.3%, better on 10 of 13); quantising at load 0.8 -> 2.7 s |
| 3.2 Longer copied blocks | Waits for live figures (item 1 now counts copied steps) | in short agent turns 1-2 copied steps a turn gave 5-15 tokens each |
| 3.3 Draft head | Not attempted (low priority in the plan) | |
| 4.1 Steps of 17-64 rows | **Built, narrowly**: exactly 64 rows (four sessions' double blocks) now use the whole-tile kernels with deferred norms (`SPLOSH_SPLIT_WHOLE=0` reverts) | 64 rows 139 -> 136 ms (8K), 171 -> 168 (36K); 32, 40 and 48 rows are 9-19 ms *slower* that way, so they stay on split-K; same output tokens either way |
| 4.2 Fuller scan tile | **Built as candidates, slower**: `r60c64s8v16`, `r60c64s16v16`, `r48c64s8v16` pass the CPU-reference test | 40K, 128 rows: 11.0, 11.2, 11.3 ms against 9.9 for the shipped `r48c64s8` |
| 4.2 Skipping negligible chunks | **Probe run, idea dropped** | real 85K-token prompt: 0.2% of chunks have every query below 1e-3 of its maximum, 0% below 1e-4 |
| 4.3 Block-scaled q4 | Not attempted (last in the plan, a re-quantisation of the model) | |
| Pipeline hint (headers) | **Probe run, dropped**: `threadGroupSizeIsMultipleOfThreadExecutionWidth` | 272/275 ms (8K), 360/363 (36K); same output |
| Untracked buffers, Metal 4 barriers | Not attempted: untracked hazards across the step's several command buffers risk silent races, and the barrier probe needs a Metal 4 queue | |

## What a turn costs now, outside the GPU's steps

From the agent-like run (one conversation, a 9K system block, a 20K first message, then 2K
tool results with 3 s pauses, 31K -> 39K):

- prepare (template and tokenising) 14-29 ms; join <1 ms; admission <2 ms
- state exports 13-26 ms (1-2 copies a turn, 6 on the first)
- store copy 0-123 ms (now 0 whenever anything else is running)
- first step after the pause: ~330 ms with keep-warm, against a steady 128-row step of ~390

Decode, one session: the scheduler's cycle is the verify step plus ~10 ms (the draft pass
8.7 ms, consuming the result 0.9 ms).

## Not built, and why

- **Incremental disk store (2.1 proper).** Writing a conversation as appended pages needs a
  page-major file format; the current one is layer-major, so pages cannot be appended. A new
  format would orphan every stored conversation in the live cache at the next restart unless
  both are read. The stopgap removes the stall from batches; what is left is the disk traffic
  and the copy for a lone session, which happens while its agent runs a tool.
- **2.4** as above: kernel work that could not be checked unattended.

## Engines compared

Splosh as built above (`prefixCacheDir = "none"`); stock Splash 1.1.0 (`splash serve`); Splish
from the upstream clone. Each engine alone in memory, started fresh, sent a 21K-token warm-up
prompt, then the measured requests. Every prompt is new (no prefix cache helps): synthetic
survey records followed by a request for a 1,000-word essay, greedy, thinking off, 800 tokens
allowed, so every reply stops at 800 and decode is timed over the same 799 tokens. The prompt
sizes came out about 10% over their names (the record generator's tokens per record). Two full
passes, the second in reverse order (pass 1 Splash, Splosh, Splish; pass 2 Splish, Splosh,
Splash). Harness: `tools/bench/engine-report`, tables by `tools/bench/engine-report-summary`;
raw results in `audit/snapshots/engine-report-2026-10-03-pass{1,2}.jsonl`.

### One prompt at a time

Mean of the two passes, each pass in brackets. Prefill is prompt tokens over time to first token.

| Prompt (tokens) | Engine | Time to first token, s | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|
| 10K (10,686) | Splish | 22.1 (22.1 / 22.1) | **484** (484 / 485) | **35.6** (36.3 / 34.8) |
| | Splash | 23.9 (25.7 / 22.1) | **450** (416 / 485) | **30.8** (31.3 / 30.4) |
| | Splosh | 22.9 (22.8 / 22.9) | **467** (468 / 467) | **35.9** (35.7 / 36.1) |
| 50K (54,536) | Splish | 138.3 (138.6 / 138.0) | **394** (394 / 395) | **32.3** (32.9 / 31.8) |
| | Splash | 151.0 (163.4 / 138.6) | **364** (334 / 393) | **27.0** (25.5 / 28.5) |
| | Splosh | 140.4 (140.3 / 140.5) | **388** (388 / 388) | **32.8** (32.8 / 32.8) |
| 150K (165,744) | Splish | 628.2 (629.0 / 627.4) | **264** (263 / 264) | **24.7** (24.8 / 24.6) |
| | Splash | 660.9 (686.5 / 635.3) | **251** (241 / 261) | **23.8** (23.2 / 24.4) |
| | Splosh | 601.0 (599.7 / 602.3) | **276** (276 / 275) | **25.8** (25.5 / 26.0) |
| 200K (221,306) | Splish | 970.7 (971.4 / 970.0) | **228** (228 / 228) | **22.9** (21.6 / 24.3) |
| | Splash | 988.2 (989.9 / 986.5) | **224** (224 / 224) | **21.0** (20.8 / 21.2) |
| | Splosh | 919.6 (920.4 / 918.9) | **241** (240 / 241) | **23.5** (23.3 / 23.8) |

### Four 20K prompts sent at once

All four start together. "Last first token": when the last of the four prompts had been
evaluated. "All done": when the last reply ended. "Combined": every prompt and generated token
over that time.

| Engine | Pass | First tokens at, s | All done, s | Generated tok/s, four together | Combined tok/s |
|---|---|---|---|---|---|
| Splish | 1 | 52, 158, 199, 244 | 270 | 14.7 | 330 |
| | 2 | 52, 114, 199, 245 | 270 | 14.6 | 329 |
| Splash | 1 | 52, 161, 203, 251 | 285 | 13.8 | 313 |
| | 2 | 52, 135, 183, 252 | 282 | 13.9 | 316 |
| Splosh | 1 | 51, 108, 168, 228 | 251 | 16.0 | 354 |
| | 2 | 51, 108, 167, 227 | 251 | 16.0 | 354 |

### What it says

- **Long context is Splosh's**: prefill 276 vs 264 (Splish) vs 251 (Splash) tok/s at 166K, 241 vs
  228 vs 224 at 221K; a 221K prompt is answered 51 s sooner than by Splish.
- **Short context is still Splish's, narrowly**: 467 vs 484 tok/s at 10K (3.5% behind), 388
  vs 394 at 54K (1.5%). Splash's own figures there moved 15% between passes (it ran first in
  pass 1, after an hour of these experiments); Splosh's and Splish's moved under 1%.
- **Decode is level with Splish or ahead at every size** (35.9 vs 35.6, 32.8 vs 32.3, 25.8 vs
  24.7, 23.5 vs 22.9) and 10-20% ahead of Splash. This is essay text, where drafts are accepted
  least; code and copied text decode much faster on Splosh (98 tok/s for code at short context).
- **Four at once**: Splosh has all four answers 19 s before Splish and 31-34 s before Splash, and
  every prompt but the first gets its first token sooner (108/168/228 s against 114-161/199/245).
