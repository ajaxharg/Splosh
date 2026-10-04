# Why it is fast

How the engine is made, and what each choice was worth. Back to the [README](../README.md).

- [Prefill](#prefill)
- [Decode](#decode)
- [Many sessions at once](#many-sessions-at-once)
- [What did not work](#what-did-not-work)

Three numbers about the machine decide almost everything.

- The GPU reads memory at about 290 GB/s, so **one pass over 14 GiB of weights takes 52 ms**.
  A step that produces one token can never beat 19 tokens a second.
- The GPU's neural accelerator multiplies 4-bit weights by 16-bit activations at about
  **14 trillion multiply-adds a second**, far beyond what ordinary shader arithmetic manages.
- Every point in a step where one piece of work must wait for another costs **40-60
  microseconds**, and a step has hundreds.

So prefill has to keep the accelerator saturated and waste nothing around it, and decode has
to get more than one token out of each pass over the weights.

![One step through the model: rows pass through the embedding, 64 layers of three gated-delta layers then one attention layer repeated 16 times, a final norm and the output head; weights, recurrent state and the KV cache are read from memory.](figures/step.svg)

## Prefill

**The weights stay 4-bit all the way to the multiplier.** The MLX pack stores each group of 64
weights as 4-bit codes with a scale and an offset. Splosh hands the codes to the accelerator
as they are and applies scale and offset to the product afterwards, 64 inputs at a time.
Nothing is expanded to 16 bits in memory, so a step reads 14 GiB, not 56.

**GGUF weights are decoded a block at a time, on the way in.** Most of what a GGUF file holds
cannot go to the accelerator as stored: 5- and 6-bit codes, and codes that index a table.
Widening them to bytes on disk would nearly double what a step reads. So the codes stay at
their own width, and each group of threads turns the block it is about to multiply into 16-bit
weights in its scratch memory, scale and offset already applied; the accelerator multiplies by
that. At 128 rows this costs about 6% more than the raw 4-bit path for the same product; at 16
rows the time follows the bits a weight (Q5_K 28% more than raw 4-bit, Q6_K 60%), because a
narrow step is bound by how fast the weights can be read.

![How a weight reaches the multiplier: the MLX 4-bit pack goes to the accelerator as stored and is corrected per group afterwards; a GGUF block is decoded to 16-bit weights in scratch memory first, so nothing needs correcting.](figures/weights.svg)

**The file is laid out in the order the GPU reads it.** Weights are stored as tiles of 128
output rows, group by group, so each thread group streams one contiguous block. The file is
mapped once and used in place.

**128 prompt tokens a step, in tiles of 32 by 128.** That shape gives each group of GPU
threads the 32 x 32 block of results it handles best. At this width the step is limited by the
accelerator's own rate, not by memory: making steps wider (up to 2,048 rows) changed nothing.

**Kernels are kept small.** The same arithmetic ran 7% faster as a kernel with nothing
optional in it than as one mode of a general kernel, and a single run-time flag cost 20%. What
a kernel carries, not only what it executes, decides how many copies the GPU keeps running.

**Stages are removed, not tuned.** The matrix product that updates the residual stream also
writes the normalised input the next product needs, so the normalisation pass between them is
gone. The gated-delta layers run as three dispatches. Removing a stage is worth more than
speeding one up.

**Attention is one kernel on the accelerator.** Keys and values are stored as 8-bit codes with
one scale per vector (32.5 KiB a token; 4-bit was tried and was worse in both accuracy and
speed). For each block of eight rows the kernel computes scores against 64 tokens at a time,
turns them into probabilities with the softmax laid out so that sums and maxima stay in
registers, and multiplies by the values, without leaving the GPU's fast path. The context is
split into spans that run side by side and are merged afterwards. The cost is about 3 ms per
thousand tokens of context for a 128-row step.

**Rows nobody will read skip the last layer.** Only the final prompt token needs an output.

**The GPU is not allowed to fall asleep on a conversation.** Weights and scratch memory are
pinned resident, and while the server is idle a tiny pass touches the KV pool twice a second.
Without that, the first step after an agent's tool call waited about a quarter of a second.

**Nothing is evaluated twice.** This matters more than any kernel: in a day of agent traffic
91% of prompt tokens had been seen before.

- Each conversation keeps its state in a slot, and the next request evaluates only what was
  added.
- The model's recurrent layers cannot be rewound, so the state is saved where each prompt
  opens the reply. If the client sends the last reply back differently (without its thinking,
  say), the conversation carries on from there instead of from the start.
- The state at the end of the system prompt is kept, so a new agent with the same
  instructions and tools starts from there. One agent's work never costs another its context.
- Conversations are written to disk and read back at about a second per 50K tokens, across
  restarts and evictions.

![A conversation's prompt as one strip: the system prompt, earlier turns and last reply are kept from earlier work and only the new tool result is evaluated; conversations are saved to disk, and 91% of prompt tokens in a day of agent traffic had been seen before.](figures/reuse.svg)

## Decode

**Speculative decoding, sized to the hardware.** A small draft model (five layers, reading the
big model's own hidden states) proposes a block of tokens in 7 ms. The big model then checks
the whole block in one pass. A pass that checks 16 rows costs about the same 62 ms as one that
produces a single token, so blocks are 16 long: the eight the draft model was trained to
produce, kept isolated so they are exactly what it would have said alone, and eight more that
count when the first eight are all right. A step yields about 3 tokens on prose, 4-5 on new
code, and 7-10 when the reply restructures code it has been given.

**Copying beats guessing.** When the reply is repeating text already in the context (editing
a file, quoting a tool result), the best draft is the text itself. Splosh keeps an index of
every three-token run in the conversation and proposes what followed last time: about 15
tokens a step, with no draft pass at all.

![One speculative-decoding cycle: a draft model proposes 16 tokens in 7 ms, the big model checks them in one 62 ms pass, the first few are accepted, its own token replaces the first wrong guess and the rest are discarded; a shorter path copies the block from the context.](figures/decode.svg)

**A wrong guess costs one step and nothing else.** Rows being checked do not write the
recurrent state; they write a journal, and only the accepted part is folded in. Nothing has to
be recomputed or rolled back.

**The output is the big model's.** Every token is verified. With greedy decoding it is the big
model's own choice; with sampling, the accept-or-reject rule preserves its distribution
exactly.

**The draft model gets the same care as the main one.** Its weights are quantised at load into
the same tiled 4-bit layout (17 ms a pass became 7), with each group's range fitted by least
squares instead of taken from its extremes, which the draft repays with 2% more accepted
tokens.

## Many sessions at once

One thread owns the GPU. Each step it packs rows from every live session into a single pass
over the weights: a block to verify for each session that is writing, and prompt rows for
those still reading. The weights are read once however many sessions share the step: a step
for eight sessions takes about 210 ms where a step for one takes 80, so eight agents together
get about three times what one gets alone, not one eighth each.

![Rows from eight conversations are packed into one 128-row step that reads the weights once: a step for eight takes about 210 ms against 80 ms for one, so eight agents together get about three times what one gets alone.](figures/shared-step.svg)

Steps come in widths the accelerator likes (16, 32, 64, 96, 128 rows). The scheduler picks the
width that makes the most progress per millisecond, counting what each session's drafts have
been yielding. A prompt that is waiting gets a full-width step with the writers riding in it.

How a step is shared between writers and waiting prompts can be changed with `decodeWeight`:
see [Settings](server.md#settings).

## What did not work

Kept here so nobody tries them again: wider prefill steps, taller tiles, a second verified
branch, 4-bit KV, fusing the MLP's two products with its activation, attention tiles of ten
rows, skipping attention chunks that look negligible (0.2% qualify on a real 85K prompt),
waking the GPU when a request arrives, and relaxed arithmetic. The measurements are in
[`audit/ENGINE-PERFORMANCE-RESEARCH.md`](../audit/ENGINE-PERFORMANCE-RESEARCH.md), which is the
full log of what was tried and what each thing was worth.

To time a change of your own, see [Measuring it yourself](development.md#measuring-it-yourself).
