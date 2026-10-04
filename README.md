# Splosh

A Swift and Metal inference engine for **Qwen3.8-27B** (the MLX 4-bit pack, or Unsloth's GGUF
files at their own size) on Apple silicon, with an OpenAI-compatible server built for what
agent harnesses actually send: several long conversations at once, each growing by a tool
result at a time. It is inspired by ideas in [Splash](https://github.com/incoai/splash) and
[Splish](https://github.com/publicExcess/splish).

Running it gives you two things on one local port:

- **The model**, behind the OpenAI chat API, so any OpenAI-compatible client or agent harness
  can use it.
- **A web page that shows what the server is doing**, in detail and as it happens: each
  session's context and how far its prompt has been read, the reply as it is being written,
  the rates, the memory in use, the conversations it has cached and where the time of each
  step goes. A second page holds the settings and the models.

It exists to be fast at two things: reading long prompts (prefill) and writing replies
(decode). This page says how to get it running, and then why it is as fast as it is.

<p align="center">
  <a href="https://buymeacoffee.com/andysteafund"><img src="https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20tea&amp;emoji=%F0%9F%8D%B5&amp;slug=andysteafund&amp;button_colour=FFDD00&amp;font_colour=000000&amp;font_family=Cookie&amp;outline_colour=000000&amp;coffee_colour=ffffff" alt="Buy me a tea"></a>
  <br>
  <sub>Splosh is free. If it runs well on your Mac, a tea keeps the work on it going.</sub>
</p>

## Contents

- [What it does on an M5 Pro](#what-it-does-on-an-m5-pro): the measured rates
- [Getting it running](#getting-it-running): from a Mac with nothing installed to a served model
- [Models](#models): the four there are, how to get one, changing between them
- [The server](#the-server): the API, the dashboard, the models page, the settings
- [Why it is fast](#why-it-is-fast)
- [Measuring it yourself](#measuring-it-yourself)
- [Where things are](#where-things-are)
- [Limits](#limits)
- [Credits](#credits)
- [Licence](#licence)
- [Support](#support)
- [See also](#see-also)

## What it does on an M5 Pro

Measured 2026-10-03 on an M5 Pro (20-core GPU, 64 GB), High Power mode, against Splash 1.1.0
(`incoai/splash`) and Splish (a tuned fork of it). The model is the same 4-bit Qwen3.8-27B in
all three: Splosh ran the `mlx-community/Qwen3.8-27B-4bit` pack, and Splash and Splish ran
`incoai/Qwen3.8-27B-Splash`, Inco's package of that pack. Each engine alone in memory; every
prompt new, so no cache helps; greedy, thinking off, an 800-token essay as the reply; two
passes in opposite orders, mean shown.

| Prompt (4-bit weights) | Prefill, tokens/s (Splosh / Splish / Splash) | Decode, tokens/s (Splosh / Splish / Splash) |
|---|---|---|
| 10.7K tokens | 467 / **484** / 450 | **35.9** / 35.6 / 30.8 |
| 54.5K | 388 / **394** / 364 | **32.8** / 32.3 / 27.0 |
| 165.7K | **276** / 264 / 251 | **25.8** / 24.7 / 23.8 |
| 221.3K | **241** / 228 / 224 | **23.5** / 22.9 / 21.0 |

| Four 21K prompts sent together (4-bit weights) | Splosh | Splish | Splash |
|---|---|---|---|
| All four replies finished | **251 s** | 270 s | 282-285 s |
| Every token, prompt and reply, over that time | **354 /s** | 330 /s | 313-316 /s |

Essays are the slowest thing to decode, because a draft model guesses prose badly. On the same
4-bit pack, code runs at 92-98 tokens/s for one session at short context, and eight agents
writing code at once (about 23K tokens of context each) were generating 156-175 tokens/s
between them in a live run.

With Unsloth's Q5 file (`UD-Q5_K_M`, 18.2 GiB of weights) and more than 150K tokens of context,
a session writing code was seen at about 85 tokens/s: 84.4 on the dashboard at 171K tokens,
with 13.9 tokens a step and 99% of drafts accepted.

These are one machine's figures. How they were taken, and what else was tried, is in
[`audit/SPEED-PLAN-RESULTS.md`](audit/SPEED-PLAN-RESULTS.md); the harness is
`tools/bench/engine-report`.

## Getting it running

From a Mac with nothing on it to a model being served is five steps. Steps 1 and 4 are the
ones that take time: each is a large download.

| | Step | What it does |
|---|---|---|
| 1 | [Install Xcode](#1-install-xcode) | the compilers for Swift and for the Metal kernels |
| 2 | [Get the code](#2-get-the-code) | `git clone` |
| 3 | [Build](#3-build) | the kernels, then the `splosh` program |
| 4 | [Start the server](#4-start-the-server) | on its first launch it offers the models, and downloads and installs the one you pick |
| 5 | [Use it](#5-use-it) | point a client at the API, and open the dashboard |

**What the Mac needs**

- A GPU with the neural accelerator that Metal 4's tensor operations run on. Splosh was
  developed and measured on an M5 Pro only; nothing else has been tried.
- macOS 27.
- Memory: 14.1 GiB for the default model's weights (15 to 21 GiB for the others, under
  [Models](#models)), 154 MiB of recurrent state per conversation slot, and 32.5 KiB of KV
  cache per token of context, taken from a shared pool as contexts grow. On 64 GB the pool
  holds about 570K tokens across all conversations with the default model.
- Disk: about 55 GB free to install the default model; 36 GB of it stays in use, or 20 GB once
  the downloaded pack is deleted.

### 1. Install Xcode

Install Xcode 27 from the App Store and open it once, so that it finishes installing itself.
It brings Swift 6.4, `git` and `make`. The Metal compiler is a separate component, fetched from
a terminal:

```bash
xcodebuild -downloadComponent MetalToolchain
```

### 2. Get the code

```bash
git clone https://github.com/ajaxharg/Splosh.git
```

```bash
cd Splosh
```

Everything below is run from this directory: the server looks for its settings, its models and
its downloads under it.

### 3. Build

```bash
make shaders && swift build -c release --disable-sandbox
```

If anything is missing, `.build/release/splosh doctor` checks the machine and the toolchain
and says what to do about each thing it finds.

### 4. Start the server

```bash
.build/release/splosh serve
```

There is no model yet, so the server starts without one, says so, and opens its page in your
browser. Choose a model in either place:

- **In the browser**, at `http://127.0.0.1:8091/`. The page lists the models with what each
  costs to download and to hold in memory; press **Download** on one. `mq4` is the one to start
  with.
- **In the terminal** the server is running in, press Enter for the default (`mq4`), or type
  another model's name and press Enter.

Splosh then does the rest, naming each step in the terminal and on the page with its progress:

```mermaid
flowchart LR
    serve["splosh serve"] --> have{"a model<br>installed?"}
    have -- no --> choose["choose one:<br>the page, or Enter<br>in the terminal"]
    choose --> download["download<br>from Hugging Face"]
    download --> check["check each file's<br>SHA-256"]
    check --> convert["convert for<br>the engine"]
    convert --> draft["fetch the<br>draft model"]
    draft --> load["load the model"]
    have -- yes --> load
    load --> ready["API and dashboard<br>on port 8091"]
```

The download for `mq4` is 20 GB in all (the weights and the draft model). If it is stopped, or
the connection drops, it carries on from where it got to the next time it is asked for. When
the model is loaded, the page in the browser becomes the dashboard. The next `splosh serve`
finds the model installed and loads it straight away.

If you already have the model's files, or would rather fetch them yourself, see
[Files you already have](#files-you-already-have).

### 5. Use it

The server listens on `127.0.0.1:8091` only. Point any OpenAI-compatible client at
`http://127.0.0.1:8091/v1` with model `qwen3.8-27b`:

```bash
curl http://127.0.0.1:8091/v1/chat/completions -H 'content-type: application/json' -d '{"model":"qwen3.8-27b","stream":true,"messages":[{"role":"user","content":"Write a Python function that merges overlapping intervals."}]}'
```

Open `http://127.0.0.1:8091/` to watch it work. Ctrl+C in the server's terminal stops it:
requests in flight finish, and every conversation is written to disk for the next start.

## Models

Everything Splosh runs is one model, Qwen3.8-27B, at four precisions. One is in memory at a
time, and the server changes between the ones that are installed.

| Name | Weights | Download | In memory | Bits a weight | Weight error | Decode against `mq4` |
|---|---|---|---|---|---|---|
| `mq4` | MLX 4-bit pack, `mlx-community/Qwen3.8-27B-4bit` | 16.1 GB | 14.1 GiB | 4.5 | 9.3% | the default |
| `uq4` | Unsloth `UD-Q4_K_M` | 16.5 GB | 15.0 GiB | 4.79 | 6.7% | not timed |
| `uq5` | Unsloth `UD-Q5_K_M` | 19.8 GB | 18.1 GiB | 5.77 | 3.8% | 0.83 on code, 0.71 on prose |
| `uq6` | Unsloth `UD-Q6_K_M` | 23.1 GB | 21.2 GiB | 6.76 | 2.2% | not timed |

**Which one.** Start with `mq4`: it is the smallest, the fastest to decode, and the one the
figures at the top of this page were measured on. The Unsloth files, from
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), run at the size
they are on disk and buy accuracy with bytes: more bits a weight means more to read on every
step, so decode slows roughly in proportion while prefill barely changes.

Weight error is the RMS difference from the MLX 8-bit pack, relative, over a sample of rows
from every fourth layer (`tools/gguf_check.py <file.gguf> --reference mlx-q8`). It is a plain
measure of the weights: it gives no credit for Unsloth's calibration, which spends its
precision where the outputs are most sensitive.

### Getting one

All of these start the same install, and each shows its steps and progress:

| From | How |
|---|---|
| The server's first page | with no model installed, `http://127.0.0.1:8091/` lists the four: press Download |
| The server's terminal | on that first launch, Enter installs the default; a model's name and Enter, that one |
| The models page | `http://127.0.0.1:8091/models`, under "Download models", at any time: another model can be added while one is serving |
| A shell | `.build/release/splosh download uq5`; with no name, the default |

`splosh download --list` says which models are installed. What an install does:

1. **Tokenizer**: the three small files that turn text into tokens and back, into
   `inputs/tokenizer`.
2. **Weights**: downloaded from Hugging Face into `inputs/`, at the revision Splosh was
   measured on. A download that is stopped keeps what has arrived and carries on from there.
   Every file is checked against its SHA-256 before it is used.
3. **Convert**: the weights are written in the layout the engine maps straight into memory,
   under `models/` (for `mq4`, `models/q4/weights.tiled.splw`). An Unsloth file takes about
   half a minute.
4. **Register**: a `model.<name>` line in `splosh.toml`, so the server and its clients can ask
   for the model by name.
5. **Draft model**: the DFlash 2 checkpoint `incoai/Qwen3.8-27B-DFlash2` (3.8 GB, into
   `inputs/draft`), fetched once and shared by every model. Speculative decoding needs it:
   without it Splosh writes one token a step, about 16 tokens/s.

An install needs the download and the converted file on disk at once, and for `mq4` a second
converted file for a while: 48 GB for `mq4`, 33 to 46 GB for the others, plus the draft model.

A model installed while the server is running is listed on the models page and the dashboard
at once, and can be loaded from there or by naming it in a request.

### Where it is all kept

Everything is under the directory the server is run from. An install says where it has put the
model when it ends, and the models page shows the place with each model.

| What | Where | Needed afterwards |
|---|---|---|
| The converted model, the one file the server reads | `models/q4/weights.tiled.splw` for `mq4`; `models/gguf/ud-q4_k_m.splw`, `ud-q5_k_m.splw`, `ud-q6_k_m.splw` for the others; or wherever its `model.<name>` line in `splosh.toml` says | yes |
| What it was converted from | `inputs/mlx-q4/`, `inputs/gguf/` | no: it can be deleted, which leaves about the size in the table above |
| The tokenizer and the draft model | `inputs/tokenizer/`, `inputs/draft/` | yes |
| How each install stands, for the pages | `models/.downloads/` | no |

The converted models have a directory of their own, `models/`, and not the build's: cleaning
the build (`swift package clean` empties `.build`) leaves them alone. To keep a model somewhere
else, move the file and change its `model.<name>` line.

### Files you already have

Files you have downloaded yourself go where Splosh would put them, under the names they have on
Hugging Face. Then choose the model in any of the ways above: what is already there is checked
and not downloaded again. Files in the Hugging Face cache (`~/.cache/huggingface/hub`, where
`hf download` puts them) are found too.

| Model | Put the files here |
|---|---|
| `mq4` | `inputs/mlx-q4/`: `config.json`, `model.safetensors.index.json`, `tokenizer.json` and the three `model-0000N-of-00003.safetensors` |
| `uq4` | `inputs/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` |
| `uq5` | `inputs/gguf/Qwen3.8-27B-UD-Q5_K_M.gguf` |
| `uq6` | `inputs/gguf/Qwen3.8-27B-UD-Q6_K_M.gguf` |
| the draft model | `inputs/draft/model.safetensors` |

`splosh download --list` prints the same places as whole paths. A file of yours that is not
the one Splosh was measured on (another revision, say) is used as it is, with a line saying so.

### Changing between them

`splosh.toml` registers the models by name, and the server can change between the ones that
are installed. Only one is in memory at a time.

```
model = "uq5"
model.mq4 = "models/q4/weights.tiled.splw"
model.uq4 = "models/gguf/ud-q4_k_m.splw"
model.uq5 = "models/gguf/ud-q5_k_m.splw"
model.uq6 = "models/gguf/ud-q6_k_m.splw"
```

- `model` is the one the server starts on; `splosh serve --model uq6` overrides it. If that
  one is not installed, the server starts on the first that is.
- **A chat request that names another registered model has it loaded** in place of the one in
  memory. Nothing running is cut: the request waits until the loaded model's requests have
  finished, the engine is replaced, and the request is answered by the new model. A switch
  took 10-14 s in a trial. A name that is not registered (`qwen3.8-27b`, say) is served by
  whichever model is loaded, so existing clients need no change.
- **Pick from a list** on the models page or in the dashboard's header: each model is shown
  with its size and file, and Load switches to it.
- `splosh models` prints the list and marks the loaded one; `splosh models --load uq4`
  switches from the command line. `GET /v1/models` returns the same list.
- A model that is still busy after `switchWaitSeconds` keeps its place and the request for the
  other gets a 503 to retry; a model that fails to load gives way to the one before it.
- Conversations on disk are kept per model, and found again when the server comes back to it.

![Changing model: a request naming model B reaches the supervisor, which keeps it until model A's requests have finished, then the engine process is replaced by one with model B, which answers the kept request.](docs/figures/model-switch.svg)

### What has been measured

2026-10-03, on the M5 Pro:

- **`uq5` is correct.** Its greedy continuation equals llama.cpp's on the same file, token for
  token, for 64 and 96 tokens, with the prompt read through each of the three kernel shapes
  (`tools/gguf-compare`). `uq4` and `uq6` load and answer sensibly but have not been compared.
- **`uq5` against `mq4`**, one session through `splosh generate`, the same binary, power mode
  Automatic (so both sides are below the High Power figures above), runs alternated: code with
  speculation 61 against 73 tokens/s; an essay 22 against 31; a 9.6K-token prompt 319-388
  against 268-362 tokens/s. A 16-row decode step takes 121 ms against 87. `uq4` and `uq6` have
  not been timed.

The GGUF formats understood are Q4_K, Q5_K, Q6_K, Q3_K, Q8_0, IQ4_NL, IQ4_XS and IQ3_S, which
is everything in the three files above; a file holding any other is refused, with the tensor
named. The MTP block these files carry is not used.

### By hand

`splosh download` is these commands run for you; they are here for a file it does not know, or
an artifact wanted somewhere else.

<details>
<summary>Fetching and converting without <code>splosh download</code></summary>

The tokenizer, into `inputs/tokenizer`, and the MLX pack, whole, into `inputs/mlx-q4` (the
converter reads the pack's own `config.json` and `tokenizer.json` beside its weights).

```bash
python3 tools/fetch_inputs.py --fetch
```

```bash
hf download mlx-community/Qwen3.8-27B-4bit --revision 3e6447f082e89cc7f0bc6e5441afd38dfce760ff --local-dir inputs/mlx-q4
```

Convert it from the repository root: once to Splosh's format, once more into the tiled layout
the engine runs from.

```bash
.build/release/splosh convert --input inputs/mlx-q4 --out models/q4/weights.splw --verify
```

```bash
.build/release/splosh convert --retile --input models/q4/weights.splw --out models/q4/weights.tiled.splw
```

A GGUF file. Conversion only re-orders the codes into the layout the kernels read; the result
is the size of the source.

```bash
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q5_K_M.gguf --local-dir inputs/gguf
```

```bash
.build/release/splosh convert --input inputs/gguf/Qwen3.8-27B-UD-Q5_K_M.gguf --out models/gguf/ud-q5_k_m.splw
```

Then register the artifact as a model (a `model.<name>` line in `splosh.toml`), or, with no
models registered, point `weightsPath` at it.

</details>

`tools/download-smoke` tests the downloads, the first launch and the pages against a stand-in
for Hugging Face, and `tools/switch-smoke` the changing between models; neither loads a model.

## The server

`splosh serve` listens on `127.0.0.1:8091` and gives you an API for clients and two pages for
you.

**The API**

- `POST /v1/chat/completions`, streaming or whole, with tools, `stop`, `temperature`, `top_p`,
  `top_k`, `seed`, and `reasoning_effort` (`none` turns thinking off). Text only: no images.
- `GET /v1/models`: every model the server can load, the loaded one first (see
  [Changing between them](#changing-between-them)). `GET /v1/stats` for everything the
  dashboard shows, as JSON.

**The dashboard**, at `http://127.0.0.1:8091/`, is the detailed view of what the server is
doing now:

- each session's context, how far its prompt has been read, and its rates;
- the reply being written: rest the pointer on a session to watch it in a small window that
  can be made larger; a click opens the large one. The window stays with the conversation from
  one request to the next. The text comes from `GET /v1/sessions/<id>/reply`;
- memory, divided into weights, KV cache, session state and checkpoints;
- the conversations cached for later, the time spent outside the GPU's steps, and step times
  by width.

**The models page**, at `/models`, is where a model is loaded in place of the one in memory,
where more are downloaded, and where the model settings of `splosh.toml` are. **The settings
page**, at `/settings`, edits the rest of `splosh.toml` and can restart the engine. Each has a
link to the other and to the dashboard.

Started from a terminal, the server opens its page in the browser once it is listening: the
dashboard, or on a first launch the models to download. `splosh serve --no-open`, or
`openBrowser = false`, leaves the browser alone.

**Stopping and restarting**

- `splosh serve --restart`, from another terminal: the engine is replaced (a new build, new
  settings) while the port stays open. Requests in flight finish, new ones wait, and every
  conversation is picked up from disk where it was.
- Ctrl+C lets requests finish and writes every conversation to disk before exiting.

![The dashboard with Unsloth's Q5 file loaded, while one conversation at 164K tokens of context is being answered: decode at 36 tokens/s, memory divided into weights, KV cache, session state and checkpoints, the session's row with its context, tokens per step and share of drafts accepted, and the prefixes cached for other conversations.](docs/figures/dashboard.png)

### Settings

All optional, in `./splosh.toml` or the file given to `--config`.

| Key | Default | Meaning |
|---|---|---|
| `port`, `host` | 8091, 127.0.0.1 | where to listen |
| `slots` | 8 | conversations that can hold state at once, running or cached |
| `concurrency` | as many as slots | most requests worked on at once; the rest queue |
| `contextWindow` | 262144 | longest prompt accepted |
| `kvPages` | sized from memory | KV pool in 256-token pages, shared by all sessions |
| `draftPath` | `inputs/draft`, or the Hugging Face cache | `none` disables speculative decoding |
| `prefixCacheDir` | `~/Library/Caches/Splosh/prefix-cache` | conversations on disk; `none` disables |
| `prefixCacheGiB` | 16 | disk budget; least recently used go first |
| `decodeWeight` | 1 | how a step is shared between sessions that are writing and prompts that are waiting. 1 gives a waiting prompt a full-width step with the writers riding along (best for agent batches); 20 or more leaves a running stream untouched |
| `answerReserve` | 8192 | tokens of a reply's limit kept for its answer: thinking that has used the rest is closed by the server, so an answer is still written. At most a quarter of the limit; 0 lets thinking run to the limit |
| `toolCallOverrun` | 8192 | tokens a tool call in progress at a reply's limit may run past it, so the reply ends on a whole call and an agent carries on; 0 ends every reply at its limit |
| `requestLog` | true | one line per finished request: tokens reused, evaluated, generated, rates, and where the time outside the GPU went |
| `openBrowser` | true | a server started from a terminal shows its page in the browser once it is listening |
| `model`, `model.<id>` | none | the models the server can load, and the one it starts on (see [Changing between them](#changing-between-them)) |
| `modelSwitch` | request | `request`: a chat naming another registered model has it loaded. `manual`: only `splosh models --load` does |
| `modelDwellSeconds`, `switchWaitSeconds` | 60, 120 | how long a model just loaded is kept before another may replace it, and how long a request for another model waits for the loaded one to go idle before it is refused |

`splosh cache stats` lists what is stored on disk and `splosh cache purge --all` clears it.
`tools/serve-smoke` is the end-to-end test: it loads the real model and checks, among other
things, that greedy output matches reference token ids from `mlx-lm`.

## Why it is fast

Three numbers about the machine decide almost everything.

- The GPU reads memory at about 290 GB/s, so **one pass over 14 GiB of weights takes 52 ms**.
  A step that produces one token can never beat 19 tokens a second.
- The GPU's neural accelerator multiplies 4-bit weights by 16-bit activations at about
  **14 trillion multiply-adds a second**, far beyond what ordinary shader arithmetic manages.
- Every point in a step where one piece of work must wait for another costs **40-60
  microseconds**, and a step has hundreds.

So prefill has to keep the accelerator saturated and waste nothing around it, and decode has
to get more than one token out of each pass over the weights.

![One step through the model: rows pass through the embedding, 64 layers of three gated-delta layers then one attention layer repeated 16 times, a final norm and the output head; weights, recurrent state and the KV cache are read from memory.](docs/figures/step.svg)

### Prefill

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

![How a weight reaches the multiplier: the MLX 4-bit pack goes to the accelerator as stored and is corrected per group afterwards; a GGUF block is decoded to 16-bit weights in scratch memory first, so nothing needs correcting.](docs/figures/weights.svg)

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

![A conversation's prompt as one strip: the system prompt, earlier turns and last reply are kept from earlier work and only the new tool result is evaluated; conversations are saved to disk, and 91% of prompt tokens in a day of agent traffic had been seen before.](docs/figures/reuse.svg)

### Decode

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

![One speculative-decoding cycle: a draft model proposes 16 tokens in 7 ms, the big model checks them in one 62 ms pass, the first few are accepted, its own token replaces the first wrong guess and the rest are discarded; a shorter path copies the block from the context.](docs/figures/decode.svg)

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

### Many sessions at once

One thread owns the GPU. Each step it packs rows from every live session into a single pass
over the weights: a block to verify for each session that is writing, and prompt rows for
those still reading. The weights are read once however many sessions share the step: a step
for eight sessions takes about 210 ms where a step for one takes 80, so eight agents together
get about three times what one gets alone, not one eighth each.

![Rows from eight conversations are packed into one 128-row step that reads the weights once: a step for eight takes about 210 ms against 80 ms for one, so eight agents together get about three times what one gets alone.](docs/figures/shared-step.svg)

Steps come in widths the accelerator likes (16, 32, 64, 96, 128 rows). The scheduler picks the
width that makes the most progress per millisecond, counting what each session's drafts have
been yielding. A prompt that is waiting gets a full-width step with the writers riding in it.

### What did not work

Kept here so nobody tries them again: wider prefill steps, taller tiles, a second verified
branch, 4-bit KV, fusing the MLP's two products with its activation, attention tiles of ten
rows, skipping attention chunks that look negligible (0.2% qualify on a real 85K prompt),
waking the GPU when a request arrives, and relaxed arithmetic. The measurements are in
[`audit/ENGINE-PERFORMANCE-RESEARCH.md`](audit/ENGINE-PERFORMANCE-RESEARCH.md), which is the
full log of what was tried and what each thing was worth.

## Measuring it yourself

Timing on a laptop moves by 20-35% with heat and with what ran a minute ago. Compare
alternatives interleaved, more than once, and say whether a figure is a burst or sustained.

- `tools/bench/step-bench <context> "X=1" "SOME_SWITCH=1"`: engine steps, interleaved.
- `tools/bench/server-ab NAME=PORT NAME=PORT`: two running servers, alternating requests.
- `tools/bench/engine-report`: the whole comparison above.
- The request log and `/v1/stats` say, for real traffic, what was reused, what was evaluated,
  and where each request's time went.

## Where things are

| Path | What |
|---|---|
| `Sources/Shaders/` | the Metal kernels; `candidates/` holds variants, shipped and rejected |
| `Sources/Shaders/engine_gguf.metal`, `gguf_formats.h` | the GGUF kernels and their format decoders |
| `Sources/SploshModel/Gguf*.swift` | reading a GGUF file, its reference decoders, the converter |
| `Sources/SploshRuntime/Engine.swift` | one step: rows in, logits out |
| `Sources/SploshRuntime/BatchScheduler.swift` | sessions, slots, reuse, what goes in each step |
| `Sources/SploshRuntime/DraftModel.swift`, `LookupIndex.swift` | the two sources of drafts |
| `Sources/SploshRuntime/PrefixStore.swift` | conversations on disk |
| `Sources/SploshServer/` | the HTTP API, chat template, tool-call parsing, dashboard |
| `Sources/SploshCLI/` | `serve`, `download`, `models`, `generate`, `convert`, `cache`, `doctor`; the supervisor that holds the port and replaces the engine |
| `Sources/SploshCLI/ModelLibrary.swift`, `HubDownload.swift`, `DownloadCommand.swift` | the models that can be fetched, each file's pinned SHA-256, the resumed download and the install |

## Limits

One model, Qwen3.8-27B, in the four quantisations under [Models](#models), one loaded at a
time. One class of machine. Text and tool calls only. It binds to the loopback address and has
no authentication, so it is a local server, not something to expose. It has run under real
agent batches for days, not months.

## Credits

Splosh was built to be measured against Splash and Splish, and it learned from them: the
row-wise layout of the attention softmax follows Splash's, and timing Splash's prefill tile is
what showed that small kernels run faster. The engine for the MLX pack shares no code with
them. The GGUF path does: decoding a block of weights into scratch memory before each product
is how Splash and Splish run GGUF, and the layout of the re-ordered codes and the per-format
decoders in `Sources/Shaders/gguf_formats.h` are adapted from theirs (Apache-2.0).

- The GGUF files are [Unsloth's](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). The formats,
  the arithmetic that makes a weight of a code and the two tables (`kvalues_iq4nl`,
  `iq3s_grid`) are [llama.cpp](https://github.com/ggml-org/llama.cpp)'s (MIT).

- The model is [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); the weights are the
  `mlx-community` 4-bit pack.
- The draft checkpoint is Inco's `incoai/Qwen3.8-27B-DFlash2`. The draft model's arithmetic
  follows `dflash/model.py` in `z-lab/dflash` (Apache-2.0).
- Drafts copied from the context (`Sources/SploshRuntime/LookupIndex.swift`) are the copy rule
  of [Splish](https://github.com/publicExcess/splish), which has it from
  [TensorFold](https://github.com/ashhart/TensorFold): when the last tokens of a reply repeat
  earlier text, the draft is what followed them. The idea is theirs; the code is not.
- The HTTP server is [Hummingbird](https://github.com/hummingbird-project/hummingbird).

## Licence

Apache-2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE). The Apache-2.0 and MIT work named
under Credits keeps its own licence; its notices are in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

## Support

If Splosh is useful to you, you can put something in
[Andy's Tea Fund](https://buymeacoffee.com/andysteafund).

## See also

- [Splash](https://github.com/incoai/splash): Inco's inference engine for Apple silicon, the
  yardstick Splosh was measured against.
- [Splish](https://github.com/publicExcess/splish): a tuned fork of Splash. Its copy rule is
  where Splosh's drafts copied from the context come from.
- [TensorFold](https://github.com/ashhart/TensorFold): where Splish has the copy rule from.
