# Models

The four models Splosh runs, how each is installed, where its files are kept and how the
server changes between them. Back to the [README](../README.md).

- [The four](#the-four)
- [Getting one](#getting-one)
- [Where it is all kept](#where-it-is-all-kept)
- [Files you already have](#files-you-already-have)
- [Changing between them](#changing-between-them)
- [What has been measured](#what-has-been-measured)
- [By hand](#by-hand)

## The four

Everything Splosh runs is one model, Qwen3.8-27B, at four precisions. One is in memory at a
time, and the server changes between the ones that are installed.

| Name | Weights | Download | In memory | Bits a weight | Weight error | Decode against `mq4` |
|---|---|---|---|---|---|---|
| `mq4` | MLX 4-bit pack, `mlx-community/Qwen3.8-27B-4bit` | 16.1 GB | 14.1 GiB | 4.5 | 9.3% | the default |
| `uq4` | Unsloth `UD-Q4_K_M` | 16.5 GB | 15.0 GiB | 4.79 | 6.7% | not timed |
| `uq5` | Unsloth `UD-Q5_K_M` | 19.8 GB | 18.1 GiB | 5.77 | 3.8% | 0.83 on code, 0.71 on prose |
| `uq6` | Unsloth `UD-Q6_K_M` | 23.1 GB | 21.2 GiB | 6.76 | 2.2% | not timed |

**Which one.** Start with `mq4`: it is the smallest, the fastest to decode, and the one the
figures in the [README](../README.md#what-it-does-on-an-m5-pro) were measured on. The Unsloth
files, from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), run
at the size they are on disk and buy accuracy with bytes: more bits a weight means more to read
on every step, so decode slows roughly in proportion while prefill barely changes.

Weight error is the RMS difference from the MLX 8-bit pack, relative, over a sample of rows
from every fourth layer (`tools/gguf_check.py <file.gguf> --reference mlx-q8`). It is a plain
measure of the weights: it gives no credit for Unsloth's calibration, which spends its
precision where the outputs are most sensitive.

## Getting one

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

## Where it is all kept

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

## Files you already have

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

## Changing between them

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

The settings named here are in the table under [Settings](server.md#settings).

![Changing model: a request naming model B reaches the supervisor, which keeps it until model A's requests have finished, then the engine process is replaced by one with model B, which answers the kept request.](figures/model-switch.svg)

## What has been measured

2026-10-03, on the M5 Pro:

- **`uq5` is correct.** Its greedy continuation equals llama.cpp's on the same file, token for
  token, for 64 and 96 tokens, with the prompt read through each of the three kernel shapes
  (`tools/gguf-compare`). `uq4` and `uq6` load and answer sensibly but have not been compared.
- **`uq5` against `mq4`**, one session through `splosh generate`, the same binary, power mode
  Automatic (so both sides are below the High Power figures in the README), runs alternated:
  code with speculation 61 against 73 tokens/s; an essay 22 against 31; a 9.6K-token prompt
  319-388 against 268-362 tokens/s. A 16-row decode step takes 121 ms against 87. `uq4` and
  `uq6` have not been timed.

The GGUF formats understood are Q4_K, Q5_K, Q6_K, Q3_K, Q8_0, IQ4_NL, IQ4_XS and IQ3_S, which
is everything in the three files above; a file holding any other is refused, with the tensor
named. The MTP block these files carry is not used.

How a GGUF weight reaches the multiplier is under
[Prefill](why-it-is-fast.md#prefill).

## By hand

`splosh download` is these commands run for you; they are here for a file it does not know, or
an artifact wanted somewhere else.

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
