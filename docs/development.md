# Working on Splosh

Where things are in the source, how to time a change, and the end-to-end tests. Back to the
[README](../README.md).

- [Where things are](#where-things-are)
- [Measuring it yourself](#measuring-it-yourself)
- [Tests](#tests)

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

What these do, and why they are made the way they are, is in
[Why it is fast](why-it-is-fast.md).

## Measuring it yourself

Timing on a laptop moves by 20-35% with heat and with what ran a minute ago. Compare
alternatives interleaved, more than once, and say whether a figure is a burst or sustained.

- `tools/bench/step-bench <context> "X=1" "SOME_SWITCH=1"`: engine steps, interleaved.
- `tools/bench/server-ab NAME=PORT NAME=PORT`: two running servers, alternating requests.
- `tools/bench/engine-report`: the whole comparison in the
  [README](../README.md#what-it-does-on-an-m5-pro).
- The request log and `/v1/stats` say, for real traffic, what was reused, what was evaluated,
  and where each request's time went.

How the README's figures were taken, and what else was tried, is in
[`audit/SPEED-PLAN-RESULTS.md`](../audit/SPEED-PLAN-RESULTS.md).

## Tests

- `tools/serve-smoke` is the end-to-end test: it loads the real model and checks, among other
  things, that greedy output matches reference token ids from `mlx-lm`.
- `tools/download-smoke` tests the downloads, the first launch and the pages against a
  stand-in for Hugging Face. It does not load a model.
- `tools/switch-smoke` tests the changing between models. It does not load a model.
- `tools/gguf-compare` compares a GGUF model's greedy continuation with llama.cpp's on the
  same file.
