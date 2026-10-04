# The server

The API, the three pages, stopping and restarting, and every setting. Back to the
[README](../README.md).

- [The API](#the-api)
- [The dashboard](#the-dashboard)
- [The models and settings pages](#the-models-and-settings-pages)
- [Stopping and restarting](#stopping-and-restarting)
- [Settings](#settings)
- [Conversations on disk](#conversations-on-disk)

`splosh serve` listens on `127.0.0.1:8091` and gives you an API for clients and three pages
for you.

## The API

- `POST /v1/chat/completions`, streaming or whole, with tools, `stop`, `temperature`, `top_p`,
  `top_k`, `seed`, and `reasoning_effort` (`none` turns thinking off). Text only: no images.
- `GET /v1/models`: every model the server can load, the loaded one first (see
  [Changing between them](models.md#changing-between-them)).
- `GET /v1/stats` for everything the dashboard shows, as JSON.

## The dashboard

At `http://127.0.0.1:8091/`, the detailed view of what the server is doing now:

- each session's context, how far its prompt has been read, and its rates;
- the reply being written: rest the pointer on a session to watch it in a small window that
  can be made larger; a click opens the large one. The window stays with the conversation from
  one request to the next. The text comes from `GET /v1/sessions/<id>/reply`;
- memory, divided into weights, KV cache, session state and checkpoints;
- the conversations cached for later, the time spent outside the GPU's steps, and step times
  by width.

![The dashboard with Unsloth's Q5 file loaded, while one conversation at 164K tokens of context is being answered: decode at 36 tokens/s, memory divided into weights, KV cache, session state and checkpoints, the session's row with its context, tokens per step and share of drafts accepted, and the prefixes cached for other conversations.](figures/dashboard.png)

## The models and settings pages

**The models page**, at `/models`, is where a model is loaded in place of the one in memory,
where more are downloaded, and where the model settings of `splosh.toml` are. **The settings
page**, at `/settings`, edits the rest of `splosh.toml` and can restart the engine. Each has a
link to the other and to the dashboard.

Started from a terminal, the server opens its page in the browser once it is listening: the
dashboard, or on a first launch the models to download. `splosh serve --no-open`, or
`openBrowser = false`, leaves the browser alone.

## Stopping and restarting

- `splosh serve --restart`, from another terminal: the engine is replaced (a new build, new
  settings) while the port stays open. Requests in flight finish, new ones wait, and every
  conversation is picked up from disk where it was.
- Ctrl+C lets requests finish and writes every conversation to disk before exiting.

## Settings

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
| `model`, `model.<id>` | none | the models the server can load, and the one it starts on (see [Changing between them](models.md#changing-between-them)) |
| `modelSwitch` | request | `request`: a chat naming another registered model has it loaded. `manual`: only `splosh models --load` does |
| `modelDwellSeconds`, `switchWaitSeconds` | 60, 120 | how long a model just loaded is kept before another may replace it, and how long a request for another model waits for the loaded one to go idle before it is refused |

## Conversations on disk

`splosh cache stats` lists what is stored on disk and `splosh cache purge --all` clears it.
What is kept, and why, is under [Prefill](why-it-is-fast.md#prefill): nothing is evaluated
twice.
