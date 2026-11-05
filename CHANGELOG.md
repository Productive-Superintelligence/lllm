# Changelog

## 0.3.0

One project: LLLM now includes the package hub and the SSSN and AAAX
modules, with a single `lllm` command.

**Tactic**

- `from lllm import Tactic` is the program class again: write `call()` (or
  the protocol hook `_run()` or `_arun()`), declare `agent_group` for native
  agents built per call, use `Tactic.quick()` for a zero-config agent,
  `bcall`/`ccall` for batches. It subclasses the runtime-neutral
  `lllm.protocol.Tactic`, so call()-style tactics are served, packaged, and
  used as tools directly; `NativeTacticAdapter` is optional.
  `input_model`/`output_model` alias `input_type`/`output_type`, and a
  subclass's own declaration wins over the types it inherits.
- `tactic(x)`, `acall`, `bcall` and `ccall` run `call()` when a subclass
  writes it, else its `_run()` (or `_arun()`) hook. An `_arun()`-only tactic
  gets `run()` too, an `async def call()` is awaited on every entry point,
  and an overridden `events()` (or `stream()`) streams through `aevents()`
  and `/stream`. A hook that hands its call back (`super()._run()`,
  `self.call()`) raises `TacticUnsupportedError` at once instead of
  recursing.
- `call()` and every stream from an overridden sync `events()`/`stream()`
  run on a per-call copy with fresh agents and their own observed session,
  so repeated `/stream` requests start with clean dialogs; protocol hooks
  and async stream overrides run on the instance, like any protocol tactic
  (with `agent_group` they warn). Sync streams are pulled one item at a
  time on a worker thread: they never block the event loop or run ahead of
  the reader, and closing a stream (or a client disconnecting) closes the
  generator. A stream closed early leaves its session `"cancelled"`.
- Async `call()` programs are awaited on the caller's loop by `acall`,
  `arun`, `ccall` and a service's `/run`, and run on the calling thread's
  own event loop (kept per thread, fresh after a fork) from `tactic(x)`,
  `run()` and `bcall`, so clients they keep between calls (an
  `httpx.AsyncClient`) stay usable. `await agent.arespond()` calls an agent
  from async code without blocking the loop. A cancelled or interrupted call
  leaves its session `"cancelled"`.
- Sub-tactic calls roll up into the caller's session and cost. Sessions and
  sub-tactic records use the instance name, so `Tactic(name=...)` wins over
  the class `name`.
- The native vocabulary is importable from `lllm` again (`Prompt`, `Agent`,
  `Dialog`, `tool`, `load_runtime`, `resolve_config`, `build_tactic`,
  `install_package`, `BaseProxy`, ...).
- Pydantic AI: `tactic_as_tool()` and `Tactic.as_tool()` give agents typed
  parameters in task and kwargs mode. `PydanticAITactic` validates input on
  its stream and event paths too, sends JSON events to `/stream`, and takes
  its description only from text you wrote. `result_mode="result"` no
  longer needs an `output_type`: it returns the raw run result, unvalidated.
  Keep such tactics in process: when served, `/run` returns the whole run,
  message history included.
- Services: a single-tactic app's `/openapi.json` documents the tactic's
  input and output schemas, and falls back to FastAPI's own document when
  they cannot be added. Custom GET and DELETE endpoints called without a
  body get the query parameters (a dict, or `None`), unvalidated.
- Fixes: an explicit `runtime_kind` is inherited by subclasses, an Enum
  given as a tactic name becomes its value, and `RemoteTactic(url)` without
  `name=` works for host:port URLs (`127.0.0.1-8000`).

**Native runtime**

- Agents keep their own tools (system prompt, config `tools:`, proxies,
  skills) on every turn; `Prompt(agent_tools=False)` opts out.
- Proxies call their APIs end to end: `CALL_API` resolves documented
  endpoint paths (with `{param}` templates), adds API keys, and sends the
  request; only declared endpoints are reachable. Each path value is
  URL-encoded as one segment (`BRK/A` becomes `BRK%2FA`); a missing or
  empty value, `.` or `..` raises `ValueError` before any request is sent;
  placeholders such as `{series-id}` work.
- Inference hints: agents can send session identity (one dialog, one
  session; forks name their parent), priority, expected output length, and
  KV-cache intent to NVIDIA Dynamo, vLLM, and SGLang.
- Tool schemas follow type annotations; strict mode is chosen automatically,
  and never for object parameters. Recursive Pydantic models give small
  schemas: each model in a cycle is spelled out once, and later references
  back into the cycle become `{"type": "object", "title": "<Model>"}`.
- Rate-limit retries back off and stop after `max_rate_limit_retries`.
- `Prompt.from_dict()` and `Dialog.from_dict()` never import code: a saved
  format's model is used only if its module is already loaded; otherwise
  the stored JSON schema is kept.
- Fixes: Python 3.10 support, package loading with Python files in custom
  sections or `services/`, prompts without variables, JSON-safe dialogs,
  `load_proxy("pkg:name")`, `export_package` names, Jupyter `CALL_API`,
  skill descriptions with braces, prompts copied with `model_copy()`,
  custom sections named like hub sections (`[schemas]`, ...) that set
  `paths`, and `tactic_as_function(parameter_mode="kwargs")` for inputs
  without fields.

**Packages and hub**

- `lllm.toml` is the one package manifest: native discovery and hub
  metadata in the same file (formerly `psi.toml`); refs are `lllm://`.
- `lllm.hub`: validation, local hub, cards, agent cards, config templates,
  hub API, and `lllm hub index` for static catalogs that any web server can
  host. The LLLM Hub website at lllm.one/hub is a separate project (the
  lllm-hub repository).
- Validation checks a section's fallback folder like declared `paths`, and
  READMEs that are not UTF-8 no longer break `lllm hub index`.
- Serving: a service `entry` that fails to import or build, or builds
  something other than an app, a tactic, or a data payload, is a
  `LaunchError` instead of serving the declared tactic without the
  service's auth. Served packages keep their modules loaded, so tactics can
  import them when called; packages served in one process need distinct
  top-level module names. A `Tactic` class gets `name=` only if its
  constructor takes it.
- `lllm create my-app` brings back the native project templates (minimal,
  pipeline, proxy, research, service); generated projects validate and
  serve as created. The proxy template's endpoints take the params dict
  `CALL_API` sends, and `main.py --catalog` lists them.

**Modules**

- `lllm.network` (SSSN, experimental): channels, events, subscriptions,
  artifacts, snapshots, stores, and `TacticProcessor`.
- `TacticProcessor`: `errors=` also covers building and writing the output.
  Reusing a subscription that reads another channel or kind raises
  `ValueError`, and `process()` rejects async stores with a `TypeError`
  (use `aprocess()`).
- Stores: a duplicate event id raises `EventExistsError` (409 over HTTP),
  `put_snapshot()` takes the same arguments on every store, and custom
  store endpoints get query parameters through a `params` argument.
- `lllm.shell` (AAAX, experimental): strategies built from lllm services. A
  strategy has one channel store, served at `/network/` and opened only
  when the strategy is served or a channel is used; channel resources
  without a handler are backed by it, and attaching a different store
  raises `ValueError`. Shell routes return the services error envelope (a
  lookup that fails inside a handler is a 500, not a 404), and the
  `/channels/` routes serve only channels.
- `lllm.Resolver` resolves tactic, channel, and snapshot refs from
  `.lllm/config.toml`; a channel binding's `store` can name a
  `[stores.<name>]` table.

**CLI**

- One command: `lllm create | init | setup | inspect | serve | hub |
  network | shell`. `lllm serve` and `lllm inspect` accept package paths
  and check declared API keys; `lllm setup` stores them, also for library
  packages with nothing to launch.
- `lllm serve` lists what a package can launch when the selected resource
  (a config, a snapshot, ...) cannot be launched, and `lllm shell inspect`
  reports an invalid strategy name as a usage error.

**Behavior changes**

- `tactic.call(x)` on a tactic that implements only `_run()` or `_arun()`
  raises `TacticUnsupportedError`; run it with `tactic(x)`, `run()`, or
  `arun()`.
- Tactic names must be single path segments: a name containing `/`, `:`,
  `\`, `;` or `%` raises `ValueError` when the tactic is created.
- A failing native tactic call logs one INFO line (traceback at DEBUG), not
  an ERROR with a traceback; the exception still reaches the caller.
- Tactic entrypoints (`lllm serve module:attr`, package tactics, shell
  packages) tell factories from handlers by one rule. A factory is a class
  that can be built without arguments, a callable with no parameters, or a
  callable that can be called without arguments and is annotated to return
  a `Tactic` or a FastAPI/Starlette app; it is called once at load.
  Anything else is a handler, called per input, so `def greet(name="world")`
  now answers each input instead of running at load.
- `lllm hub validate` rejects raw secret-shaped metadata keys (`api_key`,
  `token`, `password`, `privateKey`, `proxyAuthorization`, ...) with
  `metadata_secret_key`, so `lllm hub publish` refuses such packages; use a
  reference such as `api_key_ref`.
- A hook-only tactic (`_run()`/`_arun()`) runs on the instance under
  `tactic(x)`, `acall`, `bcall` and `ccall` too, not on a per-call copy.
- `tactic(x)` on an `async def call()` returns the result instead of a
  coroutine: in async code, `await tactic.acall(x)`.
- Static hub catalogs write zips to `packages/<org>/<name>/<version>.zip`
  (was `packages/<org>-<name>-<version>.zip`), with the package files at
  the archive root instead of under a folder; `install_package()` reads
  both layouts.
- `TacticProcessor` is at-least-once: an event is acknowledged only after
  its output is written. With the default `errors="raise"`, a failing event
  stays unacknowledged and the next `process()` call runs it again; with
  `errors="record"`, it is acknowledged once its error event is written,
  and is not retried. After an interrupted run the tactic can see an event
  again, and the sink can hold a repeat.
- Default `TacticProcessor` consumers are `<tactic name>-<source>-processor`,
  or `<tactic name>-<source>-<kind>-processor` with a kind filter. A new
  consumer starts from the channel's first event, so a processor that ran
  under another id processes the channel's history again unless you pass
  that id as `consumer=`. Pass `consumer=` also when the default would
  contain whitespace, `?` or `#`.
- The shell's `POST /channels/{name}/events` returns 400 and appends
  nothing when `input` is missing or not a JSON object; `{"input": {}}`
  still appends an empty event.
- Shell tactic resources run as tactics on `/resources/{name}/invoke` too:
  they get a `CallContext` built from the request context, not the raw
  dict, and fail as on `/tactics/{name}/run` (a `ValueError` or
  `HTTPException` is a 500). A plain function that returns a coroutine is
  not awaited on either route; write it as `async def`.

## 0.1.0

- Added the runtime-agnostic `Tactic` protocol.
- Added plain Python, Pydantic AI, and native runtime adapters.
- Added FastAPI service wrappers, clients, custom endpoints, and SSE streaming.
- Added parser, proxy, and sandbox utilities outside the native runtime.
- Added optional LLLM hub metadata helpers and `lllm create` scaffolding.
