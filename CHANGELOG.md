# Changelog

## 0.4.1

A bug-fix release, with new docs, `lllm.testing` for running native agents
without a model provider, and a few additions, such as
`allow_cross_site_requests`, `OutputSchemaError`, `Agent.extra_settings`,
and `RemoteTactic`'s `headers=`, `auth=`, and `contract=`. Some fixes
refuse requests or values that 0.4.0 accepted, raise where it did not, or
answer differently: notably browser requests and request bodies on served
tactics, a 500 for output that fails validation, `parsed` without its
`raw` key, the context manager's budget, and where the API keys you type
are stored. Each bullet says what changed and, where you need to act,
what to do. The LLLM APIs that sssn and aaax use keep their names and
signatures (`create_tactic_app` only gains `allow_cross_site_requests`),
but the changes below apply through them too, such as the cross-site
guard in `create_tactic_app`, the 500 that `status_code_for_error()` gives
for output that fails validation, and the rules for storing keys.

**Tactics, agents, and prompts**

- `format=` with a Pydantic model now validates the reply with
  `model_validate_json`, so a reply with valid JSON but wrong fields is
  repaired like any parse error, with a message that names each problem
  (`The reply does not match Triage: team: Input should be ...`). 0.4.0
  only decoded the JSON, so such a reply reached your code or failed the
  call at the tactic boundary. `parsed` is still the reply's decoded JSON
  object, unfiltered, so `Model.model_validate(parsed)` gives the model
  back (a strict model reads it with
  `Model.model_validate_json(json.dumps(parsed))`). Invalid JSON raises
  `ParseError` with the decoder's message, and a `dict` JSON schema as
  `format` needs a JSON object, where a list used to crash. Custom
  invokers that call `prompt.parse()` get the same.
- An accepted reply's `Message.parsed` no longer gets the reply text as a
  `raw` key: it is what the `format` or the parser produced (a non-dict
  wrapped as `{"value": ...}`), so an `output_model` with `extra="forbid"`
  accepts it, and `Prompt.parse()` with neither returns `{}`. 0.4.0 added
  `raw` to a parser's result; read the text from `message.content`.
  `DefaultTagParser` still returns its own `raw`, and a reply that fails
  to parse, as `invoke_traces` keeps it, still has
  `parsed={"raw": <reply text>}`.
- The `extra_settings` agent config key reaches the agent as
  `Agent.extra_settings` (empty by default); 0.4.0 parsed it and dropped
  it. Each agent gets its own copy of the plain data in it (dicts, lists,
  tuples, and sets, nested), and any other object, such as a lock or a
  client, is shared as is. A value that is neither a mapping nor null
  (null gives `{}`) raises `TypeError` when the config is read, and one
  nested more than 500 levels deep raises `ValueError`.
- `Tactic.quick()` keywords named after an `Agent` field, such as
  `max_llm_recall`, `context_manager`, `stream_handler`, or `name`, set
  that field; the others go to `model_args`, over an explicit
  `model_args=`. 0.4.0 sent every keyword to the provider.
- Pydantic models used as tactic configs keep their `global` section: the
  field aliased `global` (by `alias`, `serialization_alias`, or
  `validation_alias`, including `AliasChoices` and a one-key `AliasPath`)
  is read as `global`, where 0.4.0 read it under its field name, such as
  `global_`, and ignored it, so agents lost the shared settings. A field
  left unset at `None`, at any depth, no longer overrides `global` or the
  defaults (a `None` you set still does), and fields are read by name even
  when the model serializes by alias.
- `tactic_as_tool(..., asynchronous=True)` and
  `Tactic.as_tool(asynchronous=True)` make an `async def` tool that awaits
  `arun()`, so a Pydantic AI agent can call a tactic that runs another
  Pydantic AI agent; a sync tool raised `UserError` there.
- `ContextManager`, `DefaultContextManager`, `BaseStreamHandler`, and
  `BaseInvoker` are exported from `lllm.runtimes.native`.
- `copy_boundary_value()` (in `lllm.protocol.validation`, which sssn
  uses), and LLLM's copies of request envelopes and of `CallContext`
  metadata, copy a value nested deeper than Python's recursion limit,
  where 0.4.0 raised `RecursionError`.

**Dialogs, sessions, and cost**

- Dialogs no longer keep the agent loop's working copies. `respond()`
  answered each model call on `dialog.fork()`, so every working copy,
  repairs included, stayed a child of your dialog: `children` and
  `tree_overview()` listed them, and `to_dict()` saved them all, growing
  quadratically with the number of turns. A dialog's messages are, as
  before, the accepted exchange, and the working copies, repair prompts
  included, are not kept. The rejected replies and the parse errors that
  sent them back stay in the call's `AgentCallSession` (`invoke_traces`,
  `exception_retries`). Forks you make with `fork()` are listed as before.
- Images go to the model with their media type (PNG, JPEG, WebP, or GIF,
  read from the first bytes; `image/jpeg` otherwise), not always
  `image/jpeg`. `put_image()` uses the same check, so base64 RIFF data
  that is not WebP (a WAV file, say) now raises `ValueError`.
  `Dialog.overview()` shows an image as `<image/png, 2.0 KB>` and its
  caption instead of its base64 data.
- A `respond()` that raises (repairs or provider retries used up, the
  tool budget spent) is recorded: the exception's `agent_session` is the
  call's `AgentCallSession`, with `state="failure"` and no `delivery`,
  holding every reply, repair, and provider-error retry (rate-limit waits
  are not recorded), and a tactic's session records it too. So
  `agent_sessions`, `agent_call_count`, and `total_cost` now include
  failed calls, which 0.4.0 left out.
- `AgentCallSession.tool_calls` lists every tool call of a `respond()`,
  with its arguments and result; a repeated identical call, which is not
  run again, has no result.
- Agents on OpenAI's Responses API (`api_type="response"`) report their
  prompt, completion, cached, and reasoning tokens, and prompt and
  completion costs; 0.4.0 left these at 0.
- `str(InvokeCost)` shows rates per million tokens (`In: $0.15/1M tok`)
  instead of per token with seven decimals, and a tactic's completion log
  line shows `cost=$0.000040` instead of a raw float.
- A deeply nested input no longer crashes the process that runs a native
  tactic, or fails the call. Naming the session called `repr()` on the
  task, which raised `RecursionError` for a task nested deeply enough
  (from about 1,000 levels on Python 3.10 and 3.11), and on CPython 3.12
  and 3.13 segfaulted on a small thread stack (1 MiB on python:alpine)
  for a dict a few thousand levels deep, as one 18 KB request to a served
  tactic could send. The name now hashes a bounded preview of the task
  (20 levels, 10,000 values). Native tactics also run on FIPS-mode Python
  and with a string task that holds a lone surrogate, which both failed
  in 0.4.0. Session names keep their `{tactic}_{hash}_{timestamp}` form.
  Strings, `None`, booleans, and most numbers hash as before; containers,
  Pydantic models, dataclasses, bytes, and other objects hash differently
  from 0.4.0.

**Context manager**

- `DefaultContextManager` trims to the model's context window:
  `max_tokens` if given, else LiteLLM's `max_input_tokens` (128,000 for
  gpt-4o-mini), remembered per model once found. 0.4.0 used the output
  limit from `litellm.get_max_tokens()` (16,384 for gpt-4o-mini), or 4,096
  for a model LiteLLM does not know, which is below the 5,000-token
  margin, so such agents sent only the system prompt. Now an unknown
  model's calls are not trimmed, with one warning, and `max_tokens` reads
  `None`: set `max_tokens`, or register the model with `max_input_tokens`,
  to trim them. An unknown model is looked up again on each call, so
  registering it later takes effect. For a Hugging Face model, LiteLLM's
  `max_tokens` (the context length in the model's config on the hub) is
  still the window, as in 0.4.0, and registering the model does not
  change it: set `max_tokens` when the hub does not give it. For other
  models, LiteLLM's `max_tokens` can be the output limit, so it is not
  used.
- The default margin kept free for the reply, 5,000 tokens, now shrinks to
  half of a budget under 10,000: `min(5000, max_tokens // 2)`. 0.4.0 always
  kept 5,000, so a budget of 5,000 or less sent only the system prompt. A
  `SAFETY_BUFFER` you set, on an instance, a subclass, or
  `DefaultContextManager` itself (which config-built agents use), is used
  as set.
- Messages are counted as they are sent, tool calls included, and whether
  to trim is decided by one count of the whole dialog, as in 0.4.0. LLLM
  sizes images itself: an image in a data URL is read with Pillow from its
  header and costed with OpenAI's high-detail formula (765 tokens for a
  1024x768 photo), and any other content part (an image whose size cannot
  be read, an image URL, audio) counts as `IMAGE_TOKENS` (1,500), with
  nothing fetched. 0.4.0 counted an image as its base64 text, about
  330,000 tokens for a 356 KB photo. A message LiteLLM cannot count gets
  an estimate, so counting never fails a call.
- The newest message is always sent whole, with a warning when it alone
  is over the budget, and a tool call and its results are kept or dropped
  together, since OpenAI-compatible APIs reject a result without its
  call. An older message that only partly fits keeps its end, after the
  `[...earlier content truncated...]` marker, where 0.4.0 kept its start.
  Images are never cut, which corrupted their data in 0.4.0: one that does
  not fit is dropped whole, with a warning.

**Testing**

- `lllm.testing` answers native agents' model calls without a provider:
  `ScriptedModel` replies in order, `FunctionModel` answers from each
  `ModelRequest`, and `ToolCall` replies run your real tools, also for
  agents that stream and agents on OpenAI's Responses API. The rest of the
  agent loop runs for real. It needs the `native` extra.

**Serving**

- Served tactics refuse, with a 403 `CrossSiteRequestError` in the
  services error envelope, a cross-site `POST` whose `Content-Type` is not
  JSON (`application/json` or a `+json` type) to `/run`, `/stream`, their
  `/tactics/{name}/` forms, or a custom endpoint, and a cross-site `GET` to
  a custom endpoint. Among them are the forms and text any web page can
  make a visitor's browser send without a CORS preflight, which 0.4.0 ran,
  so any page a visitor opened could run a tactic served on their
  `127.0.0.1`. Types a browser preflights, such as
  `application/octet-stream`, are refused too, whatever the app's CORS
  settings. Send JSON, or call from the app's own origin.
- `Sec-Fetch-Site` decides whether a request is cross-site when present
  (`cross-site` and `same-site` are); otherwise an `Origin` that is
  `null`, is not http(s), or names another host and port than `Host`.
  Clients that are not browsers send neither header and are not affected.
  `PUT`, `PATCH`, and `DELETE` are never refused, and the read routes
  (`/info`, `/health`, `/tactics`, the docs) answer anyone. A cross-site
  `POST` with a JSON `Content-Type` needs only the app's CORS settings;
  an app whose pages on other origins must send it other `POST`s, or
  `GET` a custom endpoint, passes `allow_cross_site_requests=True` (a
  `bool`) to `create_tactic_app` or `create_service_app`. `lllm serve`
  keeps the guard on a tactic, but serves such an app as it is
  (`lllm serve module:app`). The guard does not stop DNS rebinding: a
  local service also needs a host allowlist or a token.
- Request bodies are decoded strictly as UTF-8. A body that is not UTF-8
  is a 400 `SchemaError` unless its `Content-Type` is
  `application/octet-stream`, which the tactic still gets as bytes; 0.4.0
  passed such a body on as bytes unless it parsed as JSON, so JSON
  holding UTF-8-encoded surrogates reached the tactic as lone surrogates.
  JSON in UTF-16 or UTF-32, which 0.4.0 parsed, is no longer read as
  JSON: send UTF-8 (a UTF-8 byte order mark is fine). An escaped lone
  surrogate (`"\ud800"`) is also a 400, and so is a body whose first value
  nests arrays and objects more than 500 levels deep, since code that
  walks such a value recursively can crash the server.
- A large request body holds up other requests less: it is decoded,
  depth-checked, and parsed, and an envelope in it validated and its call
  context built, in a worker thread instead of on the event loop. The
  tactic's input-schema validation and the trace's copy of the context
  metadata still run on the event loop, in time proportional to their
  size, so a public service should cap request size at its proxy.
- A result that fails the tactic's output schema is a 500
  `OutputSchemaError`, whose message starts
  `The output of tactic '<name>' failed validation:`; 0.4.0 answered it
  with a 400, as if the input were bad. Calling the tactic in process
  raises it too. It is a `SchemaError` subclass, exported from `lllm` and
  `lllm.protocol`, so `except SchemaError` still catches it, and
  `status_code_for_error()` gives 500 for it and 403 for
  `CrossSiteRequestError` (exported from `lllm.services`).
- Failures that were bare 500s get the error envelope: an unknown tactic
  name with `;`, `:`, `%`, or `\`, a name that is `.` or `..`, or one of
  only whitespace is a 404 `TacticNotFound`, whose `tactic` field is null
  since the envelope cannot hold such a name; a result that cannot be
  encoded (bytes that are not UTF-8, a lone surrogate) is a 500; and a
  lone surrogate in an error message is written as an escape.
- `RemoteTactic` takes `headers=` and `auth=` (any httpx auth), sent with
  every run, stream, and `/info` request; headers passed to one call are
  added to them. `RemoteTactic.from_service(url)` (or `afrom_service`)
  fetches `/info` once and passes it as the new `contract=`: the tactic
  takes the service's name (unless you pass `name=`), and `info()` its
  description, schemas, and examples wherever it has none of its own, so
  a native tool made with `tactic_as_function()` gets the service's
  parameters. A Pydantic AI tool from `tactic_as_tool()` still takes its
  signature from `input_type`, so it gets only the description unless you
  pass `input_type=`. The tactic's capabilities follow the service's, with
  `stream` and `events` counted as one. `RemoteTacticError.message` is
  also filled from a plain FastAPI `{"detail": "..."}` body.

**CLI and API keys**

- `lllm serve module:attr` serves a FastAPI or Starlette app, or a factory
  that returns one, as it is, and `lllm inspect` reports it as an app with
  its routes; 0.4.0 wrapped a FastAPI app in a tactic that failed on
  every request, and stopped with a `SchemaError` traceback for a plain
  Starlette app. A factory that returns nothing usable now stops the
  command with a message that says what it returned and what it must
  return. `load_tactic_entrypoint()` raises `TypeError` for an app.
- A tactic or a class is never taken for an ASGI app, even one with
  `routes` and `router` attributes, such as a routing tactic; with
  Starlette installed, an app is a Starlette instance (FastAPI apps are).
  0.4.0 handed such a tactic, built by a package's service entry, to
  uvicorn as the app.
- `lllm serve PKG --resource channels.NAME` with a channel the package
  does not declare is an error that lists the declared ones; 0.4.0 served
  the package's whole store whatever the name. Missing services, tactics,
  and runs list the declared ones too. `lllm inspect` builds a channel
  package's app over a throwaway store, so it no longer creates
  `<pkg>/.lllm/network`, and its help no longer lists `--store`, which it
  ignores.
- `--credentials auto`, the default for `lllm serve` and `lllm setup`,
  stores the keys you type in the OS keyring only when a usable one
  exists, and in the env file otherwise. 0.4.0 chose the keyring whenever
  `keyring` imported, so on headless hosts, in CI, and in containers,
  where its backend cannot store anything, the key you typed was lost and
  `lllm serve` exited 2. `--credentials keyring` without a usable keyring
  stops with an error before any key is asked for.
- `lllm serve --no-keyring` stores the keys you type in the env file,
  which its key check reads; 0.4.0 stored them in the keyring, then missed
  them and exited 2. `--no-keyring --credentials keyring` is refused when
  serve checks a package's keys.
- Keys you type are no longer lost to an env file that cannot be written:
  the file, or the folder it goes in, is checked before any key is asked
  for, and it is opened without following links and emptied only after
  its checks pass. A new env file is readable only by you, and an
  existing one is made so only if you own it, so a shared file you can
  write keeps its owner and mode. Every failure, a file that is not UTF-8
  and a `~` that cannot be expanded included, is a `CredentialError`,
  which the commands report with exit status 2.
- An env file that is a symlink is refused even when the link is dangling
  or loops: 0.4.0 read such a link as no file, then wrote the keys you
  typed through it, or lost them. Since `lllm serve`, `inspect`, and
  `setup` read the env file (`.env.local` by default) for a package, a
  broken `.env.local` link now stops them with exit status 2; remove it,
  or pass another `--env-file`.
- `lllm serve`, `inspect`, `setup`, and `init` exit 2 with a message
  instead of a traceback for a `~` that cannot be expanded (`~typo/pkg`,
  or `./~pkg`, which pathlib reads as `~pkg`) in their target or
  `--env-file`, and so does `lllm hub` for one in a `--hub` or `--dest`
  path or the package path of `hub init` or `hub publish`; `serve` and
  `inspect` do the same for a target that cannot be read
  (`cannot read <target>: <reason>`). Called from Python,
  `load_manifest()`, `init_package()`, `LocalHub()`, and
  `LocalHub.download()` raise `ValueError` for such a `~`, and
  `load_launch_app()` and `load_launch_api_key_requirements()` raise
  `LaunchError` for one in the target, instead of pathlib's
  `RuntimeError`. A `--store` path still goes to sssn as it is, so such a
  `~` there still ends in a traceback. A `module:attr`
  target now runs from a working folder that cannot be searched (0.4.0
  failed there with a `PermissionError` traceback before Python 3.14), and
  one with too many dotted parts to import exits 2.

**Packages and hub**

- `lllm serve` and the hub's config templates pick a resource's port by
  one rule: its own `port`, in its `metadata` table or written directly
  in its `[services.NAME]` or `[tactics.NAME]` table; else, for a tactic,
  the first port declared by a service that serves it (by `tactic` or
  `tactics`), in manifest order; else 8000. `lllm serve` used to read only
  `metadata.port`, and for a tactic only services naming it in `tactic`;
  the template ignored a tactic's own port and numbered services without
  one 8000, 8001, ..., so its URLs could point elsewhere. A `port`
  written directly in a `[channels.NAME]` table now counts too:
  `lllm serve` and `lllm inspect` use it where 0.4.0 used 8000.
- A `[tactics.NAME]` entry that is a FastAPI or Starlette app, or a
  factory of one, is refused: `lllm serve` and `lllm inspect` stop with a
  `LaunchError`, and `lllm hub validate` reports `tactic_entry_is_app`
  for an app (it does not call factories), so `lllm hub publish` refuses
  the package. 0.4.0 served a FastAPI app as a broken tactic, and stopped
  on a Starlette one with a `SchemaError` traceback. Serve apps from a
  `[services.NAME]` entry. A factory that returns nothing usable, such as
  `None`, also stops both commands with a `LaunchError` that says what it
  returned and what it must return; 0.4.0 served the factory itself as a
  tactic that failed on every request.
- `install_package(..., alias=...)` accepts underscores, as in the names
  `lllm create` writes (`my_glossary`); an alias still starts and ends
  with a lowercase letter or a digit.
- Agent cards put a blank line before `## Resources` after tags, safety,
  or latency lines.

**Proxies and `run_python`**

- `run_python` captures each call's output on its own: calls that run at
  once, as under `bcall`, no longer mix their output, your program's
  prints no longer end up in a call, and when the last call's code ends,
  `sys.stdout` is your program's stream again (after overlapping calls,
  0.4.0 could leave it at a finished call's buffer, losing later prints).
  A call captures what its code prints, including endpoint methods
  `CALL_API` runs and threads the code starts; your program's other
  threads print to your program. The code's own redirects
  (`redirect_stdout`, doctest) work, but are process-wide while they last.
  The first call installs a `threading.Thread.start` hook that stays for
  the life of the process, and a timed-out call's output is dropped until
  its code ends. The Proxies page lists the rules and their limits.
- Long `run_python` output keeps its start and its end, half of
  `max_output_chars` each, around the truncation indicator, so the end of
  a traceback survives; 0.4.0 kept only the start. The interpreter-mode
  proxy prompt and the `run_python` tool description now tell the model
  so, where they said to print the most important data first.
- `use_cache=False` skips writing the HTTP cache too, not just reading it,
  in `call_api` and in the Exa proxy's own cache. POST responses are never
  cached: 0.4.0 cached them and answered repeats from the cache, so
  repeated POSTs now reach the API. Plain (non-JSON) responses are not
  cached; writing one failed, which broke the Wolfram Alpha proxy on every
  call.
- A POST endpoint whose API key is a query parameter sends the key in the
  URL; 0.4.0 put it in the JSON body.
- A params value that `dict()` cannot take stands for the endpoint's only
  parameter, or its only required one, so
  `CALL_API("tools/summarize", text)` works; 0.4.0 failed in `dict()`.
  Params are anything `dict()` takes: a mapping, a Pydantic model, an
  object with `keys()`, or `(name, value)` pairs; 0.4.0 also needed a
  truth value, so it failed on a `pandas.Series`. Where the endpoint takes
  no lone value, a falsy one still means no params, and any other raises
  a `TypeError` that lists the endpoint's parameters.
- `CALL_API` merges keyword params into the positional params, a lone
  value included: 0.4.0 dropped them, so
  `CALL_API("tools/summarize", {"text": text}, style="short")` lost
  `style`. More than one positional params value raises `TypeError`
  instead of being ignored.
- A name in `activate_proxies` that matches no registered proxy logs a
  warning that names it, lists the registered proxies, and points to
  `load_builtin_proxies()`; 0.4.0 skipped it silently.
- Proxies work under test doubles: an endpoint method wrapped by a spy
  (`mock.patch.object(proxy, name, wraps=...)`) keeps its endpoint
  metadata, and a proxy whose `call` is replaced by a mock still lists
  its endpoints. 0.4.0 listed the spy's made-up metadata, or raised
  `TypeError` when the proxy had other endpoints, and failed to list
  endpoints next to the mock. A `mocker.spy` of an endpoint method still
  hides the endpoint, as in 0.4.0: its autospec stand-in carries no
  endpoint metadata, so spy with `wraps=` instead.

**Sandbox**

- `JupyterSession.launch_server()` finds Notebook 7's server URL
  (`/tree?token=...`, or one without a token) and returns the notebook's
  URL; 0.4.0 could hang there. The wait has a real deadline,
  `launch_server(timeout=20)`, and the server's log is drained after
  startup, so a chatty server cannot block on a full pipe.

**Extras**

- The `sandbox` extra installs `ipykernel`; without a kernel, Jupyter
  sessions could not run cells.
- `lllm-core[all]` lists PyYAML, which the `native` extra needs. It was
  only installed as a dependency of other packages.

**Docs**

- New docs at lllm.one, organized by task: Get started, a guide (agents,
  prompts, tools, dialogs, tactics, typed I/O, configuration, streaming,
  models, inference hints, proxies, sandbox), production (serving,
  testing, observability, packages, the hub), integrations (Pydantic AI,
  other runtimes), four complete examples, and an API reference whose
  Python pages are generated from the code (the `docs` extra adds
  `mkdocstrings[python]`). Nearly every Python example runs in the test
  suite, which checks the output each page shows. The old tutorial, core,
  protocol, packages, and reference pages redirect to their new homes.
- The pages state this release's rules: Serving covers requests from
  browsers and DNS rebinding (with a `TrustedHostMiddleware` example),
  the body rules, and the status codes; Dialogs covers what the context
  manager counts and sends, and when LiteLLM downloads a tokenizer for
  Llama and Replicate models and how to stop it; Proxies covers what
  `run_python` captures; Observability covers what sessions and dialogs
  hold.

## 0.4.0

SSSN and AAAX are separate packages again, built on LLLM: `sssn` (channels)
and `aaax` (strategies) depend on lllm-core and plug into the `lllm` command.
The package hub stays in LLLM.

- `lllm.network` and `lllm.shell` moved to the `sssn` and `aaax` packages:
  `pip install sssn aaax`, or the new `lllm-core[network]` and
  `lllm-core[shell]` extras (`all` includes both). Their tests and
  `examples/network` moved with them.
- The old import paths are deprecated shims. `import lllm.network` imports
  `sssn`, warns once with a `DeprecationWarning` that points to
  `import sssn`, re-exports its public names, and aliases the old submodules
  (`lllm.network.server.fastapi` is `sssn.server.fastapi`); `lllm.shell` does
  the same for `aaax`. Without the package they raise
  `ImportError: lllm.network moved to the sssn package: pip install sssn`.
- CLI plugins: `lllm` discovers subcommands from the `lllm.commands`
  entry-point group, which maps a command name to a module with
  `main(argv)`. Installing sssn adds `lllm network`, installing aaax adds
  `lllm shell`, and `lllm --help` lists the installed plugins. Without the
  package, `lllm network` prints
  `lllm network needs the sssn package: pip install sssn` and exits 2. The
  `sssn` and `aaax` commands stay. Plugins load from their installed
  package, so a local `sssn.py` or `aaax.py` cannot shadow them; only
  `lllm inspect`, `serve`, and `hub` put the working directory on
  `sys.path`.
- `lllm.protocol.validation` makes `copy_boundary_value` public for packages
  built on LLLM; `lllm.protocol._validation` is unchanged. `lllm.services`
  exports `call_context_from_data` (the `CallContext` a request's `context`
  gives a served tactic) and `status_code_for_error` (the status a tactic
  error is served with); the private `_context_from_data` and
  `_status_code_for_error` in `lllm.services.fastapi` stay as aliases.
- `lllm.Resolver` and `lllm serve` import channel support from sssn when they
  need it. Tactic refs and settings work without sssn; channel and snapshot
  refs, and channel resources, say to install it. `Resolver.channels` is made
  on first use and can still be assigned, as in 0.3. `lllm.hub` keeps its own
  list of channel forms, which sssn's tests keep in step with
  `sssn.ChannelForm`.
- Docs: lllm.one documents LLLM and the hub, and SSSN and AAAX have their own
  sites again, sssn.one and aaax.one; lllm.one/sssn and lllm.one/aaax
  redirect there. The header switches between the three sites, and the nav
  has no Home section (the logo leads home). Lesson 11 shows SSSN and AAAX
  with LLLM.

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
