# Refs and Composition

A ref is a stable name for a package resource:

```text
lllm://demo/shout/tactics/shout
```

A ref says what a resource is, not where it runs. Local config decides whether
it resolves to an object in your process, a service on another host, or a
channel store. Code that calls tactics by ref keeps working when one of them
moves from a notebook into a service.

## Ref shape

Refs have the form `lllm://ORG/PACKAGE/SECTION/NAME`, where `SECTION` is one of
the package sections:

| Section | Names |
| --- | --- |
| `tactics` | A tactic. |
| `services` | A service. |
| `channels`, `snapshots` | An SSSN channel or snapshot. |
| `schemas`, `runs`, `configs`, `docs`, `examples`, `assets` | Other package resources. |

A package gives each resource a ref built from its `[package]` `org` and
`name`: `[tactics.shout]` in package `demo/shout` is
`lllm://demo/shout/tactics/shout` (see [The lllm.toml manifest](../packages/manifest.md)).

```python
from lllm import TacticRef, TacticRefError
from lllm.protocol.refs import parse_resource_ref

ref = TacticRef("lllm://demo/shout/tactics/shout")
print(ref.org, ref.package, ref.resource_kind, ref.name)

print(parse_resource_ref("lllm://demo/shout/channels/events").resource_kind)

try:
    TacticRef("lllm://demo/shout/channels/events")
except TacticRefError as exc:
    print(exc)
```

```text
demo shout tactics shout
channels
Tactic ref must point at /tactics/, got /channels/: lllm://demo/shout/channels/events
```

Each segment must be a plain path segment: no whitespace, percent escapes, `.`
or `..`, and none of `/`, `\`, `:`, `;`. Unknown sections are rejected. A
tactic's `package_ref` is a tactic ref; its `service_ref` is a `/services/` ref
or an `http(s)` URL.

## lllm.Resolver

`lllm.Resolver` turns refs into live objects: tactics (in-process or remote),
channel stores, and project settings.

### Register in-process tactics

```python
from lllm import Resolver, Tactic

LOWER = "lllm://demo/text/tactics/lower"


class Lower(Tactic):
    """Lowercase the text."""

    name = "lower"

    def call(self, task: str) -> str:
        return task.lower()


resolver = Resolver()
resolver.register(LOWER, Lower())

print(resolver.run(LOWER, "QUIET PLEASE"))
print(resolver.tactic(LOWER).package_ref)
```

```text
quiet please
lllm://demo/text/tactics/lower
```

Registering a tactic also sets its `package_ref`, so the ref appears in its
`info()` and, once served, in the `tactic_ref` of every request context.

### Bind refs in .lllm/config.toml

A project keeps its bindings in `.lllm/config.toml`. This one points the tactic
ref at the service from [Services](services.md#serve-a-tactic) and gives a
channel ref a local store:

```toml
[refs."lllm://demo/shout/tactics/shout"]
url = "http://127.0.0.1:8000"

[refs."lllm://demo/shout/tactics/shout".metadata]
owner = "docs"
api_key_ref = "openai.default"

[refs."lllm://demo/shout/channels/events"]
store = ".lllm/network"

[settings]
region = "us"
```

`Resolver.from_config()` reads the file once:

```python
from lllm import Resolver

resolver = Resolver.from_config(".")

shout = resolver.tactic("lllm://demo/shout/tactics/shout")
print(type(shout).__name__, shout.tactic_name, shout.url)
print(shout.run({"text": "via config"}))
print(shout.info().metadata)

store = resolver.store("lllm://demo/shout/channels/events")
print(type(store).__name__)
print(resolver.setting("region"), resolver.setting("timeout", 30))
print(resolver.refs())
```

```text
RemoteTactic shout http://127.0.0.1:8000/run
{'text': 'VIA CONFIG', 'trace_id': None}
{'owner': 'docs', 'api_key_ref': 'openai.default', 'ref': 'lllm://demo/shout/tactics/shout', 'url': 'http://127.0.0.1:8000/run'}
LocalStore
us 30
('lllm://demo/shout/channels/events', 'lllm://demo/shout/tactics/shout')
```

The rules:

- `from_config(path)` takes a project folder (and reads
  `PATH/.lllm/config.toml`) or the path of the file itself.
- A tactic ref binds to `url`: an absolute `http(s)` service URL, either the
  service base or `.../tactics/NAME` inside a multi-tactic app. It resolves to a
  `RemoteTactic` named after the ref, with the ref as its `package_ref`. It has
  no `input_type` or `output_type`, so it returns JSON as-is.
- A channel or snapshot ref binds to `store`, a relative path resolved against
  the project folder (a `LocalStore`; opening it creates the folder), or to
  `url`, an SSSN store service (a `RemoteStore`).
- Each binding declares exactly one target. A tactic ref bound to anything but
  a `url` is rejected:

    ```text
    TacticRefError: Tactic URL binding must declare a url target, got store: lllm://demo/shout/tactics/shout
    ```

- The `metadata` table (and any extra keys) becomes binding metadata, which a
  remote tactic reports in `info().metadata` next to `ref` and `url`. Raw
  secrets are refused; point at credentials with a reference such as
  `api_key_ref` instead:

    ```text
    TacticRefError: Tactic URL binding metadata must not include raw secret key 'api_key': lllm://demo/shout/tactics/shout
    ```

- Refs to other sections, such as `/services/`, may sit in the same file for
  other tools. The resolver validates their shape and otherwise leaves them
  alone.
- `[settings]` is free-form; `setting(key, default=None)` reads it.
- `register()` replaces a config binding for the current process, which is
  handy in tests.

`lllm hub config-template` renders a starting config for a published package
([The Hub](../packages/hub.md)).

### Resolver reference

| Method | Does |
| --- | --- |
| `Resolver()` | An empty resolver; register tactics yourself. |
| `Resolver.from_config(path=".")` | Load bindings and settings from `.lllm/config.toml`. |
| `register(ref, tactic)` | Bind a tactic ref to an in-process tactic. |
| `tactic(ref)` | The tactic bound to a `/tactics/` ref. Raises `TacticRefError` when unbound. |
| `store(ref)` | The `LocalStore` or `RemoteStore` behind a `/channels/` or `/snapshots/` ref. |
| `resolve(ref)` | `tactic()` or `store()`, chosen by the ref's section. |
| `run(ref, value, *, context=None, **kwargs)`, `await arun(...)` | Resolve a tactic ref and call it. |
| `setting(key, default=None)` | A value from `[settings]`. |
| `refs()` | Every bound tactic and channel ref. |

Its parts are attributes: `resolver.tactics` is a `TacticResolver` (also
importable from `lllm`) whose `bind_url(ref, url, **kwargs)` binds a ref to a
service in code, passing extra keywords such as `input_type` or `timeout` to
`RemoteTactic`. `resolver.channels` is the `ChannelResolver` from
`lllm.network`, and `resolver.config` holds the parsed file.

## Composition patterns

### Call tactics by ref

A tactic that holds a resolver can call other tactics without knowing where
they run:

```python
from lllm import CallContext, Resolver, Tactic

SHOUT = "lllm://demo/shout/tactics/shout"
LOWER = "lllm://demo/text/tactics/lower"


class Contrast(Tactic):
    """Say it loud, then say it quietly."""

    name = "contrast"
    output_type = dict

    def __init__(self, resolver: Resolver, **kwargs):
        super().__init__(**kwargs)
        self.resolver = resolver

    def call(self, task: str, *, context=None) -> dict:
        loud = self.resolver.run(SHOUT, {"text": task}, context=context)
        quiet = self.resolver.run(LOWER, task, context=context)
        return {"loud": loud["text"], "quiet": quiet, "trace_id": loud["trace_id"]}


resolver = Resolver.from_config(".")  # SHOUT is bound to the service
resolver.register(LOWER, Lower())     # LOWER runs in this process

contrast = Contrast(resolver)
print(contrast.run("Hello There", context=CallContext(trace_id="trace-7")))

session = contrast("Hello There", return_session=True)
print(list(session.sub_tactic_sessions))
```

```text
{'loud': 'HELLO THERE', 'quiet': 'hello there', 'trace_id': 'trace-7'}
['lower']
```

- `Contrast` does not know that `SHOUT` is remote. Moving `LOWER` behind a
  service means binding its ref to a `url` in `.lllm/config.toml` instead of
  registering it; `Contrast` stays the same.
- Passing `context` along carries the `trace_id` across the HTTP hop: the
  remote tactic received `trace-7` and echoed it back.
- In-process tactics that run in a session, such as the `call()` program
  `Lower`, are recorded in the caller's session (`lower` above). The remote
  call is an HTTP request, so the service keeps its own record of it.
- To serve a tactic that takes constructor arguments, point `lllm serve` at a
  factory, such as `def build(): return Contrast(Resolver.from_config("."))`,
  which it calls once at load
  ([Loading entrypoints](runtimes.md#loading-entrypoints)).

### Sub-tactics

Inside one program, assigning a tactic to an attribute in `__init__` makes it a
sub-tactic: it is listed in `self.sub_tactics`, and its calls roll up into the
parent's session and cost. See
[Sessions for call() programs](tactic.md#sessions-for-call-programs).

### Test against an in-process service

`bind_url` accepts httpx transports, so a test can exercise the full HTTP path
(envelope, context, errors) without opening a port:

```python
import asyncio

import httpx
from lllm import Resolver, create_tactic_app

resolver = Resolver()
app = create_tactic_app(Lower())
resolver.tactics.bind_url(LOWER, "http://testserver", async_transport=httpx.ASGITransport(app=app))

print(type(resolver.tactic(LOWER)).__name__)
print(asyncio.run(resolver.arun(LOWER, "OVER HTTP")))
```

```text
RemoteTactic
over http
```

`httpx.ASGITransport` is async-only, so call `arun()`; for synchronous `run()`,
pass a synchronous transport as `transport=`.

### Tactics as tools

Any tactic, local or bound by ref, can be handed to an agent as a tool. See
[Tactics as Pydantic AI tools](runtimes.md#tactics-as-pydantic-ai-tools).

### Tactics over channels

A channel ref resolves to a store, and a `TacticProcessor` runs a tactic over a
channel's events, appending each output to another channel:

```python
from lllm import Resolver
from lllm.network import TacticProcessor

resolver = Resolver.from_config(".")
store = resolver.store("lllm://demo/shout/channels/events")

store.create_channel({"name": "inbox"})
store.create_channel({"name": "outbox"})
store.append_event({"channel": "inbox", "payload": "HELLO THERE"})

processor = TacticProcessor(Lower(), store, source="inbox", sink="outbox")
for event in processor.process():
    print(event.kind, event.payload, event.parent_ids != ())
print(processor.process())
```

```text
result hello there True
[]
```

The processor reads through a named subscription, so a second `process()` picks
up only new events. Each output event links back to its source through
`parent_ids`, and the tactic receives a `CallContext` naming the channel and
event. Channels, stores, and processors are covered in [SSSN](../sssn/index.md);
mounting several packages and channels behind one API is [AAAX](../aaax/index.md).

## Next

- [The Tactic](tactic.md): the boundary every ref resolves to.
- [Services](services.md): the HTTP side of a `url` binding.
- [Runtimes](runtimes.md): what can sit behind a tactic.
