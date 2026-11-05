# AAAX: the shell module

AAAX, the **Advanced Autonomous Agent Executor**, is the shell around an
agentic system. It mounts tactics, packages, and channels into one `Strategy`
and serves them as a single HTTP API: every tactic gets its full service
surface, every channel store gets the network API, and the strategy adds a
resource table and a top-level `/run`. In code, AAAX is `lllm.shell`.

AAAX is an experimental module of LLLM.

```bash
pip install "lllm-core[server]"
```

## A first strategy

```python
from pydantic import BaseModel

from lllm import Tactic
from lllm.shell import Strategy


class Text(BaseModel):
    text: str


class Shout(Tactic[Text, Text]):
    name = "shout"
    description = "Upper-case a message."
    input_type = Text
    output_type = Text

    def _run(self, input_value, *, context=None):
        return Text(text=input_value.text.upper())


strategy = Strategy("loud-desk", description="Shout at things.")
strategy.add_tactic(Shout(), ref="lllm://demo/loud-desk/tactics/shout")


@strategy.runner
async def run(input_value, *, context=None):
    first = await strategy.tactics()["shout"].arun(input_value)
    return {"once": first.text, "twice": f"{first.text}!!"}
```

`create_strategy_app` turns it into a FastAPI app:

```python
from fastapi.testclient import TestClient

from lllm.shell import create_strategy_app

client = TestClient(create_strategy_app(strategy))
print(client.post("/tactics/shout/run", json={"input": {"text": "hello"}}).json()["output"])
print(client.post("/run", json={"input": {"text": "hello"}}).json())
```

```text
{'text': 'HELLO'}
{'output': {'once': 'HELLO', 'twice': 'HELLO!!'}, 'strategy': 'loud-desk', 'metadata': {}}
```

Saved as `desk.py`, the same strategy runs from the shell with
`lllm shell serve desk.py` (see [CLI](#cli)).

## Resources

`Strategy(name, *, description="", resources=(), runner=None, metadata=None)`
is a table of resources plus an optional runner. Each resource is a
`StrategyResource`:

| Field | Meaning |
| --- | --- |
| `name` | Local name, unique across the whole strategy, used in URLs. A path segment. |
| `kind` | `tactic`, `channel`, `service`, `schema`, `package`, `run`, `config`, `doc`, `example`, `snapshot`, `asset`, or `custom`. |
| `ref` | Stable external identity, usually an `lllm://` ref. |
| `entrypoint` | Python import target (`module:attr`) for package resources. |
| `description` | Free text. |
| `metadata` | Any JSON-friendly dict: runtime, schema, package, version, card. |

Names are local and refs are stable: mounting the same package twice under
different prefixes changes the names but keeps the refs. Add resources with
the helpers `strategy.tactic()`, `.channel()`, `.service()`, `.package()`,
`.snapshot()`, the general `strategy.resource(kind, name, ...)`, or
`add_resource(StrategyResource(...))`. Each takes `ref`, `entrypoint`,
`description`, `metadata`, and an optional `handler`:

```python
strategy.channel("log", ref="lllm://demo/loud-desk/channels/log", description="Everything shouted.")
strategy.resource("doc", "readme", ref="lllm://demo/loud-desk/docs/readme", metadata={"path": "README.md"})

for resource in strategy.resources:
    print(resource.kind, resource.name, resource.ref)
```

```text
tactic shout lllm://demo/loud-desk/tactics/shout
channel log lllm://demo/loud-desk/channels/log
doc readme lllm://demo/loud-desk/docs/readme
```

A resource without a handler is metadata: it shows up in `/strategy` and
`/resources`, and invoking it echoes the resource record and the input back. A
handler makes it callable.

## Tactics

`add_tactic(tactic, *, name=None, ref=None, description=None, metadata=None)`
mounts an lllm tactic. The name defaults to `tactic.info().name`, the ref to the
tactic's `package_ref`, and the runtime kind is recorded in metadata.

A plain function also works as a tactic handler. `strategy.tactics()` wraps it
with `lllm.as_tactic`, so it is served like any other tactic:

```python
def count_words(payload: dict) -> dict:
    return {"words": len(payload["text"].split())}


strategy.tactic("count", handler=count_words, description="Count words.")
print(sorted(strategy.tactics()))
```

```text
['count', 'shout']
```

`strategy.tactics()` returns every tactic resource as an lllm `Tactic`. Each one
is mounted with `create_tactic_app` at `/tactics/<name>/`, so it gets the full
[service surface](../protocol/services.md): `run`, `stream`, `info`, and any
custom tactic endpoints, with the services request and error envelopes.

## Channels and stores

`attach_store(store)` backs the strategy with a channel store and mounts the
[store API](../sssn/serving.md#http-api) at `/network/`:

```python
from lllm.network import ChannelExistsError, LocalStore

store = LocalStore(".lllm/network")
try:
    store.create_channel({"name": "log", "description": "Everything shouted."})
except ChannelExistsError:
    pass
strategy.attach_store(store)


@strategy.runner
async def run(input_value, *, context=None):
    output = await strategy.tactics()["shout"].arun(input_value)
    strategy.store.append_event({"channel": "log", "kind": "shout", "payload": output})
    return output


client = TestClient(create_strategy_app(strategy))
client.post("/run", json={"input": {"text": "hello"}})
print([event["payload"] for event in client.get("/network/events", params={"channel": "log"}).json()])
```

```text
[{'text': 'HELLO'}]
```

`attach_store` does not create channels or bind handlers. The channel
shortcuts `/channels/{name}/events` call a channel resource's handler, so they
work for package channels (below), which get one automatically. For channels
you declare with `strategy.channel()`, use `/network/` instead; the shortcuts
only return the resource description. A
[`TacticProcessor`](../sssn/processing.md) over `strategy.store` is the usual
way to connect a mounted tactic to a channel; [Tutorial
11](../tutorials/11-channels.md) does that inside a runner.

## Packages

A package folder with an `lllm.toml` (see [The manifest](../packages/manifest.md))
mounts in one call:

```text
support-desk/
  lllm.toml
  tactics.py
```

`support-desk/lllm.toml`:

```toml
[package]
org = "demo"
name = "support-desk"
version = "0.1.0"
kind = "app"
description = "Triage support messages."

[tactics.triage]
entry = "tactics:triage"
description = "Assign a priority to a support message."

[channels.inbox]
description = "Support messages as they arrive."
```

`support-desk/tactics.py`:

```python
def triage(ticket: dict) -> dict:
    urgent = "down" in ticket["text"].lower().split()
    return {"priority": "urgent" if urgent else "normal"}
```

```python
from pathlib import Path

from lllm.shell import strategy_from_package

desk = strategy_from_package("support-desk")
for resource in desk.resources:
    print(resource.kind, resource.name, resource.ref)
print(desk.store.root.relative_to(Path("support-desk").resolve()))
```

```text
package support-desk lllm://demo/support-desk
tactic triage lllm://demo/support-desk/tactics/triage
channel inbox lllm://demo/support-desk/channels/inbox
.lllm/network
```

The package's channels live in `<package>/.lllm/network`, created with every
declared channel, and their resources get handlers, so the shortcuts work:

```python
client = TestClient(create_strategy_app(desk))
client.post(
    "/channels/inbox/events",
    json={"input": {"kind": "message", "payload": {"text": "Checkout is down"}}},
)
events = client.get("/channels/inbox/events").json()["output"]
print([(event["kind"], event["source"], event["payload"]) for event in events])
print(client.post("/tactics/triage/run", json={"input": events[0]["payload"]}).json()["output"])
```

```text
[('message', 'lllm.shell', {'text': 'Checkout is down'})]
{'priority': 'urgent'}
```

Package tactics load lazily: building or inspecting the strategy never imports
package code; building the app, calling `strategy.tactics()`, or invoking the
resource does.

To combine packages and your own resources, mount packages into a strategy
with `use_package` (or `add_package(strategy, path, ...)`). A `prefix` keeps
names apart:

```python
ops = Strategy("ops", description="Support and loud desks in one API.")
ops.use_package("support-desk", prefix="support")
ops.add_tactic(Shout())
print([resource.name for resource in ops.resources])
```

```text
['support.support-desk', 'support.triage', 'support.inbox', 'shout']
```

The prefixed tactic is served at `/tactics/support.triage/run` and the channel
at `/channels/support.inbox/events`.

| Function | What it does |
| --- | --- |
| `strategy_from_package(path, *, name=None, prefix=None, store=None, bind=True)` | A new strategy named after the package, with the package card and version in its metadata. |
| `strategy.use_package(path, *, prefix=None, store=None, bind=True)` | Mount a package into an existing strategy. Same as `add_package(strategy, path, ...)`. |

`path` is a package folder or an `lllm.toml` file. `store` picks the channel
store folder. `bind=False` mounts metadata only: no handlers, no store, and no
tactic routes.

Every manifest section becomes resources:

| Manifest section | Resource kind | Handler when bound |
| --- | --- | --- |
| `[package]` | `package` (named after the package) | |
| `[config]` | `config` named `default` | |
| `[schemas.<name>]` | `schema` | |
| `[tactics.<name>]` | `tactic` | the imported entrypoint, as a tactic |
| `[channels.<name>]` | `channel` | append, query, info on the package store |
| `[snapshots.<name>]` | `snapshot` | |
| `[services.<name>]` | `service` | its `tactic`'s handler, when it names one |
| `[runs.<name>]`, `[docs.<name>]`, `[examples.<name>]`, `[assets.<name>]` | `run`, `doc`, `example`, `asset` | |

Things to know when mounting several packages:

- Each package with channels opens its own store unless you pass `store=`.
  The `/network/` mount serves the store of the last package mounted; the
  channel shortcuts always reach the right store.
- Prefixes rename resources, not the channels inside a store. Packages that
  share one `store=` folder must not declare the same channel name.

## The runner

`/run` calls the strategy's runner, set with the `@strategy.runner` decorator or
`Strategy(..., runner=fn)`. The runner may be sync or async and is called as
`runner(input_value, context=...)`; it receives `context` only if it declares
that parameter (or `**kwargs`). Without a runner, `/run` returns the strategy
itself, which makes a fresh mount easy to check. `desk` has no runner:

```python
output = client.post("/run", json={"input": {"q": 1}}).json()["output"]
print(output["strategy"], output["input"], [resource["name"] for resource in output["resources"]])
```

```text
support-desk {'q': 1} ['support-desk', 'triage', 'inbox']
```

`await strategy.arun(input_value, context=None)` and
`await strategy.invoke_resource(name, input_value, context=None)` do the same
from Python.

## HTTP surface

| Route | What it does |
| --- | --- |
| `GET /health` | `{"ok": true, "strategy": "<name>"}` |
| `GET /strategy` | The `StrategyInfo`: name, description, resources, metadata. |
| `GET /resources` | Every resource. |
| `GET /packages`, `GET /tactics`, `GET /channels` | Resources of one kind. |
| `POST /run` | Call the runner. |
| `POST /resources/{name}/invoke` | Call any resource's handler. |
| `GET /channels/{name}/events` | Query a channel resource: `?after_cursor=0&limit=100&kind=...`. |
| `POST /channels/{name}/events` | Append to a channel resource. The `input` holds the event fields: `payload`, `kind` (default `event`), `source` (default `lllm.shell`), `schema`, `metadata`, `correlation_id`, `parent_ids`. |
| `/tactics/{name}/...` | The mounted tactic service: `POST run`, `POST stream`, `GET info`, custom endpoints. |
| `/network/...` | The store API, when the strategy has a store. |

The shell routes (`/run`, `/resources/...`, `/channels/...`) take a
`StrategyRunRequest`, `{"input": ..., "context": {...}}`, and return a
`StrategyRunResponse`, `{"output": ..., "strategy": ..., "metadata": {...}}`:

```python
response = client.post(
    "/resources/inbox/invoke",
    json={"input": {"op": "info"}, "context": {"request": "demo"}},
)
print(response.json())
```

```text
{'output': {'name': 'inbox', 'schema': None, 'form': 'log', 'description': 'Support messages as they arrive.', 'metadata': {}}, 'strategy': 'support-desk', 'metadata': {'resource': 'inbox', 'endpoint': 'resource'}}
```

Their `context` is a plain object handed to the runner or handler as a dict;
a mounted `Tactic` invoked this way receives it as `CallContext.metadata`. The
`/tactics/{name}/` routes use the services envelope instead, where `context`
has the shape of a `CallContext` (`{"metadata": {...}, "tags": {...}}`) and
the response is `{"output": ..., "request_id": ..., "tactic": ...}`.
A plain-function handler therefore receives a dict through
`/resources/{name}/invoke` and a `CallContext` through `/tactics/{name}/run`;
write it so it does not depend on which one it gets, or make it a `Tactic`.
`/resources/{name}/invoke` with a channel resource takes an `op`: `append`,
`query` (the default), or `info`.

Unknown resource names return 404 and handler `ValueError`s return 400, both as
`{"detail": "<message>"}`; any other handler exception is a 500. Tactic routes
and the store API use their own error envelope,
`{"detail": {"error": {"type": ..., "message": ...}}}`.

## Loading strategies

`load_strategy(target)` accepts:

| Target | Loads |
| --- | --- |
| a directory | `strategy.py`, else `shell.py`, else its `lllm.toml` as a package |
| a `.py` file | its `build_strategy()` factory, else its `strategy` object |
| an `lllm.toml` file | the package, as with `strategy_from_package` |
| `module:attr` | a `Strategy` or a no-argument factory that returns one, imported from the working directory |

The CLI uses the same loader.

## CLI

```bash
lllm shell inspect TARGET [--json]
lllm shell serve TARGET [--host 127.0.0.1] [--port 8400] [--log-level info]
```

`inspect` prints the resource table without starting a server; `--json` prints
the full `StrategyInfo`. With `desk.py` holding the strategy from the first
section:

```bash
lllm shell inspect desk.py
```

```text
loud-desk: Shout at things.
tactic shout lllm://demo/loud-desk/tactics/shout
```

`serve` builds the app and runs it with uvicorn; `launch` is an alias:

```bash
lllm shell serve desk.py --port 8400
```

From another terminal:

```bash
curl -s -w '\n' -X POST http://127.0.0.1:8400/tactics/shout/run \
  -H 'content-type: application/json' -d '{"input": {"text": "hello"}}'
```

```text
{"output":{"text":"HELLO"},"request_id":"af421ae3-e594-4fc9-90af-2694b1b5d25b","tactic":"shout"}
```

Inspecting a package with channels creates its store folder, since mounting
opens the store.

## Status

Names and routes may change between releases. The strategy app has no
authentication and handlers run in the server process, so serve it on
localhost or behind your own gateway.

## Next

- [Tutorial 11: Channels with SSSN](../tutorials/11-channels.md) ends by
  serving a package and a `TacticProcessor` as one strategy.
- [SSSN](../sssn/index.md): the channels and stores a strategy mounts.
- [Services](../protocol/services.md): the tactic service each tactic gets.
