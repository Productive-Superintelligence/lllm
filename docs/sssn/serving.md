# Serving stores

A store becomes a service with one call: `create_app(store)` returns a FastAPI
app that exposes every store operation over HTTP. `RemoteStore` and
`AsyncRemoteStore` call that API with the same methods as `LocalStore`, so code
written against a local store runs against a remote one by swapping the
constructor (the few differences are listed under [Clients](#clients)). Refs
(`lllm://org/package/channels/name`) let code name a channel and leave the
choice of store to a config file.

```bash
pip install "lllm-core[server,client]"
```

## Serve a store

From the shell:

```bash
lllm network --store .lllm/network serve --host 127.0.0.1 --port 7700
```

`--store` defaults to `.lllm/network` and `--port` to `7700`. In Python, the
app is a regular ASGI app:

```python
import uvicorn

from lllm.network import LocalStore, create_app

app = create_app(LocalStore(".lllm/network"))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7700)
```

`create_app()` without a store opens `LocalStore(".lllm/network")` in the
working directory. The app keeps the store at `app.state.network_store`.

The server has no authentication. Keep it on localhost or put it behind a
gateway that handles access control.

## HTTP API

| Method and path | Body or query | Returns |
| --- | --- | --- |
| `GET /health` | | `{"ok": true}` |
| `POST /channels` | a channel: `name`, `schema`, `form`, `description`, `metadata` | the channel |
| `GET /channels` | | all channels |
| `GET /channels/{name}` | | one channel |
| `POST /events` | an event: `channel`, `payload`, `kind`, `source`, `schema`, `metadata`, `correlation_id`, `parent_ids`, optional `id` and `timestamp` | the stored event, with `cursor` |
| `GET /events` | `?channel=...&after_cursor=0&limit=100&kind=...` | events, oldest first |
| `GET /events/{id}` | | one event |
| `POST /subscriptions` | `channel`, `id`, `consumer`, `batch_size`, `filters`, `metadata` | the subscription (existing one if the id is taken on the same channel) |
| `GET /subscriptions/{id}` | | the subscription and its cursor |
| `POST /subscriptions/{id}/pull` | `?limit=...` | pending events; moves the cursor |
| `POST /artifacts` | `data`, `encoding` (`text` or `base64`), `channel`, `media_type`, `metadata`, `event_ids` | artifact metadata |
| `GET /artifacts/{id}` | | the raw bytes, with the artifact's media type |
| `GET /artifacts/{id}/metadata` | | artifact metadata |
| `PUT /snapshots/{name}` | `value`, `channel`, `timestamp`, `schema`, `source_event_id`, `metadata` | the snapshot (replaces any previous value) |
| `GET /snapshots/{name}` | | the snapshot |

FastAPI also serves the OpenAPI schema at `/openapi.json` and interactive docs
at `/docs`.

```bash
curl -s -w '\n' -X POST http://127.0.0.1:7700/channels \
  -H 'content-type: application/json' -d '{"name": "inbox"}'
curl -s -w '\n' -X POST http://127.0.0.1:7700/events \
  -H 'content-type: application/json' \
  -d '{"channel": "inbox", "kind": "message", "payload": {"text": "Checkout is down"}}'
curl -s -w '\n' "http://127.0.0.1:7700/events?channel=inbox&limit=10"
```

```text
{"name":"inbox","schema":null,"form":"log","description":"","metadata":{}}
{"id":"6857626f-ce8f-4240-adee-0ac098a3622e","cursor":1,"channel":"inbox","timestamp":1793846751.1789103,"source":null,"kind":"message","payload":{"text":"Checkout is down"},"schema":null,"metadata":{},"correlation_id":null,"parent_ids":[]}
[{"id":"6857626f-ce8f-4240-adee-0ac098a3622e","cursor":1,"channel":"inbox","timestamp":1793846751.1789103,"source":null,"kind":"message","payload":{"text":"Checkout is down"},"schema":null,"metadata":{},"correlation_id":null,"parent_ids":[]}]
```

### Errors

Errors come back in one envelope, with the error class name as `type`:

```bash
curl -s -w '\n' http://127.0.0.1:7700/channels/missing
```

```text
{"detail":{"error":{"type":"ChannelNotFoundError","message":"Channel not found: missing"}}}
```

| Status | When |
| --- | --- |
| 400 | `InvalidPayloadError`, other `NetworkError`s, and request bodies that fail validation (reported as `InvalidPayloadError`). |
| 404 | `ChannelNotFoundError`, `EventNotFoundError`, `SubscriptionNotFoundError`, `ArtifactNotFoundError`, `SnapshotNotFoundError`. |
| 409 | `ChannelExistsError`, `SubscriptionExistsError`. |
| 500 | Anything else, with the exception's class name and message. |

## Custom endpoints

Add domain routes next to the built-in ones with the `endpoint` decorators
(`endpoint.get`, `.post`, `.put`, `.patch`, `.delete`) and pass the functions
to `create_app`:

```python
from fastapi.testclient import TestClient

from lllm.network import InvalidPayloadError, LocalStore, create_app, endpoint


@endpoint.get("/channels/{name}/count", scope="channel", tags=("stats",))
def count_events(store, name: str):
    """Count every event in a channel."""
    total, cursor = 0, 0
    while batch := store.query_events(name, after_cursor=cursor, limit=500):
        total += len(batch)
        cursor = batch[-1].cursor
    return {"channel": name, "events": total}


@endpoint.post("/channels/{name}/messages", scope="channel")
def post_message(store, name: str, body=None):
    """Append a message event from a {"text": ...} body."""
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        raise InvalidPayloadError('body must be {"text": "..."}')
    event = store.append_event(
        {"channel": name, "kind": "message", "source": "api", "payload": {"text": body["text"]}}
    )
    return {"id": event.id, "cursor": event.cursor}


store = LocalStore(".lllm/network")
store.create_channel({"name": "inbox"})
app = create_app(store, custom_endpoints=[count_events, post_message])

client = TestClient(app)
client.post("/channels/inbox/messages", json={"text": "Checkout is down"})
print(client.get("/channels/inbox/count").json())
print(client.post("/channels/inbox/messages", json={"txt": "typo"}).json())
```

```text
{'channel': 'inbox', 'events': 1}
{'detail': {'error': {'type': 'InvalidPayloadError', 'message': 'body must be {"text": "..."}'}}}
```

How arguments are bound:

- A parameter named `store` receives the store.
- A parameter named `body` receives the parsed JSON body (`None` for `GET`,
  `DELETE`, or a body that is not JSON).
- Any other parameter is filled from a path parameter of the same name.
  Query strings are not passed; a parameter with a default keeps it.

The function may return any JSON-encodable value, or be `async`. Exceptions map
to statuses like the built-in routes, so raise `NetworkError` subclasses for
4xx responses.

Routes must not collide with the built-in ones (`GET /channels/{x}` is taken,
whatever the parameter is called), and endpoint names and method-plus-path
pairs must be unique; `create_app` raises `ValueError` otherwise. Functions
without a decorator are ignored. The decorator arguments are metadata: `name`
(defaults to the function name), `scope` (`store`, `channel`, `subscription`,
`artifact`, or `snapshot`), `description` (defaults to the docstring), and
`tags` show up in the OpenAPI schema and in
[`channel_resource`](#package-metadata).

## Clients

`RemoteStore` is the synchronous client, `AsyncRemoteStore` the async one. Both
take `base_url`, `timeout=30.0`, and an optional httpx `transport`, and both
have the `LocalStore` methods. With the store from the previous sections served
on port 7700:

```python
from lllm.network import RemoteStore, RemoteStoreError

store = RemoteStore("http://127.0.0.1:7700")
event = store.append_event(
    {"channel": "inbox", "kind": "message", "payload": {"text": "Search is broken"}}
)
print(event.cursor, store.get_event(event.id).payload)

store.put_snapshot("queue-depth", {"open": 2}, channel="inbox", source_event_id=event.id)
print(store.get_snapshot("queue-depth").value)

try:
    store.get_channel("missing")
except RemoteStoreError as exc:
    print(exc.status_code, exc.error_type, exc.message)
```

```text
2 {'text': 'Search is broken'}
{'open': 2}
404 ChannelNotFoundError Channel not found: missing
```

Where the clients differ from `LocalStore`:

| Method | `LocalStore` | `RemoteStore` / `AsyncRemoteStore` |
| --- | --- | --- |
| `put_snapshot` | `put_snapshot(snapshot)` with a `Snapshot` or a dict | `put_snapshot(name, value=None, *, channel=None, timestamp=None, schema=None, source_event_id=None, metadata=None)`; `value` may also be a `Snapshot` |
| `write_artifact` | `bytes`, `bytearray`, or `memoryview` | `bytes` (sent base64-encoded) or `str` (sent as UTF-8 text) |
| errors | `NetworkError` subclasses | `RemoteStoreError` for server errors, with `status_code`, `error_type`, `message`, and the raw `detail`. Names and ids are checked before sending and raise `InvalidPayloadError`. |

`RemoteStoreError` is a `RuntimeError`, not a `NetworkError`; branch on
`error_type` (for example `"SubscriptionNotFoundError"`) or `status_code`.
Each call opens its own short-lived HTTP connection.

One `put_snapshot` caveat: when called without keyword arguments, a dict
`value` that contains any of the keys `value`, `channel`, `timestamp`,
`schema`, `source_event_id`, or `metadata` is sent as the request body itself,
so `put_snapshot("config", {"value": 3, "unit": "ms"})` stores `3`. Wrap such
values, or pass a `Snapshot`, to store them whole.

`AsyncRemoteStore` has the same methods as coroutines. Pass
`httpx.ASGITransport` to call an app in-process, which is the easiest way to
test code that uses a remote store:

```python
import asyncio

import httpx

from lllm.network import AsyncRemoteStore, LocalStore, create_app


async def main():
    app = create_app(LocalStore("test-store"))
    store = AsyncRemoteStore("http://test", transport=httpx.ASGITransport(app=app))
    await store.create_channel({"name": "inbox"})
    await store.append_event({"channel": "inbox", "payload": {"text": "hi"}})
    return [event.payload for event in await store.query_events("inbox")]


print(asyncio.run(main()))
```

```text
[{'text': 'hi'}]
```

## Refs and `.lllm/config.toml`

A channel ref names a channel by package, not by location:

```text
lllm://<org>/<package>/channels/<name>
lllm://<org>/<package>/snapshots/<name>
```

`ChannelRef` parses and validates one:

```python
from lllm.network import ChannelRef

ref = ChannelRef("lllm://demo/support-desk/channels/inbox")
print(ref.org, ref.package, ref.resource_kind, ref.name)
```

```text
demo support-desk channels inbox
```

A project binds refs to stores in `.lllm/config.toml`, the same file that binds
tactic refs (see [Refs](../protocol/refs.md)). Each binding has exactly one
target:

```toml
[refs."lllm://demo/support-desk/channels/inbox"]
store = ".lllm/network"            # a LocalStore folder, relative to the project root

[refs."lllm://demo/support-desk/channels/triaged"]
url = "http://127.0.0.1:7700"      # a served store
owner = "support"                  # extra keys are kept as binding metadata

[refs."lllm://demo/support-desk/snapshots/latest-reply"]
path = "data/support-store"        # same as store
```

`ChannelResolver` reads the `/channels/` and `/snapshots/` bindings and ignores
the others (tactics, services, schemas, ...), which other resolvers read from
the same file:

```python
from lllm.network import ChannelResolver

channels = ChannelResolver.from_config(".")
print(channels.refs())

triaged = channels.resolve("lllm://demo/support-desk/channels/triaged")
print(triaged.name, triaged.url, triaged.metadata)

print(type(channels.local_store("lllm://demo/support-desk/channels/inbox")).__name__)
print(type(channels.client("lllm://demo/support-desk/channels/triaged")).__name__)
```

```text
('lllm://demo/support-desk/channels/inbox', 'lllm://demo/support-desk/channels/triaged', 'lllm://demo/support-desk/snapshots/latest-reply')
triaged http://127.0.0.1:7700 {'owner': 'support'}
LocalStore
RemoteStore
```

| Method | Returns |
| --- | --- |
| `ChannelResolver.from_config(path)` | A resolver for `path/.lllm/config.toml`, or for a config file path. Relative store paths resolve against the folder that contains `.lllm/`. The file must exist. |
| `ChannelResolver(root=".")` and `bind(ref, *, url=None, store=None, path=None, metadata=None)` | Bind refs in code. |
| `resolve(ref)` | A `ResolvedChannelRef`: `ref`, `resource_kind`, `name`, `url`, `store`, `path`, `metadata`. |
| `local_store(ref)`, `store_path(ref)` | A `LocalStore` (or its path) for `store`/`path` bindings. |
| `client(ref)`, `async_client(ref)` | A `RemoteStore` or `AsyncRemoteStore` for `url` bindings; both accept `timeout=` and `transport=`. |
| `refs()` | The bound refs, sorted. |

Malformed refs and unbound refs raise `ChannelRefError`. `store` and `path` are
both folder paths; optional `[stores.<name>]` tables are exposed through
`channels.stores()` but are not looked up by bindings.

A ref resolves to a whole store; the channel inside it is the ref's last
segment. To get a store for any ref without caring which kind of binding it
has, use `lllm.Resolver`, which reads tactic, channel, and snapshot bindings
from the same file:

```python
from lllm import Resolver

resolver = Resolver.from_config(".")
ref = "lllm://demo/support-desk/channels/triaged"

store = resolver.store(ref)            # LocalStore or RemoteStore, per the binding
print(type(store).__name__, ChannelRef(ref).name)
print(type(resolver.resolve("lllm://demo/support-desk/channels/inbox")).__name__)
```

```text
RemoteStore triaged
LocalStore
```

`resolver.resolve(ref)` returns a tactic for `/tactics/` refs and a store for
`/channels/` and `/snapshots/` refs. [Tutorial 11](../tutorials/11-channels.md)
uses it to point a worker at a served store.

## Package metadata

A package declares its channels and snapshots in `lllm.toml` (see
[The manifest](../packages/manifest.md)):

```toml
[channels.inbox]
form = "log"
description = "Support messages as they arrive."

[channels.triaged]
form = "log"
description = "One triage result per message."

[snapshots.latest-reply]
channel = "triaged"
description = "The newest drafted reply."
```

`open_package_store(manifest, *, store=None)` opens the package's store,
`<package>/.lllm/network` by default, and creates every declared channel that
does not exist yet. For a package folder `support-desk/` whose `lllm.toml` has
a `[package]` table and the sections above:

```python
from lllm.hub import load_manifest
from lllm.network.resources import open_package_store

manifest = load_manifest("support-desk")
store = open_package_store(manifest)
print(store.root.relative_to(manifest.base_dir), [channel.name for channel in store.list_channels()])
```

```text
.lllm/network ['inbox', 'triaged']
```

`lllm serve` uses it to serve a package's channels as a store:

```bash
lllm serve support-desk --resource channels.inbox --port 7700
```

That serves the whole package store (every declared channel), not only
`inbox`; `--store DIR` picks another folder.

`channel_resource` and `snapshot_resource` go the other way: they turn models
into the dictionaries the [hub](../packages/hub.md) indexes, including custom
endpoint metadata:

```python
import json

from lllm.network import Channel, Snapshot, channel_resource, snapshot_resource

print(json.dumps(channel_resource(
    Channel(name="inbox", form="log", description="Support messages as they arrive."),
    custom_endpoints=[count_events],
), indent=2))
print(snapshot_resource(Snapshot(name="latest-reply", channel="triaged"), description="The newest drafted reply."))
```

```text
{
  "name": "inbox",
  "schema": null,
  "form": "log",
  "description": "Support messages as they arrive.",
  "metadata": {},
  "endpoints": [
    {
      "name": "count_events",
      "method": "GET",
      "path": "/channels/{name}/count",
      "scope": "channel",
      "description": "Count every event in a channel.",
      "tags": [
        "stats"
      ]
    }
  ]
}
{'name': 'latest-reply', 'schema': None, 'channel': 'triaged', 'description': 'The newest drafted reply.', 'metadata': {}, 'endpoints': []}
```

## CLI

`lllm network` works on a local store folder. `--store DIR` (default
`.lllm/network`) goes before the command, as in
`lllm network --store ../other/.lllm/network channels`; the folder is created if
it does not exist. Every command prints one JSON object per line; errors print
a message and exit with status 2.

```bash
lllm network create-channel inbox --description "Support messages."
lllm network append inbox '{"text": "Checkout is down"}' --kind message --source cli --correlation-id T-100
lllm network create-subscription inbox --id dashboard --kind message
lllm network pull-subscription dashboard
```

```text
{"name":"inbox","schema":null,"form":"log","description":"Support messages.","metadata":{}}
{"id":"a445d7b4-acff-4b8f-9221-13c54195ad86","cursor":1,"channel":"inbox","timestamp":1793846753.0231912,"source":"cli","kind":"message","payload":{"text":"Checkout is down"},"schema":null,"metadata":{},"correlation_id":"T-100","parent_ids":[]}
{"id":"dashboard","channel":"inbox","cursor":0,"consumer":null,"batch_size":100,"filters":{"kind":"message"},"metadata":{}}
{"id":"a445d7b4-acff-4b8f-9221-13c54195ad86","cursor":1,"channel":"inbox","timestamp":1793846753.0231912,"source":"cli","kind":"message","payload":{"text":"Checkout is down"},"schema":null,"metadata":{},"correlation_id":"T-100","parent_ids":[]}
```

| Command | Arguments and options |
| --- | --- |
| `channels` | |
| `create-channel NAME` | `--schema`, `--form` (default `log`), `--description`, `--metadata JSON` |
| `get-channel NAME` | |
| `append CHANNEL PAYLOAD_JSON` | `--kind` (default `event`), `--source`, `--schema`, `--timestamp`, `--metadata JSON`, `--correlation-id`, `--parent-id ID` (repeatable) |
| `query-events CHANNEL` | `--after-cursor` (default 0), `--limit` (default 100), `--kind` |
| `get-event ID` | |
| `create-subscription CHANNEL` | `--id`, `--consumer`, `--batch-size` (default 100), `--kind` |
| `get-subscription ID` | |
| `pull-subscription ID` | `--limit` |
| `write-artifact DATA` | `--encoding text\|base64`, `--channel`, `--media-type`, `--metadata JSON`, `--event-id ID` (repeatable) |
| `get-artifact ID` | prints metadata |
| `read-artifact ID` | `--encoding text\|base64`; prints the payload |
| `put-snapshot NAME VALUE_JSON` | `--channel`, `--schema`, `--source-event-id`, `--metadata JSON` |
| `get-snapshot NAME` | |
| `serve` | `--host` (default `127.0.0.1`), `--port` (default `7700`), `--log-level` |

See the [CLI reference](../reference/cli.md) for the other `lllm` commands.
