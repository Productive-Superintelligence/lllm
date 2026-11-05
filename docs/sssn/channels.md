# Channels and stores

A store holds five kinds of resources: channels, events, subscriptions,
artifacts, and snapshots. This page walks through each one with `LocalStore`.
`RemoteStore` and `AsyncRemoteStore` expose the same methods over HTTP; see
[Serving](serving.md) for the few differences.

The examples run top to bottom as one session.

```python
from lllm.network import LocalStore

store = LocalStore(".lllm/network")   # the folder is created if missing
```

`LocalStore()` with no argument uses `.lllm/network` relative to the working
directory.

## Channels

A channel is a named stream. Create one from a `Channel` model or a plain dict:

```python
from lllm.network import Channel

store.create_channel(
    Channel(
        name="inbox",
        schema="schemas.ticket",
        form="queue",
        description="Support messages as they arrive.",
        metadata={"owner": "support"},
    )
)
store.create_channel({"name": "triaged", "description": "One result per message."})

print([channel.name for channel in store.list_channels()])
print(store.get_channel("inbox").form)
```

```text
['inbox', 'triaged']
queue
```

| Field | Meaning |
| --- | --- |
| `name` | Unique in the store. Must be a path segment: no whitespace, `/`, `\`, `:`, `%`, and not `.` or `..`. Letters, digits, `-`, `_`, and `.` are safe. |
| `schema` | Optional schema ref, such as a manifest key (`schemas.ticket`) or an import path. Recorded, not enforced. |
| `form` | How the channel is meant to be used. One of `log` (default), `queue`, `topic`, `blackboard`, `latest-state`, `artifact-index`, `pipeline-edge`, `database-view`, `feed`, `time-series`, `tree`, `forum`, `hub`. |
| `description` | Free text for people and tools. |
| `metadata` | Any JSON object. |

The form is a declaration for readers and tools. `LocalStore` keeps every
channel as an append-only log whatever its form. Channels cannot be updated or
deleted through the API.

Creating a channel that already exists raises `ChannelExistsError`, so setup
code that may run twice catches it:

```python
from lllm.network import ChannelExistsError

try:
    store.create_channel({"name": "inbox"})
except ChannelExistsError as exc:
    print(exc)
```

```text
Channel already exists: inbox
```

## Events

An event is one record in one channel. Events are append-only.

```python
from lllm.network import Event

ticket = store.append_event(
    Event(
        channel="inbox",
        source="web-form",
        kind="message",
        payload={"text": "Checkout is down"},
        correlation_id="T-100",
        metadata={"locale": "en"},
    )
)
print(ticket.cursor, ticket.kind, ticket.correlation_id)
```

```text
1 message T-100
```

| Field | Default | Meaning |
| --- | --- | --- |
| `id` | random UUID | Unique in the store. You may set your own (a path segment); appending an id that exists raises `EventExistsError`. |
| `cursor` | assigned by the store | Position in the store; see [Cursors](#reading-with-cursors). |
| `channel` | required | Must exist. |
| `timestamp` | now | Seconds since the epoch, as a float. |
| `source` | `None` | Who wrote it: a service, a processor, `cli`. |
| `kind` | `"event"` | A token that consumers filter on (`message`, `result`, `error`, ...). |
| `payload` | `None` | Any JSON value. Pydantic models are stored as JSON and read back as dicts. |
| `schema` | `None` | Optional schema ref for the payload. |
| `metadata` | `{}` | JSON object for small hints. |
| `correlation_id` | `None` | Groups related events across channels: a ticket, a run, an episode. A token. |
| `parent_ids` | `()` | Ids of the events this one was derived from, in any channel. |

`append_event` returns the stored event with its `cursor` filled in. Pass a
dict instead of an `Event` if you prefer; the same validation applies.

### Lineage

`parent_ids` and `correlation_id` make derived data traceable. The store
rejects parent ids that do not exist, so every link resolves:

```python
result = store.append_event(
    Event(
        channel="triaged",
        source="triage",
        kind="result",
        payload={"priority": "urgent"},
        correlation_id=ticket.correlation_id,
        parent_ids=(ticket.id,),
    )
)
print(store.get_event(result.parent_ids[0]).payload)
```

```text
{'text': 'Checkout is down'}
```

`TacticProcessor` fills both fields for you; see [Processing](processing.md).

## Reading with cursors

`query_events(channel, *, after_cursor=0, limit=100, kind=None)` returns events
with a cursor greater than `after_cursor`, oldest first:

```python
store.append_event({"channel": "inbox", "kind": "message", "payload": {"text": "How do I export data?"}})
store.append_event({"channel": "inbox", "kind": "ping", "payload": {}})

print([(event.cursor, event.kind) for event in store.query_events("inbox")])
print([event.cursor for event in store.query_events("inbox", kind="message")])
print([event.cursor for event in store.query_events("inbox", after_cursor=ticket.cursor, limit=1)])
```

```text
[(1, 'message'), (3, 'message'), (4, 'ping')]
[1, 3]
[3]
```

Cursor 2 is missing from `inbox` because it went to the event in `triaged`.
Cursors increase across the whole store, so inside one channel they are
ordered but have gaps. Never compute `cursor + 1`; resume from the last cursor
you saw:

```python
def read_all(store, channel, *, page=500):
    cursor = 0
    while batch := store.query_events(channel, after_cursor=cursor, limit=page):
        yield from batch
        cursor = batch[-1].cursor


print(sum(1 for _ in read_all(store, "inbox", page=2)))
```

```text
3
```

`after_cursor` must be zero or more and `limit` at least 1; anything else
raises `InvalidPayloadError`. `get_event(event_id)` reads one event by id.

## Subscriptions

A subscription is a cursor that the store keeps for a consumer, so the consumer
can stop and pick up where it left off, from any process.

```python
store.create_subscription(
    "inbox",
    subscription_id="dashboard",
    consumer="dashboard",
    batch_size=1,
    filters={"kind": "message"},
)

print([event.payload["text"] for event in store.pull_subscription("dashboard")])
print(store.get_subscription("dashboard").cursor)
print([event.payload["text"] for event in store.pull_subscription("dashboard", limit=10)])
print(store.pull_subscription("dashboard"))
```

```text
['Checkout is down']
1
['How do I export data?']
()
```

How subscriptions behave:

- `pull_subscription(id, *, limit=None)` returns up to `limit` (default: the
  subscription's `batch_size`) events after the stored cursor and moves the
  cursor past them in the same call. A pulled event is delivered once, whether
  or not the consumer finishes with it. `TacticProcessor` reads ahead with
  `query_events` instead and pulls an event only after writing its output;
  see [Processing](processing.md#failures).
- `filters` supports one key, `kind`. Other keys raise `InvalidPayloadError`.
- Without `subscription_id`, the store generates a UUID. Name subscriptions you
  want to resume.
- `get_subscription(id)` reads the current state without moving the cursor.

Creating a subscription with an id that already exists on the same channel
returns the existing one unchanged, cursor and settings included. That makes
`create_subscription` safe to call on every start:

```python
same = store.create_subscription("inbox", subscription_id="dashboard", batch_size=50)
print(same.cursor, same.batch_size)
```

```text
3 1
```

The cursor sits at 3, the last `message` event; the `ping` at cursor 4 never
matched the filter. To change the filter or batch size, use a new id. Reusing
an id on a different channel raises `SubscriptionExistsError`.

## Artifacts

Artifacts hold bytes that do not belong inline in an event: files, images,
reports, model outputs.

```python
artifact = store.write_artifact(
    b"%PDF-1.7 ...",
    channel="inbox",
    media_type="application/pdf",
    metadata={"filename": "march-invoice.pdf"},
    event_ids=(ticket.id,),
)
print(artifact.size, artifact.media_type, artifact.path == f"artifacts/{artifact.sha256}")
print(store.get_artifact(artifact.id).metadata)
print(store.read_artifact(artifact.id))
```

```text
12 application/pdf True
{'filename': 'march-invoice.pdf'}
b'%PDF-1.7 ...'
```

- `write_artifact` takes `bytes`, `bytearray`, or `memoryview`. `channel` and
  every id in `event_ids` must exist.
- `get_artifact` returns the metadata (`Artifact`); `read_artifact` returns the
  bytes.
- Files are content-addressed: the same bytes written twice produce two
  artifact records that share one file.

To let channel readers find an artifact, announce it with an event:

```python
store.append_event(
    {
        "channel": "inbox",
        "kind": "attachment",
        "payload": {"artifact_id": artifact.id, "filename": "march-invoice.pdf"},
        "correlation_id": "T-100",
    }
)
```

## Snapshots

A snapshot is the latest value under a name: a queue depth, a current plan, the
newest summary. Readers get it in one call instead of replaying a channel.

```python
from lllm.network import Snapshot

store.put_snapshot(
    Snapshot(name="queue-depth", channel="inbox", value={"open": 2}, source_event_id=ticket.id)
)
print(store.get_snapshot("queue-depth").value)

store.put_snapshot({"name": "queue-depth", "value": {"open": 1}})
latest = store.get_snapshot("queue-depth")
print(latest.value, latest.channel, latest.source_event_id)
```

```text
{'open': 2}
{'open': 1} None None
```

`put_snapshot` takes a `Snapshot`, a dict of snapshot fields, or a name and a
value with the other fields as keywords, as in
`store.put_snapshot("queue-depth", {"open": 1}, channel="inbox")`. The remote
clients take the same arguments. It replaces the whole snapshot: fields you
leave out are reset, as `channel` and `source_event_id` are above. `channel`
and `source_event_id` must exist when set. Snapshots have no history; keep the
history in a channel and point the snapshot at the event it came from.

## Errors

Every store error is a subclass of `NetworkError`, itself a `RuntimeError`:

```python
from lllm.network import NetworkError

for attempt in (
    lambda: store.get_channel("missing"),
    lambda: store.append_event({"channel": "inbox", "payload": {}, "parent_ids": ["nope"]}),
    lambda: store.append_event({"id": ticket.id, "channel": "inbox", "payload": {}}),
    lambda: store.query_events("inbox", after_cursor=-1),
    lambda: store.create_channel({"name": "bad name"}),
):
    try:
        attempt()
    except NetworkError as exc:
        print(type(exc).__name__)
```

```text
ChannelNotFoundError
EventNotFoundError
EventExistsError
InvalidPayloadError
InvalidPayloadError
```

| Error | Raised when | Store server status |
| --- | --- | --- |
| `ChannelExistsError` | Creating a channel whose name is taken. | 409 |
| `EventExistsError` | Appending an event whose id is taken. | 409 |
| `SubscriptionExistsError` | Reusing a subscription id on a different channel. | 409 |
| `ChannelNotFoundError` | Using a channel that does not exist. | 404 |
| `EventNotFoundError` | Unknown event id, including dangling `parent_ids`, artifact `event_ids`, or snapshot `source_event_id`. | 404 |
| `SubscriptionNotFoundError` | Unknown subscription id. | 404 |
| `ArtifactNotFoundError` | Unknown artifact id, or its file is missing. | 404 |
| `SnapshotNotFoundError` | Unknown snapshot name. | 404 |
| `InvalidPayloadError` | Bad names, negative cursors, non-positive limits, unsupported filters, values that are not JSON serializable. | 400 |
| `ChannelRefError` | Malformed or unbound `lllm://` refs. Also a `ValueError`. | not served |

Over HTTP, `RemoteStore` raises `RemoteStoreError` with the status code and the
error type name instead; see [Serving](serving.md#errors).

## The `LocalStore` layout

```text
.lllm/network/
  network.sqlite          # channels, events, subscriptions, artifact metadata, snapshots
  artifacts/
    cd98c5bce101ab55...   # artifact bytes, named by their SHA-256
```

- An event's cursor is its SQLite `rowid`, which is why cursors are store-wide.
- JSON values are stored with sorted keys, so payload dicts read back with
  their keys in alphabetical order.
- Every call opens its own SQLite connection, so several processes can share a
  store folder, for example a store server and a local worker. SQLite
  serializes their writes.
- Delete the folder to start over.
