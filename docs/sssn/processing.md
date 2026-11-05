# Processing channels with tactics

`TacticProcessor` runs a tactic over the events of one channel. For each
pending event it calls the tactic with the event's payload, which the tactic
validates against its input type, and appends the output to a sink channel.
Each output points back at its input through `parent_ids` and carries the
input's `correlation_id`, so you can chain processors into pipelines and still
trace every result to where it came from.

```text
inbox ──subscription──▶ TacticProcessor(tactic) ──append──▶ triaged
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `tactic` | required | Any lllm `Tactic`: a subclass, `as_tactic(fn)`, a native tactic with agents, a `RemoteTactic`. |
| `store` | required | `LocalStore` or `RemoteStore` for `process()`; any store, including `AsyncRemoteStore`, for `aprocess()`. |
| `source` | required | Channel to read. |
| `sink` | `None` | Channel to append outputs to. Without one, outputs are returned instead. |
| `consumer` | `"<tactic.name>-processor"` | Subscription id, and the `source` field of output events. |
| `kind` | `None` | Only process source events of this kind. |
| `output_kind` | `"result"` | `kind` of the output events. |
| `errors` | `"raise"` | `"raise"` stops on the first failure; `"record"` writes an error event and continues. |

The examples on this page run top to bottom as one session.

## One stage

```python
import re

from pydantic import BaseModel

from lllm import Tactic
from lllm.network import Event, LocalStore, TacticProcessor

URGENT = re.compile(r"\b(down|outage|failing|broken)\b", re.IGNORECASE)


class Ticket(BaseModel):
    text: str


class Triage(BaseModel):
    priority: str
    summary: str


class TriageTicket(Tactic[Ticket, Triage]):
    name = "triage"
    input_type = Ticket
    output_type = Triage

    def _run(self, input_value, *, context=None):
        priority = "urgent" if URGENT.search(input_value.text) else "normal"
        return Triage(priority=priority, summary=input_value.text[:40])


store = LocalStore(".lllm/network")
store.create_channel({"name": "inbox"})
store.create_channel({"name": "triaged"})


def submit(text, ticket):
    store.append_event(
        Event(
            channel="inbox",
            source="web-form",
            kind="message",
            payload={"text": text},
            correlation_id=ticket,
        )
    )


submit("Checkout is down", "T-100")
submit("How do I export my data?", "T-101")

processor = TacticProcessor(TriageTicket(), store, source="inbox", sink="triaged")
for event in processor.process():
    print(event.source, event.kind, event.correlation_id, event.payload)
```

```text
triage-processor result T-100 {'priority': 'urgent', 'summary': 'Checkout is down'}
triage-processor result T-101 {'priority': 'normal', 'summary': 'How do I export my data?'}
```

`process()` returns the sink events it wrote. Each one holds:

| Field | Value |
| --- | --- |
| `channel` | The sink. |
| `source` | The consumer name. |
| `kind` | `output_kind` (`"result"`). |
| `payload` | The tactic output as JSON. Pydantic models are dumped with `model_dump(mode="json")`. |
| `parent_ids` | `(input_event.id,)` |
| `correlation_id` | The input's `correlation_id`, or the input's `id` when it has none. |

Create the sink channel before processing. The processor does not create
channels.

## What the tactic sees

The tactic gets the event's payload as its input and a `CallContext` that
describes the event:

| Context field | Value |
| --- | --- |
| `caller` | The consumer name. |
| `tags` | `{"channel": <source>}` |
| `metadata["event_id"]` | The input event's id. |
| `metadata["event_kind"]` | The input event's kind. |
| `metadata["correlation_id"]` | The input's correlation id, or its id. |

```python
from lllm import as_tactic


def show_context(payload: dict, *, context=None) -> dict:
    return {
        "caller": context.caller,
        "tags": context.tags,
        "event_kind": context.metadata["event_kind"],
        "correlation_id": context.metadata["correlation_id"],
    }


peek = TacticProcessor(as_tactic(show_context), store, source="inbox", consumer="peek")
print(peek.process(limit=1))
```

```text
[{'caller': 'peek', 'tags': {'channel': 'inbox'}, 'event_kind': 'message', 'correlation_id': 'T-100'}]
```

## Resuming

The consumer name is the id of a subscription that lives in the store. The
first `process()` call creates it; every later processor with the same
consumer, in this process or another one, continues from its cursor:

```python
submit("Search is broken", "T-102")

restarted = TacticProcessor(TriageTicket(), store, source="inbox", sink="triaged")
print([event.correlation_id for event in restarted.process()])
print(store.get_subscription("triage-processor").cursor)
```

```text
['T-102']
5
```

Rules that follow from this:

- **Name consumers explicitly** once you run more than one processor. The
  default is built from the tactic class's `name` attribute. Tactics named only
  when constructed, such as `as_tactic(fn)`, have no class-level name and all
  default to `None-processor`, so two of them would share one subscription.
- To process a channel again from the start, use a new consumer name.
- The `kind` filter is stored with the subscription when it is created.
  Changing `kind` later for the same consumer has no effect; use a new
  consumer name.
- Run one worker per consumer name. Two workers sharing a consumer can receive
  the same event.

## Failures

`process()` pulls one event at a time, and the subscription cursor moves when
the event is pulled, before the tactic runs. Delivery is therefore at most
once: an event whose processing fails is not handed out again. `errors`
decides what happens next.

With `errors="raise"` (the default), the exception propagates and the events
behind the failing one stay pending. The failing event itself is consumed:

```python
from lllm import SchemaError

store.append_event(
    Event(channel="inbox", kind="message", payload={"subject": "no text field"}, correlation_id="T-103")
)
submit("The API is down", "T-104")

try:
    processor.process()
except SchemaError as exc:
    print("failed:", type(exc).__name__)

print([event.correlation_id for event in processor.process()])
```

```text
failed: SchemaError
['T-104']
```

T-103 failed input validation and is not retried. With `errors="record"`, the
failure is written to the sink as a `kind="error"` event with the same lineage
fields, and processing continues:

```python
store.append_event(
    Event(channel="inbox", kind="message", payload={"subject": "still no text"}, correlation_id="T-105")
)

recording = TacticProcessor(
    TriageTicket(), store, source="inbox", sink="triaged", errors="record"
)
[error] = recording.process()
print(error.kind, error.correlation_id, error.payload["type"])
```

```text
error T-105 SchemaError
```

The error payload is `{"error": str(exc), "type": type(exc).__name__}`. Use
`"record"` in long-running workers so failures stay visible in the store, and
read them back with `store.query_events("triaged", kind="error")`. `"record"`
needs a sink; without one, errors are raised.

Any exception counts as a failure: payloads that do not match the tactic's
input type raise `SchemaError`, and the tactic's own exceptions pass through.
A missing sink channel also fails after the event was pulled, which is another
reason to create channels up front.

## Without a sink

With no `sink`, `process()` returns the tactic outputs themselves. That is
useful for dry runs, or when the results go somewhere other than a channel.
`limit` caps how many results one call returns:

```python
dry_run = TacticProcessor(TriageTicket(), store, source="inbox", consumer="dry-run")
print(dry_run.process(limit=2))
```

```text
[Triage(priority='urgent', summary='Checkout is down'), Triage(priority='normal', summary='How do I export my data?')]
```

`limit` counts outputs and recorded errors alike. `process()` with no limit
keeps going until the subscription has nothing pending.

## Remote and async stores

The processor only calls store methods, so a store served over HTTP works the
same way. With `lllm network serve --port 7700` running on this store:

```python
from lllm.network import RemoteStore

remote = RemoteStore("http://127.0.0.1:7700")
remote.append_event(
    {"channel": "inbox", "kind": "message", "payload": {"text": "Exports time out"}, "correlation_id": "T-106"}
)

worker = TacticProcessor(TriageTicket(), remote, source="inbox", sink="triaged")
print([event.correlation_id for event in worker.process()])
```

```text
['T-106']
```

The remote worker continued the `triage-processor` subscription where the
local processors left it.

`aprocess()` is the async variant. It calls `tactic.arun()` and awaits store
calls that return coroutines, so it works with `AsyncRemoteStore` as well as
the sync stores. `process()` needs a sync store. This example talks to the
store server in-process through `httpx.ASGITransport`, which is also how to
test a worker without a running server; in production, pass only the URL.

```python
import asyncio

import httpx

from lllm.network import AsyncRemoteStore, create_app


async def main():
    remote = AsyncRemoteStore(
        "http://store.local", transport=httpx.ASGITransport(app=create_app(store))
    )
    await remote.append_event(
        {"channel": "inbox", "kind": "message", "payload": {"text": "Uploads are failing"}, "correlation_id": "T-107"}
    )
    worker = TacticProcessor(TriageTicket(), remote, source="inbox", sink="triaged")
    return [event.correlation_id for event in await worker.aprocess()]


print(asyncio.run(main()))
```

```text
['T-107']
```

## Pipelines

A pipeline is processors whose sink is the next one's source. Add a stage that
drafts a reply for every triage result. The `kind="result"` filter skips the
error events the first stage recorded:

```python
class Reply(BaseModel):
    priority: str
    body: str


def draft_reply(result: Triage) -> Reply:
    opening = "We are on it." if result.priority == "urgent" else "Thanks for writing in."
    return Reply(priority=result.priority, body=f"{opening} Re: {result.summary}")


store.create_channel({"name": "replies"})

stages = [
    TacticProcessor(TriageTicket(), store, source="inbox", sink="triaged", errors="record"),
    TacticProcessor(
        as_tactic(draft_reply),
        store,
        source="triaged",
        sink="replies",
        consumer="draft-reply",
        kind="result",
    ),
]


def drain(processors):
    """Run every stage until none of them finds new events."""
    while any([processor.process() for processor in processors]):
        pass


submit("Refunds are failing for every customer", "T-108")
drain(stages)
print([(reply.correlation_id, reply.payload["priority"]) for reply in store.query_events("replies")])
```

```text
[('T-100', 'urgent'), ('T-101', 'normal'), ('T-102', 'urgent'), ('T-104', 'urgent'), ('T-106', 'normal'), ('T-107', 'urgent'), ('T-108', 'urgent')]
```

The new ticket went through both stages in one `drain`; the earlier results
reached the second stage because its `draft-reply` subscription started at the
beginning of `triaged`.

### Tracing a result

`parent_ids` lead from any output back to the event that started the chain:

```python
def lineage(store, event):
    """Follow parent_ids back to the event that started the chain."""
    chain = [event]
    while chain[-1].parent_ids:
        chain.append(store.get_event(chain[-1].parent_ids[0]))
    return chain


for event in lineage(store, store.query_events("replies")[-1]):
    print(f"{event.channel:8} {event.source:17} {event.payload}")
```

```text
replies  draft-reply       {'body': 'We are on it. Re: Refunds are failing for every customer', 'priority': 'urgent'}
triaged  triage-processor  {'priority': 'urgent', 'summary': 'Refunds are failing for every customer'}
inbox    web-form          {'text': 'Refunds are failing for every customer'}
```

`correlation_id` groups the whole thread across channels without walking
links:

```python
print([
    (event.channel, event.kind)
    for channel in ("inbox", "triaged", "replies")
    for event in store.query_events(channel)
    if event.correlation_id == "T-108"
])
```

```text
[('inbox', 'message'), ('triaged', 'result'), ('replies', 'result')]
```

### Fan-out

Each consumer has its own cursor, so several processors can read the same
source and each sees every event:

```python
def word_count(payload: dict) -> dict:
    return {"words": len(payload.get("text", "").split())}


store.create_channel({"name": "stats"})
stats = TacticProcessor(
    as_tactic(word_count), store, source="inbox", sink="stats", consumer="word-count"
)
print(len(stats.process()), "of", len(store.query_events("inbox")), "inbox events counted")
```

```text
9 of 9 inbox events counted
```

### Latest state

To give readers the newest result without replaying a channel, write a
snapshot that points at the event it came from:

```python
latest = store.query_events("replies")[-1]
store.put_snapshot(
    {"name": "latest-reply", "channel": "replies", "value": latest.payload, "source_event_id": latest.id}
)
print(store.get_snapshot("latest-reply").value["body"])
```

```text
We are on it. Re: Refunds are failing for every customer
```

## Running a worker

A worker is a loop around `process()` that sleeps when there is nothing to do:

```python
import time


def run_worker(processor, *, idle_seconds=1.0):
    while True:
        if not processor.process(limit=100):
            time.sleep(idle_seconds)
```

The async version, for an `AsyncRemoteStore` or an event loop you already run:

```python
import asyncio


async def run_async_worker(processor, *, idle_seconds=1.0):
    while True:
        if not await processor.aprocess(limit=100):
            await asyncio.sleep(idle_seconds)
```

Give each worker its own consumer name and `errors="record"`, and stop it at
any point: the next start resumes from the subscription cursor. To serve the
store these workers share, see [Serving](serving.md); to put workers, tactics,
and channels behind one API, see [AAAX](../aaax/index.md).
