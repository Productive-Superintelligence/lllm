# 11. Channels with SSSN

**Goal:** connect a tactic to a stream of events. You will keep support
messages in a channel, triage each one with a tactic through `TacticProcessor`,
serve the store over HTTP, continue the work from another process, and finally
mount the tactic, the channels, and the processor in one AAAX strategy.

**You need:** `pip install "lllm-core[server,client]"` and `curl`. Nothing here
calls a model, so it all runs offline. Steps 5 to 7 start servers; keep a
second terminal open in the same folder.

Work in an empty folder:

```bash
mkdir support-desk && cd support-desk
```

## Step 1: create a store and two channels

`make_store.py`:

```python
from lllm.network import ChannelExistsError, LocalStore

store = LocalStore(".lllm/network")

for name, description in [
    ("inbox", "Support messages as they arrive."),
    ("triaged", "One triage result per message."),
]:
    try:
        store.create_channel({"name": name, "form": "log", "description": description})
    except ChannelExistsError:
        pass

print([channel.name for channel in store.list_channels()])
```

```bash
python make_store.py
```

```text
['inbox', 'triaged']
```

The whole store is one folder: SQLite for channels, events, and cursors, and a
directory for artifact files.

```text
.lllm/network/
  network.sqlite
  artifacts/
```

Catching `ChannelExistsError` makes the script safe to run twice.

## Step 2: append events

Every message is an `Event`: a payload plus the fields that let other systems
route and trace it. `send.py`:

```python
from lllm.network import Event, LocalStore

store = LocalStore(".lllm/network")

messages = {
    "T-100": "The checkout page is down for every customer",
    "T-101": "How do I change the email on my account?",
    "T-102": "Invoices from March show the wrong tax rate",
}
for ticket, text in messages.items():
    event = store.append_event(
        Event(
            channel="inbox",
            source="web-form",
            kind="message",
            payload={"text": text},
            correlation_id=ticket,
        )
    )
    print(event.cursor, event.correlation_id, event.payload["text"])
```

```bash
python send.py
```

```text
1 T-100 The checkout page is down for every customer
2 T-101 How do I change the email on my account?
3 T-102 Invoices from March show the wrong tax rate
```

The store assigns each event a `cursor`, its position in the store. The
`correlation_id` is the ticket number; it will follow the message into every
channel it reaches.

The CLI writes to the same store (`.lllm/network` is its default):

```bash
lllm network append inbox '{"text": "Password reset emails never arrive"}' \
  --kind message --source cli --correlation-id T-103
```

```text
{"id":"d5339382-115a-4318-a665-b94613f8d575","cursor":4,"channel":"inbox","timestamp":1793846679.3149977,"source":"cli","kind":"message","payload":{"text":"Password reset emails never arrive"},"schema":null,"metadata":{},"correlation_id":"T-103","parent_ids":[]}
```

## Step 3: write a tactic

The tactic triages one message. It is plain Python so this tutorial runs
offline; a native tactic that calls a model plugs in the same way, because the
processor only calls `run()`. `tactics.py`:

```python
import re

from pydantic import BaseModel

from lllm import Tactic

URGENT = re.compile(r"\b(down|outage|failing|broken)\b", re.IGNORECASE)


class Ticket(BaseModel):
    text: str


class Triage(BaseModel):
    priority: str
    summary: str


class TriageTicket(Tactic[Ticket, Triage]):
    name = "triage"
    description = "Assign a priority and a short summary to a support message."
    input_type = Ticket
    output_type = Triage

    def _run(self, input_value, *, context=None):
        text = input_value.text
        priority = "urgent" if URGENT.search(text) else "normal"
        return Triage(priority=priority, summary=" ".join(text.split()[:6]))
```

Try it directly:

```bash
python -c '
from tactics import TriageTicket

print(TriageTicket().run({"text": "The API is down"}))
print(TriageTicket().run({"text": "Where can I download my invoices?"}))
'
```

```text
priority='urgent' summary='The API is down'
priority='normal' summary='Where can I download my invoices?'
```

## Step 4: run the tactic over the channel

`TacticProcessor` reads `inbox` through a named subscription, runs the tactic
on each message, and appends the result to `triaged`. `process.py`:

```python
from lllm.network import LocalStore, TacticProcessor
from tactics import TriageTicket

store = LocalStore(".lllm/network")
processor = TacticProcessor(
    TriageTicket(),
    store,
    source="inbox",
    sink="triaged",
    kind="message",
)

for event in processor.process():
    print(event.cursor, event.correlation_id, event.payload)
```

```bash
python process.py
```

```text
5 T-100 {'priority': 'urgent', 'summary': 'The checkout page is down for'}
6 T-101 {'priority': 'normal', 'summary': 'How do I change the email'}
7 T-102 {'priority': 'normal', 'summary': 'Invoices from March show the wrong'}
8 T-103 {'priority': 'normal', 'summary': 'Password reset emails never arrive'}
```

Run it again and it prints nothing. The processor's subscription,
`triage-processor` (the tactic's name plus `-processor`), lives in the store
and remembers the last event it handed out:

```bash
python process.py
lllm network get-subscription triage-processor
```

```text
{"id":"triage-processor","channel":"inbox","cursor":4,"consumer":"triage-processor","batch_size":100,"filters":{"kind":"message"},"metadata":{}}
```

Each result points back at the message it came from. `trace.py`:

```python
from lllm.network import LocalStore

store = LocalStore(".lllm/network")

result = store.query_events("triaged", limit=1)[0]
origin = store.get_event(result.parent_ids[0])
print(result.source, result.kind, result.correlation_id)
print(origin.channel, origin.payload["text"])
```

```bash
python trace.py
```

```text
triage-processor result T-100
inbox The checkout page is down for every customer
```

## Step 5: serve the store

In the second terminal, from the `support-desk` folder:

```bash
lllm network serve --port 7700
```

The store API is now at `http://127.0.0.1:7700`, with interactive docs at
`/docs`:

```bash
curl -s -w '\n' http://127.0.0.1:7700/health
curl -s -w '\n' "http://127.0.0.1:7700/events?channel=triaged&limit=1"
```

```text
{"ok":true}
[{"id":"ea41698d-0f45-4afa-b745-4604de002705","cursor":5,"channel":"triaged","timestamp":1793846681.0239878,"source":"triage-processor","kind":"result","payload":{"priority":"urgent","summary":"The checkout page is down for"},"schema":null,"metadata":{},"correlation_id":"T-100","parent_ids":["381bb4ae-10dc-4c09-846f-15d7f70b4651"]}]
```

The server opens the same folder, so local scripts and HTTP clients can share
the store.

## Step 6: read it from another process

`RemoteStore` has the same methods as `LocalStore`. A dashboard that shows new
triage results keeps its own subscription. `dashboard.py`:

```python
from lllm.network import RemoteStore

store = RemoteStore("http://127.0.0.1:7700")
store.create_subscription("triaged", subscription_id="dashboard")

for event in store.pull_subscription("dashboard"):
    print(event.correlation_id, event.payload["priority"], event.payload["summary"])
```

```bash
python dashboard.py
```

```text
T-100 urgent The checkout page is down for
T-101 normal How do I change the email
T-102 normal Invoices from March show the wrong
T-103 normal Password reset emails never arrive
```

Run it again and it prints nothing new. `create_subscription` with an existing
id returns the existing subscription, so the script is safe to rerun.

Now move the worker off the local store. Instead of hard-coding the URL, bind
the channels to the server in `.lllm/config.toml`:

```toml
[refs."lllm://demo/support-desk/channels/inbox"]
url = "http://127.0.0.1:7700"

[refs."lllm://demo/support-desk/channels/triaged"]
url = "http://127.0.0.1:7700"
```

`worker.py` asks `lllm.Resolver` for the store behind the ref:

```python
from lllm import Resolver
from lllm.network import TacticProcessor
from tactics import TriageTicket

resolver = Resolver.from_config(".")
store = resolver.store("lllm://demo/support-desk/channels/inbox")

processor = TacticProcessor(
    TriageTicket(), store, source="inbox", sink="triaged", kind="message"
)
print(type(store).__name__)
for event in processor.process():
    print(event.correlation_id, event.payload)
```

Send a message over plain HTTP, then run the worker and the dashboard:

```bash
curl -s -w '\n' -X POST http://127.0.0.1:7700/events \
  -H 'content-type: application/json' \
  -d '{"channel": "inbox", "kind": "message", "source": "curl", "correlation_id": "T-104", "payload": {"text": "Login is down on the mobile app"}}'
python worker.py
python dashboard.py
```

```text
{"id":"b969625d-89c8-4916-802c-746f9dedd11d","cursor":9,"channel":"inbox","timestamp":1793846688.7259398,"source":"curl","kind":"message","payload":{"text":"Login is down on the mobile app"},"schema":null,"metadata":{},"correlation_id":"T-104","parent_ids":[]}
RemoteStore
T-104 {'priority': 'urgent', 'summary': 'Login is down on the mobile'}
T-104 urgent Login is down on the mobile
```

The worker processed only T-104. It used the same `triage-processor`
subscription as step 4, and that subscription lives in the store, not in the
process, so a worker on another machine picks up exactly where the local one
stopped. To point the worker back at the local folder, change the binding to
`store = ".lllm/network"`; the code stays the same.

## Step 7: mount it all in an AAAX strategy

Describe the folder as a package. `lllm.toml`:

```toml
[package]
org = "demo"
name = "support-desk"
version = "0.1.0"
kind = "app"
description = "Triage support messages."

[tactics.triage]
entry = "tactics:TriageTicket"
description = "Assign a priority and a short summary."

[channels.inbox]
form = "log"
description = "Support messages as they arrive."

[channels.triaged]
form = "log"
description = "One triage result per message."
```

The shell reads it without starting anything:

```bash
lllm shell inspect .
```

```text
support-desk: Triage support messages.
package support-desk lllm://demo/support-desk
tactic triage lllm://demo/support-desk/tactics/triage
channel inbox lllm://demo/support-desk/channels/inbox
channel triaged lllm://demo/support-desk/channels/triaged
```

A strategy adds what the manifest cannot say: what `/run` should do. Here it
appends the request to `inbox` and drains the backlog through the processor.
`strategy.py`:

```python
from pathlib import Path

from lllm.network import TacticProcessor
from lllm.shell import Strategy


def build_strategy() -> Strategy:
    strategy = Strategy("support-desk", description="Triage support messages.")
    strategy.use_package(Path(__file__).parent)  # the tactic and channels in lllm.toml

    processor = TacticProcessor(
        strategy.tactics()["triage"],
        strategy.store,
        source="inbox",
        sink="triaged",
        kind="message",
    )

    @strategy.runner
    async def run(input_value, *, context=None):
        strategy.store.append_event(
            {"channel": "inbox", "kind": "message", "source": "shell", "payload": input_value}
        )
        return [event.payload for event in await processor.aprocess()]

    return strategy
```

`use_package` mounts the tactic and both channels and opens the package store,
`.lllm/network` in the package folder: the same store as every step before.
With `strategy.py` present, `lllm shell serve .` loads it instead of the bare
manifest. Stop the store server from step 5 (or leave it running; the two can
share the folder) and start the strategy:

```bash
lllm shell serve . --port 8400
```

From the other terminal, run the strategy, call the tactic on its own, and
queue a message through the channel shortcut:

```bash
curl -s -w '\n' -X POST http://127.0.0.1:8400/run \
  -H 'content-type: application/json' \
  -d '{"input": {"text": "Refunds are failing for every customer"}}'
curl -s -w '\n' -X POST http://127.0.0.1:8400/tactics/triage/run \
  -H 'content-type: application/json' \
  -d '{"input": {"text": "Where can I download my invoices?"}}'
curl -s -w '\n' -X POST http://127.0.0.1:8400/channels/inbox/events \
  -H 'content-type: application/json' \
  -d '{"input": {"kind": "message", "correlation_id": "T-105", "payload": {"text": "Search results are broken"}}}'
```

```text
{"output":[{"priority":"urgent","summary":"Refunds are failing for every customer"}],"strategy":"support-desk","metadata":{}}
{"output":{"priority":"normal","summary":"Where can I download my invoices?"},"request_id":"db41a404-25e8-446c-be59-aa7345fc76a4","tactic":"triage"}
{"output":{"id":"4f3b1f3f-a0c3-41ba-a9e9-06ba8289f381","cursor":13,"channel":"inbox","timestamp":1793846694.8979056,"source":"lllm.shell","kind":"message","payload":{"text":"Search results are broken"},"schema":null,"metadata":{},"correlation_id":"T-105","parent_ids":[]},"strategy":"support-desk","metadata":{"resource":"inbox","endpoint":"channel"}}
```

The tactic route ran the tactic without touching the channels. The queued
message waits in `inbox` until the next `/run` drains it along with the new
request:

```bash
curl -s -w '\n' -X POST http://127.0.0.1:8400/run \
  -H 'content-type: application/json' \
  -d '{"input": {"text": "Thanks, all good now"}}'
curl -s -w '\n' http://127.0.0.1:8400/network/subscriptions/triage-processor
```

```text
{"output":[{"priority":"urgent","summary":"Search results are broken"},{"priority":"normal","summary":"Thanks, all good now"}],"strategy":"support-desk","metadata":{}}
{"id":"triage-processor","channel":"inbox","cursor":14,"consumer":"triage-processor","batch_size":100,"filters":{"kind":"message"},"metadata":{}}
```

One server now offers the strategy's `/run`, the tactic at `/tactics/triage/`,
channel shortcuts at `/channels/<name>/events`, and the full store API at
`/network/`. `GET /strategy` lists everything that is mounted. Stop the
servers with Ctrl+C when you are done.

## Summary

| You want | Use |
| --- | --- |
| A store on disk | `LocalStore(".lllm/network")`, `create_channel` |
| To record something that happened | `append_event(Event(...))` or `lllm network append` |
| To run a tactic over every new event | `TacticProcessor(tactic, store, source=..., sink=...).process()` |
| To trace a result to its input | `parent_ids`, `correlation_id`, `get_event` |
| To share a store between processes | `lllm network serve`, `RemoteStore`, `AsyncRemoteStore` |
| To keep URLs out of code | `.lllm/config.toml` refs and `lllm.Resolver` |
| One API for tactics, channels, and workflows | `lllm.toml`, a `Strategy` with a runner, `lllm shell serve` |

Read more in [SSSN](../sssn/index.md) ([channels](../sssn/channels.md),
[processing](../sssn/processing.md), [serving](../sssn/serving.md)) and
[AAAX](../aaax/index.md).

**Next:** [Tutorials index](index.md)
