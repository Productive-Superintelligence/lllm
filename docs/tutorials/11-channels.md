# 11. SSSN and AAAX

**Goal:** take a tactic beyond request and response. Keep events in channels,
run a tactic over them, and serve a package's tactics and channels as one API.

Channels and strategies live in two companion packages built on LLLM, each
with its own documentation:

- **SSSN** ([sssn.one](https://sssn.one/)): channels, events, artifacts, and
  snapshots, plus a `TacticProcessor` that runs any LLLM tactic over a channel.
- **AAAX** ([aaax.one](https://aaax.one/)): strategies that mount packages,
  tactics, and channels behind one served API.

This lesson shows where they plug into what you built in lessons 1 to 10.
Nothing here calls a model, so it all runs offline. Work in an empty folder:

```bash
mkdir support-desk && cd support-desk
```

## Step 1: install the packages

```bash
pip install sssn aaax     # or: pip install "lllm-core[network,shell]"
lllm --help               # now lists the network and shell commands
```

Installing `sssn` adds `lllm network`, and installing `aaax` adds
`lllm shell`; each package also keeps its own `sssn` or `aaax` command. Without
the package, the `lllm` command says what to install:

```text
lllm network needs the sssn package: pip install sssn
```

## Step 2: write a tactic

Any LLLM tactic works over a channel. `tactics.py`:

```python
from pydantic import BaseModel

from lllm import Tactic


class Ticket(BaseModel):
    text: str


class Triage(BaseModel):
    priority: str


class TriageTicket(Tactic[Ticket, Triage]):
    name = "triage"
    description = "Assign a priority to a support message."
    input_type = Ticket
    output_type = Triage

    def _run(self, input_value, *, context=None):
        urgent = "down" in input_value.text.lower()
        return Triage(priority="urgent" if urgent else "normal")
```

## Step 3: run it over a channel

A store keeps channels of events. `TacticProcessor` reads one channel, runs the
tactic on each event, and appends the results to another. `process.py`:

```python
from sssn import ChannelExistsError, Event, LocalStore, TacticProcessor
from tactics import TriageTicket

store = LocalStore(".lllm/network")
for name in ("inbox", "triaged"):
    try:
        store.create_channel({"name": name})
    except ChannelExistsError:
        pass

store.append_event(
    Event(channel="inbox", kind="message", payload={"text": "Checkout is down"})
)
processor = TacticProcessor(
    TriageTicket(), store, source="inbox", sink="triaged", kind="message"
)
for event in processor.process():
    print(event.payload)
```

```bash
python process.py
```

```text
{'priority': 'urgent'}
```

The processor reads through a named subscription kept in the store, so the
next run handles only new events, and each result links back to its message
through `parent_ids`. The same store is open to the CLI:

```bash
lllm network channels
```

## Step 4: name channels by ref

With `sssn` installed, `lllm.Resolver` resolves channel and snapshot refs as
well as tactic refs ([lesson 9](09-serving.md)). Bind the channel in
`.lllm/config.toml`:

```toml
[refs."lllm://demo/support-desk/channels/inbox"]
store = ".lllm/network"
```

```python
from lllm import Resolver

store = Resolver.from_config(".").store("lllm://demo/support-desk/channels/inbox")
print(type(store).__name__)
```

```text
LocalStore
```

Bind the ref to the `url` of a store served with `lllm network serve` and the
same code gets a `RemoteStore`.

## Step 5: serve the package as one API

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

[channels.inbox]
form = "log"

[channels.triaged]
form = "log"
```

AAAX mounts the tactic and both channels in one strategy:

```bash
lllm shell inspect .
lllm shell serve . --port 8400
```

```text
support-desk: Triage support messages.
package support-desk lllm://demo/support-desk
tactic triage lllm://demo/support-desk/tactics/triage
channel inbox lllm://demo/support-desk/channels/inbox
channel triaged lllm://demo/support-desk/channels/triaged
```

The server runs the tactic at `/tactics/triage/run`, appends events at
`/channels/<name>/events`, and serves the store at `/network/`; `GET /strategy`
lists what is mounted. A `strategy.py` next to `lllm.toml` adds a `/run`
workflow, for example one that drains the inbox through the processor.

## Summary

| You want | Use |
| --- | --- |
| Channels, events, and `lllm network` | `pip install sssn` ([sssn.one](https://sssn.one/)) |
| To run a tactic over every new event | `TacticProcessor(tactic, store, source=..., sink=...)` |
| Channel refs in `lllm.Resolver` | `sssn` plus `.lllm/config.toml` bindings |
| One API for a package, and `lllm shell` | `pip install aaax` ([aaax.one](https://aaax.one/)) |

**Next:** [Tutorials index](index.md)
