# Streaming

A tactic can report progress while it works. A native tactic writes
`events()`, an ordinary generator that yields as each step finishes. LLLM
runs it like a call, with input validation, fresh agents, and its own
session, and serves the same stream to async code (`aevents()`) and over
HTTP (`/stream`). Streaming a model's tokens is a separate switch on an
agent.

```python title="brief_stream.py"
import asyncio

from lllm import Tactic, TacticEvent


class Brief(Tactic):
    name = "brief"
    agent_group = ["researcher", "writer"]

    def call(self, topic: str) -> str:  # (1)!
        *_, result = self.events(topic)
        return result.data

    def events(self, topic: str):  # (2)!
        researcher = self.agents["researcher"]
        researcher.open("notes")
        researcher.receive(f"List three facts about {topic}.")
        notes = researcher.respond().content
        yield TacticEvent(kind="progress", data="notes ready")  # (3)!

        writer = self.agents["writer"]
        writer.open("brief")
        writer.receive(f"Write a one-line brief from these notes:\n{notes}")
        yield TacticEvent.result(writer.respond().content)


brief = Brief(
    {
        "global": {"model_name": "gpt-4o-mini"},
        "agent_configs": [
            {"name": "researcher", "system_prompt": "You collect facts."},
            {"name": "writer", "system_prompt": "You write briefs."},
        ],
    }
)


async def main():
    async for event in brief.aevents("tide pools"):  # (4)!
        print(event.kind, event.data)
        #> progress notes ready
        #> result Tide pools shelter hardy sea life between tides.


asyncio.run(main())
```

1. `call()` reuses the steps: it runs `events()` to the end and returns the
   last event's data, so the program is written once. `brief("tide pools")`
   still returns the brief.
2. `events()` is a plain generator, run on a worker thread. Yield whenever
   there is something to report.
3. A `TacticEvent` labels what you yield. `TacticEvent.result()` makes the
   final one.
4. `aevents()` validates the input, runs `events()` on a fresh copy of the
   tactic with its own agents, and yields `TacticEvent`s.

## Events

A `TacticEvent` has a `kind`, a `data` payload, `metadata`, and an `id` and
`timestamp` filled in for you. Kinds are your own words, such as
`progress`, `token`, or `step`; two have helpers:

| You yield | Callers receive |
| --- | --- |
| `TacticEvent(kind="progress", data=...)` | that event |
| `TacticEvent.result(value)` | an event of kind `result` |
| `TacticEvent.error("message")` | kind `error`, with `data={"message": ...}` |
| any other value, such as a string or a dict | `TacticEvent(kind="message", data=value)` |

Keyword arguments to `result()` and `error()` become the event's
`metadata`. A kind is a single token, with no spaces or slashes.

## events() or stream()

There are two methods you can write, and two you call:

| Method | Who writes it | What it gives |
| --- | --- | --- |
| `events(task)` | you, on `lllm.Tactic` | labeled events |
| `stream(value, *, context=None)` | you, on any tactic | raw items |
| `aevents(value)` | LLLM | an async iterator of `TacticEvent`s |
| `astream(value)` | LLLM | an async iterator of the items `stream()` yields |

Write `events()` to report the steps of an agent program. Write `stream()`
when the output itself is a sequence, such as rows or pieces of text; it is
part of the protocol every tactic shares, so plain Python tactics and
wrapped runtimes can stream too. `aevents()` serves either one, wrapping
`stream()` items as `message` events. Overriding one of them adds `events`
or `stream` to `tactic.info().capabilities`.

For a source that is async already, you can override `astream()` or
`aevents()` instead. Those run on the tactic itself, so on a tactic with
`agent_group` LLLM warns that they would share its agents; write `events()`
or `stream()` there.

!!! warning "Consume your own tactic's stream through `aevents()`"
    Calling the `events()` or `stream()` you wrote runs it as plain Python:
    the input is not validated, no session is recorded, and the agents are
    the ones built with the tactic, so a second call finds the dialogs of
    the first (`Dialog 'notes' already exists`). In a script, run
    `aevents()` with `asyncio.run()`.

## Each stream is a call

`aevents()` runs your generator the way `tactic(task)` runs `call()`:

- **Fresh agents.** Every stream gets its own copy of the tactic, with
  agents built from the config, so concurrent streams never share dialogs.
- **A session.** The stream's `TacticCallSession` records every agent call
  and its cost. A tactic called inside `events()` is recorded in it, and
  [observers](observability.md) hear about success and failure.
- **One item at a time.** The generator runs on a worker thread and is
  advanced only when the consumer asks for the next item. Nothing runs
  ahead, and a slow step never blocks the event loop.
- **Errors reach the consumer.** An exception in `events()` is raised from
  the `async for` loop, and the session is marked `failure`.

## Stopping early

The consumer can stop a stream at any time. Closing it closes your
generator at the `yield` where it waits, so its `finally` and `with` blocks
run:

```python title="stop_early.py"
import asyncio
from contextlib import aclosing

from lllm import Tactic


def close_feed(symbol):  # stands in for closing a market-data connection
    print(f"{symbol} feed closed")
    #> ACME feed closed


class Ticker(Tactic):
    name = "ticker"

    def call(self, symbol: str) -> str:
        return symbol

    def events(self, symbol: str):
        try:
            for price in (101.2, 101.6, 100.9, 102.3):
                yield {"symbol": symbol, "price": price}
        finally:
            close_feed(symbol)  # (1)!


async def main():
    async with aclosing(Ticker().aevents("ACME")) as events:  # (2)!
        async for event in events:
            print(event.data)
            #> {'symbol': 'ACME', 'price': 101.2}
            #> {'symbol': 'ACME', 'price': 101.6}
            if event.data["price"] > 101.5:
                break


asyncio.run(main())
```

1. The cleanup runs before `aclose()` returns. Release connections or
   files here.
2. `aclosing()` calls `aclose()` when the loop ends. Without it, an
   abandoned stream is closed only when it is garbage-collected.

The session of a closed stream is marked `cancelled`, and observers receive
neither a success nor a failure. A model call in progress is not
interrupted: the generator closes when that step returns to its `yield`.
Over HTTP, a client that disconnects from `/stream` closes the stream the
same way.

## Over HTTP

A served tactic streams from `POST /stream`. The request body is the same
as for `/run`, `{"input": ..., "context": {...}}`, and the response is
`text/event-stream`: one `data:` line per event, holding the event as
JSON. See [Serving](serving.md) to run the server. Here the app runs
in-process:

```python title="stream_endpoint.py"
import json

from fastapi.testclient import TestClient

from lllm import Tactic, TacticEvent, create_tactic_app


class Countdown(Tactic):
    name = "countdown"
    input_type = int

    def call(self, n: int) -> str:
        return "liftoff"

    def events(self, n: int):
        yield from range(n, 0, -1)
        yield TacticEvent.result("liftoff")


client = TestClient(create_tactic_app(Countdown()))
with client.stream("POST", "/stream", json={"input": 3}) as response:
    print(response.headers["content-type"])
    #> text/event-stream; charset=utf-8
    for line in response.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line.removeprefix("data: "))
            print(event["kind"], event["data"])
            #> message 3
            #> message 2
            #> message 1
            #> result liftoff
```

Each event carries `id`, `timestamp`, `kind`, `data`, and `metadata`. If
the tactic raises, the stream ends with an `error` event whose metadata
names the `error_type`, the `tactic`, and the `request_id`. Invalid input
also arrives as an `error` event, because the input is validated when the
stream starts. A tactic with neither `events()` nor `stream()` answers
`/stream` with status 400.

From Python, `RemoteTactic(url).aevents(value)` yields the same
`TacticEvent`s, and `RemoteTactic(url).stream(value)` yields their `data`
in sync code.

## Streaming tokens from an agent

Token streaming happens one level down, in the agent. Give an agent a
stream handler, and every `respond()` asks the provider for a streamed
reply and passes each piece of text to it as it arrives:

```python title="tokens.py"
from lllm import Tactic
from lllm.runtimes.native.invokers.base import BaseStreamHandler


class Collect(BaseStreamHandler):
    def __init__(self):
        self.pieces = []

    def handle_chunk(self, chunk_content, chunk_response):  # (1)!
        self.pieces.append(chunk_content)


agent = Tactic.quick(
    system_prompt="You tell short stories.", model="gpt-4o-mini"
)
agent.stream_handler = Collect()
agent.open("story")
agent.receive("Tell a two-sentence story about a lighthouse.")
reply = agent.respond()  # (2)!
print(reply.content)
#> The keeper lit the lamp one last time. At dawn, a ship waved back.
print("".join(agent.stream_handler.pieces) == reply.content)
#> True
```

1. `chunk_content` is the new text; `chunk_response` is the provider's raw
   chunk (a Chat Completions chunk, or a Responses API event).
2. The reply is still assembled whole, then parsed, repaired, and recorded
   exactly as without streaming. Token usage is recorded too, when the
   provider reports it in the stream.

Inside a tactic, set the handler on an agent in `call()` or `events()`. To
send tokens to `/stream`, the handler has to hand them to `events()`, and
`respond()` blocks until the reply is complete, so run it on a thread and
yield the pieces from a queue as they arrive:

```python title="token_events.py"
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor

from lllm import Tactic, TacticEvent
from lllm.runtimes.native.invokers.base import BaseStreamHandler


class ToQueue(BaseStreamHandler):
    def __init__(self, pieces):
        self.pieces = pieces

    def handle_chunk(self, chunk_content, chunk_response):
        self.pieces.put(chunk_content)


class Story(Tactic):
    name = "story"
    agent_group = ["writer"]

    def events(self, prompt: str):
        writer, pieces = self.agents["writer"], queue.Queue()
        writer.stream_handler = ToQueue(pieces)
        writer.open("story")
        writer.receive(prompt)
        with ThreadPoolExecutor(1) as pool:
            reply = pool.submit(writer.respond)  # (1)!
            reply.add_done_callback(lambda _: pieces.put(None))
            while (piece := pieces.get()) is not None:
                yield TacticEvent(kind="token", data=piece)
        yield TacticEvent.result(reply.result().content)  # (2)!


story = Story(
    {
        "global": {"model_name": "gpt-4o-mini"},
        "agent_configs": [
            {"name": "writer", "system_prompt": "You write one-liners."}
        ],
    }
)


async def main():
    async for event in story.aevents("Write one line about the sea."):
        if event.kind == "result":  # after one "token" event per piece
            print(event.data)
            #> The sea keeps every secret it is told.


asyncio.run(main())
```

1. The model call runs on its own thread, so the generator can yield each
   piece while the reply is still arriving. When the call ends, the
   callback ends the loop.
2. `result()` returns the reply, or raises the error the call raised.

The agent call is recorded in the stream's session like any other. If the
consumer stops early, the stream closes once the model call in progress
returns.
