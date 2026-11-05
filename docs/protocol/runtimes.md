# Runtimes

A runtime is whatever executes behind the tactic boundary: LLLM's native agent
engine, ordinary Python, a Pydantic AI agent, or a service on another machine.
Callers cannot tell which. Every runtime is reached through `run()`, `arun()`,
`stream()`, and `info()`, served by the same [service adapter](services.md),
and listed in packages the same way.

| Runtime | You write | `runtime_kind` |
| --- | --- | --- |
| Native | an `lllm.Tactic` subclass with a `call()` program | `native` |
| Plain Python | an `lllm.Tactic` subclass with `_run()`, or `as_tactic(fn)` | `python` |
| Pydantic AI | `PydanticAITactic(agent)` | `pydantic-ai` |
| Remote | `RemoteTactic(url, name=...)` | `http` |

## Native

The native runtime is LLLM's own engine for agentic systems. Prompts declare a
turn's template, parser, tools, and failure handlers. Agents run prompts through
a short, readable call loop. Dialogs are forkable trees, and tactics are the
programs that wire agents together. The
[Architecture overview](../core/architecture.md) walks through the design.

A native tactic is an `lllm.Tactic` with a `call()` program, and there is no
adapter step: the class you write is the one that gets served, packaged, and
handed to other runtimes as a tool. Agent-backed programs (`agent_group`, agent
configs, sessions with cost) are shown in [The Tactic](tactic.md#agents) and
covered fully in the [Tactic reference](../core/tactic.md) and the
[Agent reference](../core/agent.md).

- `lllm.Tactic` is the same class as `lllm.runtimes.native.Tactic`. The rest of
  the native vocabulary (`Agent`, `Prompt`, `Dialog`, `Message`, `tool`,
  `Runtime`, and so on) imports from either `lllm` or `lllm.runtimes.native`,
  and the package loaders (`load_runtime`, `build_tactic`, `resolve_config`)
  from `lllm`.
- `NativeTactic` is a `Tactic` that insists on `agent_group` and raises
  `AssertionError` without one. Existing subclasses keep working.

### Wrapping native-style objects

`NativeTacticAdapter` is for objects that are not tactics but have the native
call shape: a `run()` or `call()` method, and optionally `arun()`/`acall()` and
`stream()`. Subclasses of `lllm.Tactic` do not need it.

```python
from pydantic import BaseModel
from lllm import CallContext, NativeTacticAdapter


class Summary(BaseModel):
    words: int
    tone: str


class WordCounter:
    """Count the words in a text."""

    name = "word-counter"
    output_model = Summary

    def call(self, task, *, tone="neutral", metadata=None):
        return Summary(words=len(task.split()), tone=tone)


tactic = NativeTacticAdapter(WordCounter(), input_type=str, run_kwargs={"tone": "brisk"})
print(repr(tactic.run("three short words", context=CallContext(trace_id="t-1"))))
print(tactic.info().name, "|", tactic.info().description)
```

```text
Summary(words=3, tone='brisk')
word-counter | Count the words in a text.
```

The adapter calls `run()` if the object has one, else `call()`, and uses
`arun()`/`acall()` for async calls when they exist (otherwise a worker thread).
It takes the types from the object's `input_type`/`input_model` and
`output_type`/`output_model` unless you pass them, adds `run_kwargs` to every
call, and passes `context=` when the method accepts it. When the method accepts
`metadata`, it also receives the context's metadata plus `lllm_request_id` and
`lllm_trace_id` (turn this off with `include_context_metadata=False`).

## Plain Python

### as_tactic

Any function becomes a tactic:

```python
from pydantic import BaseModel
from lllm import as_tactic


class Order(BaseModel):
    sku: str
    quantity: int


class Quote(BaseModel):
    sku: str
    total: float


def quote(order: Order) -> Quote:
    """Price an order at 2.50 per unit."""
    return Quote(sku=order.sku, total=order.quantity * 2.5)


tactic = as_tactic(quote)
print(tactic.info().name, "|", tactic.info().description)
print(repr(tactic.run({"sku": "A-1", "quantity": 4})))
```

```text
quote | Price an order at 2.50 per unit.
Quote(sku='A-1', total=10.0)
```

- The input type comes from the first parameter's annotation (ignoring one named
  `context`) and the output type from the return annotation. Override either
  with `input_type=` or `output_type=`.
- The name comes from the function name and the description from its
  docstring. `as_tactic` also takes `name`, `description`, `package_ref`,
  `service_ref`, `examples`, and `metadata`, and returns a `CallableTactic`.
- A function that accepts `context` (or `**kwargs`) receives the `CallContext`.
- Async functions work with both `arun()` and `run()`. `run()` drives them with
  `asyncio.run()`, so inside a running event loop use `arun()`.

```python
from lllm import CallContext, as_tactic


def label(text: str, *, context=None) -> dict:
    return {"text": text, "trace_id": context.trace_id if context else None}


async def slug(title: str) -> str:
    return title.lower().replace(" ", "-")


print(as_tactic(label).run("hello", context=CallContext(trace_id="trace-1")))
print(as_tactic(slug).run("Typed Tactics"))
```

```text
{'text': 'hello', 'trace_id': 'trace-1'}
typed-tactics
```

When the tactic needs configuration, several methods, streaming, or custom
endpoints, subclass `lllm.Tactic` and implement `_run()` instead
([The Tactic](tactic.md#two-ways-to-write-one)).

### Loading entrypoints

`lllm serve`, `lllm inspect`, and package loading turn an imported
`module:attribute` into a tactic with `coerce_tactic(value, name=None)`. It
accepts, in order:

1. a `Tactic` subclass (instantiated with no arguments) or instance;
2. a zero-argument factory that returns any of these shapes;
3. an object with a `run` method;
4. a plain callable, wrapped with `as_tactic`.

```python
from lllm.runtimes import coerce_tactic


class Reverser:
    def run(self, value: str) -> str:
        return value[::-1]


def make_quote_tactic():
    return as_tactic(quote, name="bulk-quote")


print(type(coerce_tactic(quote)).__name__)
print(coerce_tactic(make_quote_tactic).info().name)
print(coerce_tactic(Reverser(), name="reverse").run("abc"))
```

```text
CallableTactic
bulk-quote
cba
```

## Pydantic AI

Pydantic runs through LLLM in both directions:

- **Pydantic models are the I/O types of every tactic.** They are validated on
  the way in and out, exported as JSON Schema by `info()` and `/info`, and
  serialized by services. Whatever runtime is behind a tactic, its contract is
  written in Pydantic.
- **A Pydantic AI agent becomes a tactic** with `PydanticAITactic`, and is then
  served, packaged, composed, and streamed like any other.
- **A tactic becomes a Pydantic AI tool**, including a native multi-agent
  program, so a Pydantic AI agent can call LLLM systems.

Pydantic AI keeps owning what it is good at: models, tools, dependencies, model
settings, output validation, streaming, and instrumentation. Install it with
`pip install "lllm-core[pydantic-ai]"`. The examples below use Pydantic AI's
`TestModel`, which runs offline and fills outputs with placeholder values.

### Pydantic AI agents as tactics

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from lllm import PydanticAITactic


class Brief(BaseModel):
    title: str
    bullets: list[str]


# TestModel runs offline. For live calls use a model name, e.g. Agent("openai:gpt-4o-mini", ...).
agent = Agent(TestModel(), output_type=Brief, name="brief-writer")
tactic = PydanticAITactic(agent, description="Write a short brief.")

print(repr(tactic.run("typed tactic boundaries")))
info = tactic.info()
print(info.name, info.runtime_kind, info.capabilities)
print(info.output_schema["title"], info.output_schema["required"])
```

```text
Brief(title='a', bullets=['a'])
brief-writer pydantic-ai ('arun', 'events', 'run', 'stream')
Brief ['title', 'bullets']
```

The tactic takes its name from `agent.name`, its output type from
`agent.output_type`, and its published output schema from the agent. The
description comes from `description=` or the agent's own `description`; set one
of them, because without either it falls back to the docstring of Pydantic AI's
`Agent` class. `run()` calls `agent.run_sync()`, `arun()` calls `agent.run()`,
and both return `result.output`.

#### Typed input

Give the tactic an `input_type` to validate input and publish its schema:

```python
class BriefRequest(BaseModel):
    topic: str
    audience: str = "engineers"


typed = PydanticAITactic(agent, input_type=BriefRequest, description="Write a short brief.")
print(repr(typed.run({"topic": "refs"})))
print(typed.info().input_schema["required"])
```

```text
Brief(title='a', bullets=['a'])
['topic']
```

`input_mode` decides what the agent receives when the validated input is a
Pydantic model:

| `input_mode` | The agent receives |
| --- | --- |
| `auto` (default), `json` | The model as JSON text, e.g. `{"topic":"refs","audience":"engineers"}`. |
| `dict` | `model.model_dump(mode="json")`. |
| `python` | The model instance. |
| `text` | `str(value)`. |

For anything else, pass `input_mapper=` (a function from the validated input to
the agent's prompt) and, on the way back, `output_mapper=` (a function from the
run result to the output).

#### Run settings and context

`run_kwargs` are passed to every agent run, and keyword arguments given to
`run()` or `arun()` are merged on top, so Pydantic AI's own options such as
`model_settings`, `deps`, `usage_limits`, and `message_history` pass straight
through. The `CallContext` arrives as the run's `metadata`, with
`lllm_request_id` and `lllm_trace_id` added:

```python
from pydantic_ai import RunContext
from lllm import CallContext

probe = Agent(TestModel(), name="probe")
seen = {}


@probe.tool
def record(ctx: RunContext[None]) -> str:
    """Record what the run received."""
    seen.update(model_settings=ctx.model_settings, metadata=ctx.metadata)
    return "recorded"


probe_tactic = PydanticAITactic(
    probe,
    input_type=str,
    output_type=str,
    run_kwargs={"model_settings": {"temperature": 0}},
)
probe_tactic.run("hi", context=CallContext(trace_id="trace-1", metadata={"user": "ada"}))
print(seen["model_settings"])
print(seen["metadata"]["user"], seen["metadata"]["lllm_trace_id"])
```

```text
{'temperature': 0}
ada trace-1
```

An explicit `metadata=` argument wins over the context. Turn the forwarding off
with `include_context_metadata=False`.

#### The full result

`result_mode="result"` returns Pydantic AI's run result instead of
`result.output`. The output type would otherwise default to the agent's, so
declare `output_type=Any`, and keep this mode for in-process callers:

```python
from typing import Any

full = PydanticAITactic(agent, output_type=Any, result_mode="result")
result = full.run("typed tactic boundaries")
print(type(result).__name__, repr(result.output))
```

```text
AgentRunResult Brief(title='a', bullets=['a'])
```

#### Streaming

`stream()` and `astream()` use `run_stream_sync()` and `run_stream()` and yield
the agent's output as it builds up:

```python
print(list(tactic.stream("typed tactic boundaries")))
print(list(typed.stream(BriefRequest(topic="refs"))))
```

```text
[Brief(title='a', bullets=['a']), Brief(title='a', bullets=['a'])]
[Brief(title='a', bullets=['a']), Brief(title='a', bullets=['a'])]
```

- `stream_mode="text"` yields from Pydantic AI's `stream_text()`,
  `"response"` yields `ModelResponse` objects from `stream_response()`, and
  `"raw"` yields Pydantic AI's stream object itself.
- The streaming methods hand the input to the agent as given, without the
  validation `run()` applies, so pass a model instance (as above) or text.
- `aevents()` yields Pydantic AI's own event objects from
  `run_stream_events()` (`PartStartEvent` through `AgentRunResultEvent`), not
  `TacticEvent`s.

### Tactics as Pydantic AI tools

A tactic is a typed callable, so handing one to a Pydantic AI agent takes one
small function over `tactic.run()`. Pydantic AI builds the tool's schema from
the function's annotations and validates the model's arguments into `Term`:

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from lllm import Tactic


class Term(BaseModel):
    word: str


class Definition(BaseModel):
    word: str
    meaning: str


class Glossary(Tactic):
    """Look up a term in the team glossary."""

    name = "glossary"
    input_type = Term
    output_type = Definition
    entries = {"tactic": "a typed unit of work"}

    def call(self, task: Term) -> Definition:
        return Definition(word=task.word, meaning=self.entries.get(task.word, "not in the glossary"))


glossary = Glossary()


def lookup(term: Term) -> Definition:
    """Look up a term in the team glossary."""
    return glossary.run(term)


agent = Agent(TestModel(), tools=[lookup])
result = agent.run_sync("What does 'tactic' mean here?")
print(result.output)
```

```text
{"lookup":{"word":"a","meaning":"not in the glossary"}}
```

`TestModel` calls every tool once with placeholder arguments (`word="a"`) and
reports what came back; a real model chooses the arguments. Any tactic works
here, including a native program with its own agents or a `RemoteTactic`.

`tactic_as_tool(tactic)` and the method `tactic.as_tool()` build such a function
for you: named after the tactic, documented with its description, and signed
`task: InputType -> OutputType`, or with one keyword argument per model field
when `parameter_mode="kwargs"`:

```python
import inspect
from lllm import tactic_as_tool

tool = tactic_as_tool(glossary)
print(tool.__name__, inspect.signature(tool))
print(tool({"word": "tactic"}))

define = glossary.as_tool(name="define", parameter_mode="kwargs")
print(define.__name__, inspect.signature(define))
print(define(word="tactic"))
```

```text
glossary (task: __main__.Term) -> __main__.Definition
word='tactic' meaning='a typed unit of work'
define (*, word: str) -> __main__.Definition
word='tactic' meaning='a typed unit of work'
```

!!! warning "Registering these with Pydantic AI"
    Pydantic AI reads a tool's parameter types from the function's
    annotations. The functions from `tactic_as_tool()` and `as_tool()` carry
    the typed signature shown above but untyped annotations, so with Pydantic
    AI 2.66 a `"task"` tool is advertised with an untyped `task` argument, and
    a `"kwargs"` tool fails to register with a `KeyError`. When a Pydantic AI
    agent needs the tool's schema, register a typed function like `lookup`
    above.

## Remote tactics

`RemoteTactic` is a tactic backed by an LLLM service. It sends the
[request envelope](services.md#requests-and-responses) to `/run`, reads
`/stream` as server-sent events, and fetches `/info`:

```python
from lllm import RemoteTactic

remote = RemoteTactic("http://127.0.0.1:8000", name="shout")
print(remote.run({"text": "hello"}))
```

```text
{'text': 'HELLO', 'trace_id': None}
```

Give it `input_type` and `output_type` to validate locally and get models back.
Its URL, error handling, and test transports are covered in
[Services](services.md#calling-a-service-from-python). Usually you don't
construct one directly: an `lllm://` ref bound to a URL in `.lllm/config.toml`
resolves to a `RemoteTactic` ([Refs and Composition](refs.md)).

## Picking a runtime

- **Native** when you are building the agentic system itself: you want to see
  and steer every agent turn, fork dialogs, parse and repair outputs, run code
  in a sandbox, or orchestrate several agents with sessions and cost tracking.
- **Plain Python** for deterministic steps (parsing, scoring, lookups, glue)
  and for adapting code that has nothing to do with agents.
- **Pydantic AI** when an agent is already written for Pydantic AI, or you want
  its model and tool ecosystem and its instrumentation. Wrap it once and it
  joins services, packages, and channels.
- **Remote** when the tactic runs elsewhere: another team's service, a GPU
  host, or a process you scale separately.

The choice is local to one tactic. A native program can call a Pydantic AI
tactic, which can use a native tactic as a tool, and any of them can move
behind a service without its callers changing.

## Next

- [Services](services.md): serve a tactic and call it over HTTP.
- [Refs and Composition](refs.md): wire tactics together by `lllm://` ref.
- [Tutorials](../tutorials/index.md): step-by-step walkthroughs, including
  wrapping, serving, and packaging a Pydantic AI agent.
