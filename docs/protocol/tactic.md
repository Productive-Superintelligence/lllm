# The Tactic

A tactic is a typed unit of work. It declares what it accepts and what it
returns, validates both on every call, and describes itself in a form that
services, packages, and other agents can read. That one boundary is what lets
the same piece of work be:

- **validated**: input and output are checked against Pydantic types;
- **streamed**: chunks and events have one shape for every runtime;
- **served**: any tactic becomes a FastAPI app with `/run`, `/stream`, `/info`;
- **packaged**: an `lllm.toml` entry points at it by name and ref;
- **composed**: other tactics, Pydantic AI agents, and channel processors call
  it without knowing how it is implemented.

There is one class to learn:

```python
from lllm import Tactic
```

`lllm.Tactic` is the native program class, with agents, per-call sessions, and
batch calls. It subclasses `lllm.protocol.Tactic`, the runtime-neutral base that
supplies the typed boundary: `run()`, `arun()`, `stream()`, `aevents()`, and
`info()`. Adapters for other runtimes (plain functions, Pydantic AI, remote
services) are protocol tactics too, so everything on this page applies to them.

## Two ways to write one

Write the program in `call()`, or implement the protocol hook `_run()`. Both
produce a tactic that validates, serves, and packages the same way.

=== "Program: call()"

    ```python
    from pydantic import BaseModel
    from lllm import Tactic


    class Question(BaseModel):
        text: str


    class Answer(BaseModel):
        text: str


    class Shout(Tactic):
        """Shout the question back."""

        name = "shout"
        input_type = Question
        output_type = Answer

        def call(self, task: Question) -> Answer:
            return Answer(text=task.text.upper())


    shout = Shout()
    print(shout.run({"text": "hello"}))   # text='HELLO'
    print(shout({"text": "hello"}))       # text='HELLO'
    print(shout.info().runtime_kind)      # native
    ```

=== "Hook: _run()"

    ```python
    from pydantic import BaseModel
    from lllm import Tactic


    class Question(BaseModel):
        text: str


    class Answer(BaseModel):
        text: str


    class ShoutHook(Tactic[Question, Answer]):
        """Shout the question back."""

        name = "shout"
        input_type = Question
        output_type = Answer

        def _run(self, input_value: Question, *, context=None) -> Answer:
            return Answer(text=input_value.text.upper())


    shout = ShoutHook()
    print(shout.run({"text": "hello"}))   # text='HELLO'
    print(shout.info().runtime_kind)      # python
    ```

The input arrives as a dict here and reaches your code as a `Question`. The
return value is checked against `Answer` before the caller sees it.

| | `call()` program | `_run()` hook |
| --- | --- | --- |
| `run()`, `arun()`, `info()`, serving, packaging, use as a tool | yes | yes |
| `tactic(task)`, `acall`, `bcall`, `ccall` | yes | yes |
| Runs inside a `TacticCallSession` | every call | calls through `tactic(task)`, `acall`, `bcall`, `ccall` |
| Runs on a per-call copy with fresh `agent_group` agents | every call | never: hooks run on the instance |
| `info().runtime_kind` | `"native"` | `"python"` (`"native"` with an `agent_group`) |
| Receives | the validated task, and `context=` when `call` accepts it | the validated input and a `CallContext` |

`tactic(task)` and the other native entry points run `call()` when the class
writes one, else its `_run()` hook (or `_arun()`). A hook-only tactic has no
`call()`: `tactic.call(task)` raises `TacticUnsupportedError` and points you to
the entry points, and a hook that hands its call back with `super()._run()` or
`self.call()` raises the same error instead of recursing. A tactic that writes
only `_arun()` still gets `run()`, which runs `arun()` to completion, and an
`async def call()` is awaited on every entry point.

## Typed input and output

`input_type` and `output_type` accept anything Pydantic can validate:
`BaseModel` subclasses, builtins such as `str` or `int`, `list[str]`,
`Literal[...]`, dataclasses, and `TypedDict`s. `input_model` and `output_model`
are aliases, so tactics written with the native names publish their schemas
too. A class's own `input_model` wins over an `input_type` it inherits.

```python
from typing import Literal

from pydantic import BaseModel, Field
from lllm import SchemaError, Tactic


class Review(BaseModel):
    text: str = Field(min_length=1)
    stars: int = Field(ge=1, le=5)


class Verdict(BaseModel):
    label: Literal["positive", "negative"]
    confidence: float


class Classify(Tactic):
    """Label a product review."""

    name = "classify"
    input_type = Review
    output_type = Verdict

    def call(self, task: Review) -> Verdict:
        positive = task.stars >= 4
        # A plain dict is fine: the output is validated into Verdict.
        return {"label": "positive" if positive else "negative", "confidence": 0.9}


classify = Classify()
print(repr(classify.run({"text": "Works great", "stars": 5})))
print(repr(classify.run(Review(text="Broke in a day", stars=1))))

try:
    classify.run({"text": "", "stars": 9})
except SchemaError as exc:
    print(str(exc).splitlines()[0])
```

```text
Verdict(label='positive', confidence=0.9)
Verdict(label='negative', confidence=0.9)
Value does not match schema Review: 2 validation errors for Review
```

A few rules worth knowing:

- Input can be a model instance or JSON-compatible data. Output can be the
  model or data that validates into it.
- A failed check on either side raises `SchemaError`.
- The generic parameters in `Tactic[Review, Verdict]` are for type checkers
  only. Validation and the published JSON schemas come from `input_type` and
  `output_type`; a tactic without them passes values through unchecked.
- Pydantic's normal coercion applies, so `input_type = int` accepts `"21"`.

## Calling a tactic

Every tactic has the protocol methods:

| Method | Returns |
| --- | --- |
| `run(value, *, context=None, return_trace=False, **kwargs)` | The validated output, or a `CallResult` when `return_trace=True`. |
| `await arun(value, *, context=None, return_trace=False, **kwargs)` | The same, async. |
| `stream(value, *, context=None, **kwargs)` | An iterator of chunks, for tactics that stream. |
| `astream(...)` | An async iterator of chunks. |
| `aevents(...)` | An async iterator of `TacticEvent`s. |
| `info()` | The `TacticInfo` contract. |
| `supports(name)`, `capabilities()` | Whether a capability such as `"stream"` is available. |

Extra keyword arguments to `run()` and `arun()` are passed through to `call()`
or `_run()`.

`lllm.Tactic` adds the native entry points, for both styles:

| Method | Returns |
| --- | --- |
| `tactic(task, session_name=None, tags=None, metadata=None, return_session=False, **kwargs)` | The result, or the call's `TacticCallSession`. |
| `await tactic.acall(task, ...)` | The same, awaited: an `async def call()` runs on the caller's loop, a sync one in a worker thread. |
| `tactic.bcall(tasks, max_workers=None, fail_fast=True, return_sessions=False, ...)` | Results in input order. With `fail_fast=False`, failures come back as exception objects in place. |
| `async for index, result in tactic.ccall(tasks)` | `(index, result)` pairs as each call finishes. |

`tactic(task)` validates input and output the same way `run()` does.

### Traces

`return_trace=True` returns a `CallResult` with the output and a `CallTrace`
(request id, state, latency, type names, and the error if there was one).
Failures are recorded instead of raised:

```python
from lllm import CallContext

result = shout.run(
    {"text": "hi"},
    context=CallContext(request_id="req-1", metadata={"user": "ada"}),
    return_trace=True,
)
print(result.output)
print(result.trace.state, result.trace.request_id, result.trace.metadata)

failed = shout.run({"text": 5}, return_trace=True)
print(failed.output, failed.trace.state, failed.trace.error_type)
```

```text
text='HI'
success req-1 {'user': 'ada'}
None failure SchemaError
```

## Streaming and events

Implement `stream()` to make a tactic stream. Yield whatever chunks make sense,
and yield a `TacticEvent` when you want to label one:

```python
from lllm import Tactic, TacticEvent


class Countdown(Tactic):
    """Count down, then lift off."""

    name = "countdown"
    input_type = int
    output_type = str

    def call(self, task: int) -> str:
        return "liftoff"

    def stream(self, task: int, *, context=None):
        for n in range(task, 0, -1):
            yield n
        yield TacticEvent.result("liftoff")


countdown = Countdown()
print(countdown.info().capabilities)
print(list(countdown.stream(3))[:3])
```

```text
('acall', 'arun', 'call', 'run', 'stream')
[3, 2, 1]
```

`astream()` validates the input the same way `run()` does and then iterates
`stream()` on a worker thread, one item per request, so `stream()` receives
the typed value, a slow stream never blocks the event loop, and nothing runs
ahead of the reader. Closing the async iterator closes the generator on that
thread: when it is waiting at a `yield`, its cleanup has run by the time
`aclose()` returns; otherwise it runs once the item in progress is done. `aevents()` wraps every
chunk from `astream()` that is not already an event as
`TacticEvent(kind="message", data=chunk)`. `aevents()` is what the service's
`/stream` route sends:

```python
import asyncio


async def show():
    async for event in countdown.aevents(2):
        print(event.kind, event.data)


asyncio.run(show())
```

```text
message 2
message 1
result liftoff
```

A `TacticEvent` has `id`, `timestamp`, `kind`, `data`, and `metadata`.
`TacticEvent.result(value)` and `TacticEvent.error(message)` build the two
common kinds. Override `astream()` when the source is natively async, or
`aevents()` to control the event stream yourself; the tactic then advertises
`"stream"` or `"events"` in its capabilities. On `lllm.Tactic`, a synchronous
`events()` generator works too: `aevents()` and `/stream` serve it, passing it
the validated input and wrapping plain items as `message` events. There, each
stream runs like `call()`, on a per-call copy with fresh agents and its own
session.

Calling `stream()` directly skips that validation and passes the argument
through as given, so pass it a typed value (or call `self.validate_input(value)`
yourself). A tactic that does not stream raises `TacticUnsupportedError` from
`stream()`.

## CallContext

`CallContext` is the request metadata that travels with one call: ids for
correlating logs and traces, who called, and free-form tags and metadata.

```python
from lllm import CallContext

context = CallContext(
    trace_id="trace-42",
    caller="nightly-job",
    tags={"team": "research"},
    metadata={"user": "ada"},
)
```

| Field | Meaning |
| --- | --- |
| `request_id` | Unique id for this call. Generated (a UUID) when omitted. |
| `caller` | Who is calling: a service, worker, or user label. |
| `trace_id`, `span_id` | Distributed-tracing ids. |
| `tags` | String-to-string labels. |
| `metadata` | Any JSON-compatible values. |
| `package_ref`, `service_ref`, `tactic_ref`, `endpoint` | Where the call is going; services fill in `tactic_ref` and `endpoint`. |

Id fields must be plain tokens (no whitespace or path separators); the model
rejects anything else.

How the context reaches your code:

- `_run(input_value, *, context)` always gets a `CallContext`, through every
  entry point; one is created when the caller passes none.
- `call()` gets `context=` when its signature declares `context` (or
  `**kwargs`). Under `run()` it is a `CallContext`; under `tactic(task)` it is
  `None` unless the caller passes `context=`.
- Pass a `CallContext`. Anything else, such as a dict, raises `TypeError` from
  `run()` and `arun()`, and from `tactic(task)`, `acall`, `bcall`, and `ccall`
  when the program is a `_run()` or `_arun()` hook. Those four pass a `call()`
  program the value unchecked: a dict arrives as a dict.

```python
class WhoAsked(Tactic):
    name = "who-asked"
    output_type = dict

    def call(self, task: str, *, context=None) -> dict:
        return {"task": task, "caller": context.caller if context else None}


print(WhoAsked().run("summarize", context=context))
print(WhoAsked()("summarize"))
```

```text
{'task': 'summarize', 'caller': 'nightly-job'}
{'task': 'summarize', 'caller': None}
```

Services build the context from the request's `context` object, and
`RemoteTactic` sends it, so a `trace_id` set by a caller arrives in the
`call()` of a tactic running in another process. Keep configuration (models,
keys, runtime settings) on the tactic or its runtime; the context is per
request.

## Describing a tactic: info()

`info()` returns the tactic's public contract as a `TacticInfo`. Services return
it from `/info`, `lllm inspect` prints it, and the hub renders it into package
cards.

```python
shout = Shout(
    package_ref="lllm://demo/shout/tactics/shout",
    examples=[{"input": {"text": "hi"}, "output": {"text": "HI"}}],
    metadata={"owner": "docs"},
)
print(shout.info().model_dump_json(indent=2))
```

```json
{
  "name": "shout",
  "description": "Shout the question back.",
  "input_schema": {
    "properties": {
      "text": {
        "title": "Text",
        "type": "string"
      }
    },
    "required": [
      "text"
    ],
    "title": "Question",
    "type": "object"
  },
  "output_schema": {
    "properties": {
      "text": {
        "title": "Text",
        "type": "string"
      }
    },
    "required": [
      "text"
    ],
    "title": "Answer",
    "type": "object"
  },
  "capabilities": [
    "acall",
    "arun",
    "call",
    "run"
  ],
  "runtime_kind": "native",
  "package_ref": "lllm://demo/shout/tactics/shout",
  "service_ref": null,
  "examples": [
    {
      "input": {
        "text": "hi"
      },
      "output": {
        "text": "HI"
      }
    }
  ],
  "metadata": {
    "owner": "docs"
  }
}
```

| Field | Source |
| --- | --- |
| `name` | `name=` passed to the constructor, else the class attribute `name`, else the class name. An `Enum` member gives its value. It must be one path segment: no `/`, `\`, `:`, `;`, or `%`. |
| `description` | `description=`, else the class attribute, else your class's docstring. |
| `input_schema`, `output_schema` | JSON Schema exported from `input_type` and `output_type`. |
| `capabilities` | `run`, `arun`, plus `call`/`acall` on `lllm.Tactic`, and `stream`/`events` when implemented. |
| `runtime_kind` | `native`, `python`, `pydantic-ai`, `http`, `proxy`, or `sandbox` for the built-in tactics. A class that sets `runtime_kind` passes it on to its subclasses. |
| `package_ref`, `service_ref` | An `lllm://.../tactics/...` ref and a service ref or URL; see [Refs](refs.md). |
| `examples`, `metadata` | Whatever you pass to the constructor. |

The constructor keywords `name`, `description`, `package_ref`, `service_ref`,
`examples`, and `metadata` work on your own tactics. The built-in wrappers take
fewer: `RemoteTactic` takes `name` and `metadata` and fills in `service_ref`
and `metadata["url"]` from its URL, `ProxyTactic` takes `name`, `description`,
and `metadata`, and `SandboxedTactic` takes `name` and `description`. `/info` and `lllm inspect` drop secret-shaped
metadata keys such as `api_key`, `token`, `password`, and `cookie` while
keeping references such as `api_key_ref`. Even so, keep secrets out of
metadata.

## Errors

| Exception | Raised when |
| --- | --- |
| `SchemaError` | Input or output fails validation, or a type cannot be exported as JSON Schema. |
| `TacticUnsupportedError` | The tactic lacks the requested capability: no `call()` or `_run()`, `call()` on a tactic that implements `_run()` instead, or `stream()` on a tactic that does not stream. |
| `TacticRefError` | An `lllm://` tactic ref is malformed or not bound. |
| `RemoteTacticError` | A remote service answered with an error; see [Services](services.md#calling-a-service-from-python). |
| `TypeError` | `context` is not a `CallContext` where one is required (see [CallContext](#callcontext)), or `return_trace` is not a bool. |

```text
LLLMError (RuntimeError)
└── ProtocolError
    ├── SchemaError
    └── TacticError
        ├── TacticUnsupportedError
        ├── TacticLoadError
        ├── TacticRefError
        └── TacticServiceError
            └── RemoteTacticError
```

All of these import from `lllm`. Exceptions raised by your own code propagate
unchanged. When a tactic running in a session fails, the
`lllm.runtimes.native.core.tactic` logger writes one INFO line with the error
(the traceback goes to DEBUG), the session's `state` becomes `"failure"`, and
its `error_traceback` keeps the traceback. Over HTTP,
`SchemaError` and `TacticUnsupportedError` become 400 responses and other
exceptions become 500s, both with a structured body
([Services](services.md#errors)).

## Sessions for call() programs

Each call of a `call()` program runs on a fresh copy of the tactic inside a
`TacticCallSession`. Ask for it with `return_session=True`:

```python
class Upper(Tactic):
    name = "upper"

    def call(self, task: str) -> str:
        return task.upper()


class Twice(Tactic):
    name = "twice"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.upper = Upper()  # a tactic attribute becomes a sub-tactic

    def call(self, task: str) -> str:
        return self.upper(task) + " " + self.upper(task)


twice = Twice()
session = twice("hi", return_session=True)
print(session.state, repr(session.delivery))
print(session.sub_tactic_call_count, list(session.sub_tactic_sessions))
print(twice.bcall(["a", "b"]))
```

```text
success 'HI HI'
2 ['upper']
['A A', 'B B']
```

A session records `state`, `delivery` (the result), `error` and
`error_traceback` on failure, every agent call in `agent_sessions`, and every
sub-tactic session in `sub_tactic_sessions`. `total_cost`, `agent_call_count`,
and `sub_tactic_call_count` roll everything up, and `summary()` returns a small
dict for logs. When a tactic that runs in a session is called while another
program is running, its session is recorded in the caller's under the tactic's
name (the constructor's `name=` when given). That includes calls made from
`bcall` worker threads and through a [Resolver](refs.md#composition-patterns).
`call()` programs always run in a session. A `_run()` hook gets one only
through `tactic(task)`, `acall`, `bcall`, and `ccall`, not under `run()` or
`arun()`. Tactics that are not `lllm.Tactic` subclasses (wrapped functions,
Pydantic AI and remote tactics) do not create sessions.

### Agents

Set `agent_group` and pass agent configs, and every call gets fresh native
agents in `self.agents`, each tracked by the call's session. Agents reach models
through LiteLLM by default (`pip install "lllm-core[native]"`). This example
calls a model, so it needs a provider key (for example `OPENAI_API_KEY`):

```python
from lllm import Tactic


class Brief(Tactic):
    name = "brief"
    agent_group = ["writer", "critic"]

    def call(self, task: str) -> str:
        writer, critic = self.agents["writer"], self.agents["critic"]
        writer.open("draft")
        writer.receive(task)
        draft = writer.respond().content
        critic.open("review")
        critic.receive(draft)
        return critic.respond().content


config = {
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [
        {"name": "writer", "system_prompt": "You write a three-bullet brief."},
        {"name": "critic", "system_prompt": "You tighten the brief. Reply with the final text."},
    ],
}

brief = Brief(config)
session = brief("Why typed tactic boundaries matter", return_session=True)
print(session.delivery)
print(session.agent_call_count, session.total_cost)
```

For a single agent with no configuration at all, `Tactic.quick()` returns the
reply (or, without a query, the agent itself):

```python
from lllm import Tactic

reply = Tactic.quick("In one sentence, what is a tactic?", model="gpt-4o-mini")
print(reply.content)
```

Agents, prompts, configs, and the call loop are covered in the
[Tactic reference](../core/tactic.md) and the [Agent reference](../core/agent.md).

## Which style to use

- **`call()`** for agentic programs: the tactic orchestrates native agents or
  other tactics, and you want a session with cost roll-up on every call.
- **`_run()`** for deterministic steps, adapters, and glue code that only needs
  the typed boundary. For a single function, `as_tactic(fn)` is shorter still
  ([Runtimes](runtimes.md#plain-python)).
- **Wrap, don't subclass**, when the work already lives elsewhere:
  `PydanticAITactic` for Pydantic AI agents and `RemoteTactic` for running
  services ([Runtimes](runtimes.md)).

Callers cannot tell the difference: they use `run()`, `arun()`, and `stream()`,
and services and packages read the same `info()`.

## Design lineage

The tactic boundary grew out of the reusable reasoning units in Analytica,
which decomposes complex societal, economic, political, and scientific
questions into grounded subpropositions, runs specialized agents or tools over
them, and synthesizes the results into a more stable analysis. LLLM keeps the
boundary and drops the domain: a tactic is not tied to propositional reasoning,
forecasting, or any one agent architecture.

Apeiron applied the same boundary to autonomous software development:
full-lifecycle application synthesis with demand modeling, computer-use agent
evaluation, activity tracing, and locality-controlled iteration. Analysis
agents, software-building agents, evaluators, tools, and deployment services
all fit behind one small, typed callable.

- Cheng, Junyan, Kyle Richardson, and Peter Chin. "Analytica: Soft Propositional
  Reasoning for Robust and Scalable LLM-Driven Analysis." *The Fourteenth
  International Conference on Learning Representations (ICLR)*, 2026.
- Cheng, Junyan, Ankit Srivastava, Jessie Zeng, Milenko Drinic, and Jack W.
  Stokes. "Apeiron: A Scalable LLM-agentic Framework for Autonomous
  Full-lifecycle Demand-optimized Application Synthesis." *Findings of the
  Association for Computational Linguistics: ACL 2026*, 2026, pp. 3868-3899.

## Next

- [Runtimes](runtimes.md): native, plain Python, Pydantic AI, and remote
  tactics.
- [Services](services.md): serve a tactic over HTTP and call it back.
- [Refs and Composition](refs.md): name tactics with `lllm://` refs and wire
  them together.
