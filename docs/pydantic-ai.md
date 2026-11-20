# Pydantic AI

[Pydantic AI](https://pydantic.dev/docs/ai/) agents run inside LLLM as
tactics. `PydanticAITactic` puts a tactic's typed boundary around an agent,
so you can call it, serve it, package it, and hand it to other agents like
any LLLM program. The adapter forwards every call to the agent and adds only
the boundary, so the agent keeps working exactly as it does in Pydantic AI.

```bash
pip install "lllm-core[pydantic-ai]"
```

Here is a Pydantic AI agent that writes changelog entries, wrapped as a
tactic:

```python title="changelog.py"
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from lllm import PydanticAITactic


class Entry(BaseModel):
    category: Literal["added", "changed", "fixed"]
    summary: str


model = TestModel(  # (1)!
    custom_output_args={
        "category": "fixed",
        "summary": "Uploads retry after a timeout.",
    }
)
agent = Agent(
    model,
    output_type=Entry,
    name="changelog",
    description="Write a changelog entry for a pull request.",
)

changelog = PydanticAITactic(agent)  # (2)!

entry = changelog.run("Retry uploads when the connection times out")  # (3)!
print(repr(entry))
#> Entry(category='fixed', summary='Uploads retry after a timeout.')

info = changelog.info()  # (4)!
print(info.name, info.runtime_kind, info.capabilities)
#> changelog pydantic-ai ('arun', 'events', 'run', 'stream')
print(info.output_schema["required"])
#> ['category', 'summary']
```

1. `TestModel` answers offline, so the examples on this page run anywhere.
   In your code, pass a model name such as `"openai:gpt-4o-mini"`; nothing
   else changes.
2. The tactic takes its name, description, and output type from the agent.
   Pass `name=`, `description=`, or `output_type=` to override them.
3. `run()` calls `agent.run_sync()` and returns `result.output`, checked
   against `Entry`. `arun()` calls `agent.run()`.
4. Every tactic publishes this contract. Services return it from `/info`,
   and the [hub](publishing.md) shows it on package cards.

Pydantic AI still owns the model, tools, dependencies, model settings,
output validation, retries, streaming, and instrumentation. LLLM adds what
every tactic has: validated input and output, a published contract,
[serving](serving.md), [packaging](packages.md), and use from native agents.
The adapter never reinterprets what the agent does, so any agent you can run
with Pydantic AI can be wrapped.

## Typed input

Give the tactic an `input_type` to validate input and publish its schema,
and an `input_mapper` to turn the validated value into the agent's prompt:

```python title="typed_input.py"
from pydantic import BaseModel
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.test import TestModel

from lllm import PydanticAITactic


class PullRequest(BaseModel):
    title: str
    labels: list[str] = []


agent = Agent(
    TestModel(custom_output_text="Fixed: uploads retry after a timeout."),
    name="changelog",
)
changelog = PydanticAITactic(
    agent,
    input_type=PullRequest,
    input_mapper=lambda pr: f"{pr.title}\nLabels: {', '.join(pr.labels)}",
)

pull = {"title": "Retry uploads after a timeout", "labels": ["bug"]}
with capture_run_messages() as messages:  # (1)!
    print(changelog.run(pull))
    #> Fixed: uploads retry after a timeout.
print(messages[0].parts[-1].content)
"""
Retry uploads after a timeout
Labels: bug
"""
```

1. Pydantic AI's own tools keep working through the tactic.
   `capture_run_messages()` shows the prompt the agent received.

The input schema is published in `info()`, and input that does not match
raises `SchemaError` before the agent runs. Without an `input_mapper`,
`input_mode` decides what the agent receives for a Pydantic model:

| `input_mode` | The agent receives |
| --- | --- |
| `"auto"` (default), `"json"` | The model as JSON text: `{"title":"...","labels":["bug"]}` |
| `"text"` | `str(model)`: `title='...' labels=['bug']` |
| `"dict"`, `"python"` | A dict, or the model itself, for agent objects that accept them. Pydantic AI's `Agent` takes text and raises `ValueError`. |

An `output_mapper` works the same way on the way out: it receives the run
result and returns the tactic's output.

## Run settings, dependencies, and context

`run_kwargs` are passed to every agent run, and keyword arguments to `run()`
or `arun()` are merged over them. LLLM hands them to the agent unchanged, so
Pydantic AI's own options work: `deps`, `model_settings`, `usage_limits`,
`message_history`, and the rest. The call's `CallContext` arrives as the
run's `metadata`:

```python title="deps.py"
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from lllm import CallContext, PydanticAITactic


@dataclass
class Repo:
    pulls: dict[int, str]


def model(messages, info):  # (1)!
    if len(messages) == 1:
        call = ToolCallPart("get_pull", {"number": 812})
        return ModelResponse(parts=[call])
    title = messages[-1].parts[0].content
    return ModelResponse(parts=[TextPart(f"Fixed: {title}.")])


agent = Agent(FunctionModel(model), deps_type=Repo, name="changelog")


@agent.tool
def get_pull(ctx: RunContext[Repo], number: int) -> str:
    """Fetch the title of a pull request."""
    print(ctx.model_settings, ctx.metadata["lllm_trace_id"])  # (2)!
    #> {'temperature': 0} release-4-2
    return ctx.deps.pulls[number]


changelog = PydanticAITactic(
    agent,
    run_kwargs={
        "deps": Repo({812: "uploads retry after a timeout"}),
        "model_settings": {"temperature": 0},
    },
)
context = CallContext(trace_id="release-4-2")
print(changelog.run("Write the entry for #812.", context=context))
#> Fixed: uploads retry after a timeout.
```

1. A `FunctionModel` plays the model: it asks for pull request 812, then
   answers with what the tool returned.
2. Inside the run, `RunContext` holds the model settings from `run_kwargs`
   and the call's context as metadata.

The metadata is the context's own `metadata` plus `lllm_request_id` and
whichever other ids are set (`lllm_trace_id`, `lllm_span_id`, `lllm_tags`,
and so on). An explicit `metadata=` argument wins, and
`include_context_metadata=False` turns the forwarding off.

`result_mode="result"` returns Pydantic AI's whole run result, with its
messages and usage, instead of `result.output`. The output type is then
`Any` unless you pass one, while `info()` still publishes the agent's output
schema, so keep this mode for in-process callers.

## Streaming

`stream()` and `astream()` use the agent's `run_stream_sync()` and
`run_stream()` and yield its output as it builds up. `stream_mode` picks the
view:

| `stream_mode` | Yields |
| --- | --- |
| `"output"` (default) | The output so far, from `stream_output()` |
| `"text"` | The text so far, from `stream_text()` |
| `"response"` | `ModelResponse` objects, from `stream_response()` |
| `"raw"` | Pydantic AI's stream object itself |

`aevents()` reads `run_stream_events()` and wraps each Pydantic AI event in a
`TacticEvent`: its `kind` is the event's `event_kind`, and its `data` is the
event as JSON. This is what a served tactic's `/stream` route sends:

```python title="events.py"
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from lllm import PydanticAITactic

agent = Agent(
    TestModel(custom_output_text="Uploads retry after timeouts."),
    name="changelog",
)
changelog = PydanticAITactic(agent)


async def main():
    events = [event async for event in changelog.aevents("Retry uploads")]
    deltas = [event for event in events if event.kind == "part_delta"]
    print([event.data["delta"]["content_delta"] for event in deltas])
    #> ['Uploads ', 'retry ', 'after ', 'timeouts.']
    print(events[-1].kind, events[-1].data["result"])
    #> agent_run_result {'output': 'Uploads retry after timeouts.'}


asyncio.run(main())
```

The streaming methods validate the input the way `run()` does, so bad input
raises `SchemaError` before the agent starts.

## Calling Pydantic AI agents from LLLM

A `PydanticAITactic` is a tactic, so a native tactic can call it in its
`call()` program, and a native agent can call it as a tool. Turn it into a
tool with `tactic_as_function()`, as for any tactic
([Tactics as tools](tools.md#tactics-as-tools)):

```python test="skip" reason="fragment; changelog is the tactic from the first example"
from lllm import Prompt
from lllm.runtimes.native import tactic_as_function

NOTES = Prompt(
    path="release/notes",
    prompt="Write the release notes for these pull requests:\n{pulls}",
    function_list=[tactic_as_function(changelog)],
)
```

The tool's parameters come from the tactic's input schema, and when the
native agent calls it, the Pydantic AI agent runs. Its calls are not added
to the native tactic's session, which tracks native agents and `lllm.Tactic`
sub-tactics; trace them with Pydantic AI's own instrumentation.

## Giving LLLM tactics to Pydantic AI agents

Going the other way, `tactic_as_tool(tactic)` turns any tactic into a plain,
typed function, and `lllm.Tactic` has the same as a method, `as_tool()`.
Pydantic AI builds a tool's schema from a function's annotations, so the
function registers as it is. Here a Pydantic AI agent asks a native tactic,
which runs its own agent, to rate a change:

```python title="tactic_tools.py"
import inspect

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from lllm import Prompt, Tactic


class Risk(BaseModel):
    level: str
    reason: str


RATE = Prompt(
    path="release/risk",
    prompt="Rate the risk of shipping this change: {change}",
    format=Risk,
)


class ReleaseRisk(Tactic):
    """Rate the risk of shipping a change."""

    name = "release_risk"
    agent_group = ["reviewer"]
    input_type = str
    output_type = Risk

    def call(self, task: str) -> Risk:
        reviewer = self.agents["reviewer"]
        reviewer.open("risk")
        reviewer.receive_prompt(RATE, {"change": task})
        return reviewer.respond().parsed


config = {
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [{"name": "reviewer", "system_prompt": "You review."}],
}
tool = ReleaseRisk(config).as_tool()  # (1)!
print(tool.__name__, inspect.signature(tool))
#> release_risk (task: str) -> __main__.Risk


def model(messages, info):  # (2)!
    if len(messages) == 1:
        args = {"task": "Switch the default database driver"}
        return ModelResponse(parts=[ToolCallPart("release_risk", args)])
    risk = messages[-1].parts[0].content
    answer = f"Hold the release: the risk is {risk.level}. {risk.reason}"
    return ModelResponse(parts=[TextPart(answer)])


planner = Agent(FunctionModel(model), tools=[tool])
print(planner.run_sync("Can we ship the new driver today?").output)
#> Hold the release: the risk is high. Every query goes through the driver.
```

1. The function is named after the tactic, documented with its description,
   and annotated with its input and output types.
2. A `FunctionModel` plays the planner's model: it calls the tool, then
   answers from the `Risk` it got back.

`parameter_mode="kwargs"` gives one keyword argument per field of a model
input, as in `(*, title: str, labels: list[str] = []) -> Entry`, and `name=`
and `description=` override the tool's name and description. Any tactic
works: a native program, a function wrapped with `as_tactic()`, or a
`RemoteTactic` that calls a [service](serving.md).

The function is synchronous and calls `tactic.run()`. Pydantic AI does not
allow `Agent.run_sync()` inside a synchronous tool, so to give one Pydantic
AI agent another one wrapped as a tactic, pass `asynchronous=True`. The
tool is then an `async def` function that awaits `arun()`, and
`as_tool(asynchronous=True)` makes the same:

```python title="agent_tools.py"
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from lllm import PydanticAITactic, tactic_as_tool

writer = Agent(
    TestModel(custom_output_text="Fixed: uploads retry after a timeout."),
    name="changelog",
)
changelog = PydanticAITactic(writer, input_type=str)
tool = tactic_as_tool(changelog, asynchronous=True)  # (1)!


def model(messages, info):  # (2)!
    if len(messages) == 1:
        args = {"task": "Retry uploads after a timeout"}
        return ModelResponse(parts=[ToolCallPart("changelog", args)])
    entry = messages[-1].parts[0].content
    return ModelResponse(parts=[TextPart(f"Release notes: {entry}")])


planner = Agent(FunctionModel(model), tools=[tool])
print(planner.run_sync("Write the release notes for #812.").output)
#> Release notes: Fixed: uploads retry after a timeout.
```

1. The tool awaits `changelog.arun()`, so the changelog agent runs with
   `agent.run()` inside the planner's run. A synchronous tool would call
   `agent.run_sync()` there, and Pydantic AI raises `UserError`.
2. As above, a `FunctionModel` plays the planner's model: it calls the
   tool, then answers with the entry the tool returned.

## Serving and packaging

A wrapped agent serves like every tactic. With the agent and tactic from
the first example in `changelog.py`, point `lllm serve` at the module and
attribute:

```bash
lllm serve changelog:changelog --port 8000
curl -X POST localhost:8000/run -H 'content-type: application/json' \
  -d '{"input": "Retry uploads after a timeout"}'
```

The reply carries the entry as JSON under `output`, and `/info`, `/stream`,
and the OpenAPI docs work as for any tactic.

In a [package](packages.md), point an entry at the tactic, or at a factory
such as `def build_tactic() -> PydanticAITactic`, which is called once when
the package loads, and name the runtime:

```toml
[tactics.changelog]
entry = "changelog:build_tactic"
runtime = "pydantic-ai"
description = "Write a changelog entry for a pull request."
```

## Options

| Parameter | Default | Purpose |
| --- | --- | --- |
| `name` | `agent.name` | The tactic's name |
| `description` | `agent.description` | What `info()` and `/info` show |
| `input_type` | none: input passes through | Validates input and publishes its schema |
| `output_type` | `agent.output_type` | Validates output (`Any` when `result_mode="result"`) |
| `input_mapper`, `output_mapper` | none | Functions that build the prompt and the output |
| `input_mode` | `"auto"` | How a model input is sent: `"json"`, `"text"`, `"dict"`, `"python"` |
| `result_mode` | `"output"` | `"result"` returns the whole run result |
| `stream_mode` | `"output"` | `"text"`, `"response"`, or `"raw"` |
| `run_kwargs` | `{}` | Keyword arguments for every agent run |
| `include_context_metadata` | `True` | Pass the `CallContext` as run `metadata` |
| `package_ref`, `service_ref`, `examples`, `metadata` | none | Published in `info()`, as on every tactic |

Every option except `name` can also come as a dict in `config=` (or a
`PydanticAITacticConfig`), for example loaded from a file; keyword arguments
win. When neither these nor the agent give a description, the tactic uses a
`description` attribute or docstring on your subclass of
`PydanticAITactic`, then your own agent class's docstring, and never
Pydantic AI's own docstrings.

See [Runtimes](runtimes.md) for how this adapter fits next to plain
functions, native tactics, and remote services.
