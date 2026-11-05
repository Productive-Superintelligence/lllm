# Tactics

A tactic is the top-level abstraction of the native runtime: the "program" that
wires agents (callers) to prompts (functions). It is the uppermost building
block of an agentic system and is designed to be local, functional, and
composable.

How it relates to the rest of the runtime:

- **Prompts** define what an agent can do on one turn: template, parser, tools,
  handler.
- **Agents** execute prompts through the call loop, managing dialogs and retries.
- **Tactics** define how a *group* of agents collaborates on a task.

A tactic accepts a `str` or a Pydantic model and returns a `str` or a Pydantic
model; subclasses narrow these types for their interface.

`lllm.Tactic` is one class with two faces. As a program, you write `call()`
and, if you need agents, declare `agent_group`. As a protocol tactic, every
`Tactic` has validated `run()` / `arun()` and `info()`, so it can be served,
packaged, and composed with tactics from other runtimes without an adapter.

```python
from lllm import Tactic   # the same object as lllm.runtimes.native.Tactic
```

---

## Defining a tactic

Subclass `Tactic`, set `name` and `agent_group`, implement `call()`:

```python
from lllm import Tactic


class Analytica(Tactic):
    name = "analytica"
    agent_group = ["analyzer", "synthesizer"]

    def call(self, task: str, **kwargs) -> str:
        analyzer = self.agents["analyzer"]
        synthesizer = self.agents["synthesizer"]

        analyzer.open("work", prompt_args={"task": task})
        analysis = analyzer.respond()

        synthesizer.open("synth", prompt_args={"analysis": analysis.content})
        return synthesizer.respond().content
```

```python
config = {
    "global": {"model_name": "o4-mini", "model_args": {"max_completion_tokens": 20000}},
    "agent_configs": [
        {"name": "analyzer", "system_prompt_path": "analytica/analyzer_system"},
        {"name": "synthesizer", "system_prompt_path": "analytica/synthesizer_system"},
    ],
}

tactic = Analytica(config)
result = tactic("Analyze this paper")
```

With `agent_group` set, the tactic builds a fresh, configured `Agent` for each
name every time it runs `call()` (or streams), and puts them in `self.agents`.
Without it, `call()` is plain Python: deterministic steps, or orchestration
over other tactics, tools, and runtimes.

### `call()` or a protocol hook

A tactic writes one program: `call()`, or the protocol hook `_run()` (or
`_arun()` for async-only code). The native entry points (`tactic(task)`,
`acall`, `bcall`, `ccall`) run whichever one the class writes, inside a
`TacticCallSession`:

```python
from lllm import Tactic


class Upper(Tactic):
    name = "upper"

    def _run(self, input_value, *, context=None):
        return input_value.upper()


upper = Upper()
upper("hi")       # 'HI', recorded in a TacticCallSession
upper.run("hi")   # 'HI'
```

- `run()` and `arun()` run that program too.
- `call()` runs on a per-call copy of the tactic with fresh agents (the v1
  design), whichever entry point starts it. A hook runs on the tactic itself,
  like any protocol tactic, so it should not keep per-call state on `self`.
  Agents are built fresh only for `call()` programs and streams, so a class
  that declares `agent_group` but writes only a hook gets a warning: write
  `call()` instead.
- A hook always receives a `CallContext` as `context=` (a fresh one when the
  caller passes none).
- A hook-only tactic has no `call()`: `upper.call("hi")` raises
  `TacticUnsupportedError` and points to `tactic(task)`, `run()`, and `arun()`.
  A hook that hands its call back, through `super()._run()` or `self.call()`,
  raises the same error at once instead of recursing.
- An `_arun()`-only tactic also gets `run()` and `tactic(task)`, which run the
  hook to completion.
- `call()` may be `async def`. The async entry points (`acall`, `arun`,
  `ccall`, and a service's `/run`) await it on the caller's event loop; the
  sync ones (`tactic(task)`, `run()`, `bcall`) run it on one long-lived
  background loop, so clients it keeps between calls stay usable. From async
  code, use the async entry points: calling `tactic(task)` on an async
  program from inside another async program raises `TacticUnsupportedError`.
- A tactic that writes neither raises `TacticUnsupportedError` when called.

`NativeTactic` is a thin subclass that only insists on `agent_group` (raising
`AssertionError` without it), kept for code that subclasses it.

### Constructor

```python
Tactic(config=None, runtime=None, registry=None, tactic_path=None, context=None, observer=None,
       *, name=None, description=None, package_ref=None, service_ref=None, examples=None, metadata=None)
```

| Argument | Meaning |
| --- | --- |
| `config` | Dict or Pydantic model. With `agent_group`: `global`, `agent_configs`, and optionally `invoker`, `invoker_config`, `max_workers`. |
| `registry` / `runtime` | Registry used to resolve prompts, tools, and configs (the same object; pass one). Defaults to the default runtime. |
| `tactic_path` | Stable identity recorded on sessions, e.g. `"my_pkg::analytica"`. |
| `context` | A `TacticContext(registry=..., tactic_path=..., observer=..., metadata=...)` bundling the above. |
| `observer` | Object with tactic lifecycle hooks; see [Logging](logging.md#persisting-sessions-with-an-observer). |
| `name`, `description`, `package_ref`, `service_ref`, `examples`, `metadata` | Override what `info()` publishes. The description defaults to the class docstring. A name is one path segment (no `/`, `:`, `\`, `;`, or `%`); an `Enum` member stands for its value. |

---

## Registration

A tactic class must be in a registry for `build_tactic`, config `tactic_type`,
or tactic refs such as `my_pkg.tactics:analytica` to find it.

- **Package discovery** (`[tactics] paths` in `lllm.toml`) imports the modules and
  registers every `Tactic` subclass that defines `name`, under that name.
- **At class definition**, by passing a registry as a class keyword:

    ```python
    from lllm import Runtime, Tactic

    registry = Runtime()


    class Analytica(Tactic, runtime=registry):   # or registry=registry
        name = "analytica"
        agent_group = ["analyzer", "synthesizer"]
        ...
    ```

- **Explicitly**, with `lllm.register_tactic_class(Analytica, runtime=registry)`.

Defining a subclass without one of these does not register it anywhere.

### Building from config

```python
from lllm import build_tactic, resolve_config

tactic = build_tactic(resolve_config("experiment"), name="analytica")
result = tactic("Analyze this paper")
```

Or let the config name the tactic:

```yaml
# configs/experiment.yaml
tactic_type: analytica

agent_configs:
  - name: analyzer
    model_name: o4-mini
    system_prompt_path: analytica/analyzer_system
  - name: synthesizer
    model_name: o4-mini
    system_prompt_path: analytica/synthesizer_system
```

```python
tactic = build_tactic(resolve_config("experiment"))   # reads tactic_type
```

`build_tactic(config, name=None, runtime=None, *, registry=None, **kwargs)`
looks the class up in the registry, sets the stable `tactic_path` to
`"<package>::<name>"` for package-discovered tactics, and passes extra keyword
arguments (for example `observer=`) to the constructor.

---

## Agent initialization

`agent_group` lists the agent names the tactic needs. At construction, the
tactic reads `global` and `agent_configs` from the config, merges each entry over
`global`, and parses it into an `AgentSpec`:

```yaml
global:
  model_name: o4-mini
  model_args:
    temperature: 0.1

agent_configs:
  - name: analyzer
    system_prompt_path: analytica/analyzer_system
    max_completion_tokens: 20000        # unknown key -> model_args (with a warning)
  - name: synthesizer
    system_prompt: "Synthesize this analysis: {analysis}"
```

`AgentSpec` separates config parsing from agent construction. Known keys
(`model_name`, `system_prompt`, `system_prompt_path`, `api_type`, `model_args`,
the retry budgets, `tools`, `proxy`, `skills`, `context_manager`,
`inference_hints`, `extra_settings`) are extracted; anything else is merged into
`model_args`. A missing agent, a missing `model_name`, or a prompt path that
does not resolve raises before any model call.

The constructor also builds the invoker (LiteLLM unless the config sets
`invoker`), so constructing a tactic with `agent_group` needs
`lllm-core[native]` installed. See [Configuration](config.md) for every key.

---

## Stateless execution and per-call isolation

A tactic holds no per-call state. Each call of a `call()` program, whatever the
entry point, runs on a shallow copy with fresh agents:

```python
result_a = tactic("Analyze this paper")        # call 1
result_b = tactic("Analyze this other paper")  # call 2: no interference
```

For each call, `__call__`:

1. copies the tactic (`copy.copy`), sharing the config, registry, invoker, and
   agent specs;
2. builds fresh `Agent`s from the specs, each with no dialogs, and wraps them in
   `_TrackedAgent` proxies that record every `respond()`;
3. validates the task against the input type, runs `call()` (awaiting it if it
   is async), validates the result against the output type;
4. marks the `TacticCallSession` as `"success"` or `"failure"` (re-raising the
   exception).

Concurrent calls (`bcall`, threads) never share mutable state: each gets its own
agents, dialogs, and session. Streams from an overridden sync `events()` or
`stream()` get the same treatment, one copy per stream. Protocol hooks
(`_run()`, `_arun()`) and async `astream()`/`aevents()` overrides run on the
instance itself, like any protocol tactic: keep per-call state out of `self`
there (an `agent_group` tactic that overrides them warns).

---

## Session tracking

Every `agent.respond()` inside `call()` is recorded into a `TacticCallSession`.
You write normal agent code; tracking is invisible:

```python
def call(self, task: str, **kwargs) -> str:
    analyzer = self.agents["analyzer"]
    analyzer.open("work", prompt_args={"task": task})
    result = analyzer.respond()      # recorded into the TacticCallSession
    return result.content
```

Ask for the session with `return_session=True`:

```python
session = tactic("Analyze this paper", return_session=True)

print(session.state)              # "success"
print(session.delivery)           # the return value of call()
print(session.total_cost)         # InvokeCost across agents and sub-tactics
print(session.agent_call_count)   # number of agent.respond() calls
print(session.summary())          # compact dict
```

`tactic(task, session_name=None, tags=None, metadata=None, return_session=False, **kwargs)`:
extra keyword arguments are passed to `call()` (or the hook). `session_name`
defaults to `"<name>_<task hash>_<timestamp>"`; it, `tags`, and `metadata` are
handed to the tactic's observer, which is how sessions get persisted (see
[Logging](logging.md)).

### TacticCallSession

| Field | Type | Description |
| --- | --- | --- |
| `tactic_name` | `str` | The tactic's name: the constructor's `name=`, else `Tactic.name`, else the class name |
| `tactic_path` | `str \| None` | Stable identity, `"<package>::<name>"` when known |
| `state` | `str` | `"initial"`, `"running"`, `"success"`, or `"failure"` |
| `agent_sessions` | `Dict[str, List[AgentCallSession]]` | Per-agent call records |
| `sub_tactic_sessions` | `Dict[str, List[TacticCallSession]]` | Sessions of tactics called during this call |
| `delivery` | `Any` | Return value of `call()` (or the hook) on success |
| `error` | `str \| None` | `"ExcType: message"` on failure |
| `error_traceback` | `str \| None` | Full traceback on failure |

Properties:

```python
session.agent_cost             # cost from this tactic's agents
session.sub_tactic_cost        # cost from sub-tactic sessions
session.total_cost             # agent_cost + sub_tactic_cost (recursive)
session.agent_call_count       # agent.respond() calls
session.sub_tactic_call_count  # sub-tactic calls
```

---

## Sub-tactic composition

Tactics compose like `nn.Module` children. Assign a tactic as an attribute and
it is tracked in `sub_tactics`:

```python
from lllm import Tactic


class ResearchTactic(Tactic):
    name = "research"
    agent_group = ["planner"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.analyzer = Analytica(config, registry=self.context.registry)
        self.searcher = Searcher(config, registry=self.context.registry)

    def call(self, task: str, **kwargs) -> str:
        planner = self.agents["planner"]
        planner.open("plan", prompt_args={"task": task})
        plan = planner.respond()

        # Sub-tactic calls: each runs with its own fresh agents
        analysis = self.analyzer(plan.content)
        search = self.searcher(task)

        planner.receive(f"Analysis: {analysis}\nSearch: {search}")
        return planner.respond().content


tactic.sub_tactics      # {'analyzer': Analytica(...), 'searcher': Searcher(...)}
```

Sessions roll up automatically. Any tactic called while another tactic is
running records its session into the caller's, under its name
(`session.sub_tactic_sessions["analytica"]`, or the `name=` it was constructed
with), so sub-tactic agent calls and cost reach the parent's `total_cost`. This
tracks the call, not the attribute: a tactic created inside `call()` is
recorded too, and so are calls made through `acall`, `bcall`, and `ccall`,
whose worker threads carry the caller's context.

---

## Tactic inheritance

Concrete tactics subclass like any Python class, so you can build reusable
pipeline bases and extend them.

### Extending a concrete tactic

```python
class WritingPipeline(Tactic):
    """Reusable base: outline, then draft."""
    name = "writing_pipeline"
    agent_group = ["outliner", "writer"]

    def call(self, task: str) -> str:
        outliner = self.agents["outliner"]
        writer = self.agents["writer"]

        outliner.open("outline")
        outliner.receive(f"Create a concise outline about: {task}")
        outline = outliner.respond().content

        writer.open("write")
        writer.receive(f"Expand this outline:\n\n{outline}")
        return writer.respond().content


class EditedWritingPipeline(WritingPipeline):
    """Adds an editing stage on top of the base pipeline."""
    name = "edited_writing_pipeline"
    agent_group = ["outliner", "writer", "editor"]

    def call(self, task: str) -> str:
        draft = super().call(task)        # reuse outline -> draft

        editor = self.agents["editor"]
        editor.open("edit")
        editor.receive(f"Polish this draft:\n\n{draft}")
        return editor.respond().content
```

`super().call(task)` works because agents are built for **every** name in the
subclass's `agent_group` before `call()` runs.

### Abstract bases with `register=False`

`register=False` keeps a helper base out of the registry even when a registry is
passed:

```python
class BasePipeline(Tactic, register=False):
    """Shared helpers; never registered."""

    def _run_stage(self, agent_name: str, alias: str, message: str) -> str:
        agent = self.agents[agent_name]
        agent.open(alias)
        agent.receive(message)
        return agent.respond().content


class SummaryPipeline(BasePipeline):
    name = "summary_pipeline"
    agent_group = ["extractor", "writer"]

    def call(self, text: str) -> str:
        facts = self._run_stage("extractor", "extract", f"Extract key facts:\n{text}")
        return self._run_stage("writer", "write", f"Summarise:\n{facts}")
```

### Typed I/O inheritance

Subclasses can tighten the output type:

```python
class BaseAnalysis(Tactic, register=False):
    agent_group = ["analyzer"]

    def call(self, task: str) -> str:
        agent = self.agents["analyzer"]
        agent.open("analyze")
        agent.receive(task)
        return agent.respond().content


class StructuredAnalysis(BaseAnalysis):
    name = "structured_analysis"
    output_type = AnalysisOutput

    def call(self, task: str) -> AnalysisOutput:
        return AnalysisOutput.model_validate_json(super().call(task))
```

A type the subclass declares itself wins over an inherited one under either
spelling: an `output_model` set on the subclass replaces an inherited
`output_type`, and `input_model` likewise replaces `input_type`.

---

## Batch and concurrent execution

Model calls are I/O-bound, so tactics fan out on thread pools. Each task gets
its own agents; there is no lock contention.

### Synchronous batch

```python
tasks = ["Analyze paper A", "Analyze paper B", "Analyze paper C"]
results = tactic.bcall(tasks, max_workers=3)   # results[i] belongs to tasks[i]
```

`bcall(tasks, max_workers=None, fail_fast=True, tags=None, metadata=None,
return_sessions=False, **kwargs)` returns results in input order. With
`fail_fast=True` the first failure raises; with `fail_fast=False` exceptions are
returned in place of results.

### Async single

```python
result = await tactic.acall("Analyze this paper")
```

`acall` runs the call in the default executor so it does not block the event
loop.

### Async concurrent (fastest first)

```python
async for idx, result in tactic.ccall(tasks, max_workers=3):
    print(f"Task {idx} finished: {result}")
```

Yields `(index, result)` pairs as tasks complete, not in input order.

All three accept `return_session(s)=True` to get `TacticCallSession` objects.
`max_workers` defaults to the config's `max_workers` key (4 if unset).

---

## Typed I/O with Pydantic

For shareable tactics, declare the input and output types so callers can
validate against them and services can publish schemas:

```python
from pydantic import BaseModel
from lllm import Tactic


class AnalysisInput(BaseModel):
    topic: str
    depth: int = 3


class AnalysisOutput(BaseModel):
    reasoning: str
    conclusion: str
    confidence: float


class Analytica(Tactic):
    name = "analytica"
    input_type = AnalysisInput
    output_type = AnalysisOutput
    agent_group = ["analyzer"]

    def call(self, task: AnalysisInput) -> AnalysisOutput:
        analyzer = self.agents["analyzer"]
        analyzer.open("work", prompt_args={"topic": task.topic, "depth": task.depth})
        parsed = analyzer.respond().parsed          # prompt uses DefaultTagParser
        tags = parsed["xml_tags"]
        return AnalysisOutput(
            reasoning=tags["reasoning"][0],
            conclusion=tags["answer"][0],
            confidence=float(tags["confidence"][0]),
        )


tactic({"topic": "attention", "depth": 2})   # the dict is validated into AnalysisInput
```

`input_type` / `output_type` (or the aliases `input_model` / `output_model`)
accept any type Pydantic's `TypeAdapter` can validate. The tactic validates the
task before `call()` and the result after it, and `tactic.info()` publishes both
as JSON schema, which is what a service's `/info` returns.

Simple tactics can stay `str` in, `str` out.

---

## Tactics as tools

Tactic tools make the abstraction graph recursive. A `Tactic` is the top-level
orchestration unit and a `Prompt` is the low-level contract for one turn;
exposing a tactic as a tool lets a turn call back into a complete agentic
subsystem. A planner prompt can invoke a code-review tactic, that tactic runs
its own agents and prompts, and the result returns through the normal tool loop.

Mechanically, a tactic tool is a generated `Function` backed by a decorated
tactic method. Use `@tool` when a plain function is enough; use `@tactictool`
when the implementation needs config, agents, prompts, or several turns.

### Exposing a method

```python
from pydantic import BaseModel
from lllm import Tactic, tactictool


class CodeInput(BaseModel):
    code: str
    language: str = "python"


class CodeReviewTactic(Tactic):
    """Review code and return concrete issues."""
    name = "code_review"
    agent_group = ["reviewer"]

    @tactictool("code_review", description="Review code and list issues.", config="shared_pkg:default")
    def call(self, task: CodeInput) -> str:
        reviewer = self.agents["reviewer"]
        reviewer.open("review", prompt_args={"language": task.language})
        reviewer.receive(task.code)
        return reviewer.respond().content
```

`tactictool(name, *, description=None, input_model=None, output_model=None, config=None)`:

- **`name`** is required and is the tool name the model sees.
- **Input schema**: a `BaseModel` parameter (or `input_model=`) becomes the
  tool's JSON schema; the model's arguments are validated into it. Otherwise
  annotated parameters are mapped to JSON types, and unannotated ones become
  strings with a warning.
- **Description**: the decorator's `description`, else the method docstring,
  else the class docstring, else a generated one (with a warning).
- **Result**: a `BaseModel` return value is sent as JSON, anything else with
  `str()`.
- **`config`** is required for refs: the URL identifies the class, not the
  config to instantiate it with. Pass a config ref (resolved relative to the
  tactic's package, `base` inheritance applied) or an inline dict.

Each tool call instantiates the tactic with that config and calls the method
directly.

### Referencing tactic tools

Put the tactic's URL in a prompt's `function_list`:

```python
from lllm import Prompt

prompt = Prompt(
    path="agent/system",
    prompt="Use the available tools when useful.",
    function_list=["shared_pkg.tactics:code_review"],
)
```

Full URLs (`shared_pkg.tactics:code_review`) are canonical; package shorthand
(`shared_pkg:code_review`) and bare names resolve relative to the prompt's
package, then the runtime's default namespace.

A tactic can expose several tools. Select one with a URL fragment:

```python
from lllm import Prompt, Tactic, tactictool


class ReviewTactic(Tactic):
    name = "review"
    agent_group = ["reviewer"]

    @tactictool("review_code", config="shared_pkg:default")
    def call(self, task: CodeInput) -> str:
        ...

    @tactictool("summarize", config="shared_pkg:default")
    def summarize_review(self, task: SummaryInput) -> str:
        ...


Prompt(path="lead/system", prompt="...", function_list=[
    "shared_pkg.tactics:review",            # the decorated call()
    "shared_pkg.tactics:review#summarize",  # a named method (tool name or method name)
])
```

Selection rules: a decorated `call()` is the default; a tactic with exactly one
decorated method uses it; with several decorated methods and no decorated
`call()`, a fragment is required. A tactic with no decorated methods falls back
to `call()` with a warning, which only works for callers that pass a config
explicitly, such as a proxy endpoint.

### Proxy endpoints

The same tactic can be exposed as a proxy endpoint:

```python
from lllm import BaseProxy, ProxyRegistrator


@ProxyRegistrator(path="shared", name="Shared tools", description="Shared tactic endpoints.")
class SharedProxy(BaseProxy):
    code_review = BaseProxy.tactic_endpoint(
        "shared_pkg.tactics:code_review",
        config="shared_pkg:default",
    )
```

Tactic-backed endpoints appear in `query_api_doc` (their params come from the
tactic's schema) and run through `CALL_API` like any other endpoint.
`SharedProxy.register_tactic("shared_pkg.tactics:review", endpoint="review",
config=...)` adds one imperatively.

### Agent-level tool config

To give every turn of an agent access, list the refs in config:

```yaml
global:
  model_name: gpt-4o
  tools:
    - shared_pkg.tools:search
    - shared_pkg.tactics:code_review
    - shared_pkg.proxies:market_data

agent_configs:
  - name: reviewer
    system_prompt_path: system/reviewer
```

See [Agent](agent.md#tools) for how agent tools reach the model.

This design is reference-based to avoid definition cycles: configs name tools by
URL instead of importing and building agents while a tool module loads. Regular
tool and tactic refs stay lazy until the prompt runs.

### The other direction: tactics as plain callables

```python
fn = tactic.as_tool()                         # plain callable: fn(task) -> tactic.run(task)
fn = tactic.as_tool(parameter_mode="kwargs")  # fn(**fields), validated into input_type
```

`as_tool()` hands a tactic to other agent frameworks (Pydantic AI or anything
that accepts Python callables). To give a native prompt any protocol tactic (a
`Tactic` from another package, a Pydantic AI tactic, a remote tactic), wrap it as
a `Function`:

```python
from lllm import Prompt
from lllm.runtimes.native import tactic_as_function

lookup = tactic_as_function(search_tactic, parameter_mode="kwargs")
prompt = Prompt(path="research/answer", prompt="Use tools when needed.", function_list=[lookup])
```

The tool publishes the tactic's input schema: in the default `"task"` mode as
one `task` parameter holding the whole input, and with `parameter_mode="kwargs"`
as the input model's fields, one parameter each. An input without fields, such
as `str`, keeps the single `task` parameter in either mode. Nested models are
inlined as for [`@tool`](prompts.md#tool-coupled-tools). The tool's description
is the tactic's description.

---

## Quick constructor

For prototyping without config files:

```python
from lllm import Tactic

agent = Tactic.quick(system_prompt="You are a helpful assistant.", model="gpt-4o")
agent.open("chat")
agent.receive("What is the capital of France?")
print(agent.respond().content)
```

This returns a plain `Agent`, the same object a tactic builds internally. Pass a
query as the first argument to get the reply directly; see
[Agent](agent.md#quick-constructor).

---

## Serving and the protocol

Because `Tactic` is a protocol tactic, the same object you call in Python is
the one you serve:

| Surface | What it does |
| --- | --- |
| `tactic(task, ...)`, `acall`, `bcall`, `ccall` | Native calls; return the result or a `TacticCallSession`. |
| `tactic.run(value, context=None, return_trace=False)` | Protocol call: validates input and output; `return_trace=True` returns a `CallResult` with a `CallTrace`. |
| `await tactic.arun(value, context=None)` | Async protocol call. |
| `tactic.info()` | `TacticInfo`: name, description (the class docstring unless overridden), input/output JSON schemas, capabilities, `runtime_kind`. |
| `tactic.events(task)` | Override it to stream: `aevents()` and a service's `/stream` route serve what it yields, as `TacticEvent`s. Each stream runs like `call()`: on a per-call copy with fresh agents and its own observed session. It is pulled one item at a time on a worker thread, so a slow stream never blocks the event loop and nothing runs ahead of the reader; closing the stream (or a client disconnecting) closes the generator. Overriding the protocol's sync `stream()` works the same way. |

`runtime_kind` is `"native"` for tactics that write `call()` (or declare
`agent_group`) and `"python"` for ones that only implement `_run()` or
`_arun()`. A `runtime_kind` set on a class is inherited by its subclasses.

`run()` and `arun()` accept a `CallContext` (request id, trace id, refs, tags,
metadata). It is passed to `call()` as `context=` when `call` accepts a
`context` keyword (or `**kwargs`), and the per-call copy also exposes it as
`self.call_context`:

```python
from lllm import CallContext, Tactic


class Brief(Tactic):
    """Write a short brief."""
    name = "brief"
    agent_group = ["writer"]

    def call(self, task: str, *, context=None, tone: str = "plain") -> str:
        writer = self.agents["writer"]
        writer.open("draft", prompt_args={"tone": tone})
        writer.receive(task)
        return writer.respond().content


tactic = Brief(config)
tactic.run("Summarize the release", context=CallContext(trace_id="trace-1"))
tactic.info().runtime_kind      # 'native'
```

A `_run()` or `_arun()` hook always receives the `CallContext` as `context=`.

### Serving over HTTP

Pass the tactic to `create_tactic_app`:

```python
# research_kit/service.py
from lllm import build_tactic, create_tactic_app, load_runtime, resolve_config


def build_public_tactic():
    load_runtime()
    return build_tactic(resolve_config("default"))


app = create_tactic_app(build_public_tactic())
```

```bash
uvicorn research_kit.service:app
# or let the CLI build it (run from the project folder):
lllm serve research_kit.service:build_public_tactic --port 8000

curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input": {"topic": "attention"}}'
```

`lllm serve module:attr` accepts a tactic instance, a `Tactic` subclass it can
construct with no arguments, a factory, or a plain function; point it at a
factory when the tactic needs a loaded runtime and config. It imports from the
working directory, so no `PYTHONPATH` setup is needed. See
[Tactic protocol](../protocol/tactic.md) and [Services](../protocol/services.md).

### Factories and handlers

`lllm serve module:attr`, `lllm inspect module:attr`, and a package tactic's
`entry = "module:attr"` tell a factory from a handler by one rule:

- A **factory** is called once, with no arguments, when the entry is loaded,
  and returns the tactic. A function is a factory when it takes no parameters,
  or when it can be called without arguments and its return annotation is a
  `Tactic`.
- Any other function is a **handler**: it is wrapped with `lllm.as_tactic` and
  called once per input.

```python
from lllm import Tactic, build_tactic, load_runtime, resolve_config


def build_public_tactic():                       # factory: no parameters
    load_runtime()
    return build_tactic(resolve_config("default"))


def build_named(config_name: str = "default") -> Tactic:   # factory: annotated
    load_runtime()
    return build_tactic(resolve_config(config_name))


def greet(name: str = "world") -> str:            # handler: runs on every input
    return f"Hello {name}"
```

Without its `-> Tactic` annotation, `build_named` would be a handler and
receive each input as `config_name`. An annotation that cannot be evaluated at
load time, such as a class imported only under `TYPE_CHECKING`, counts when the
class name ends in `Tactic`.

### Objects that only look like tactics

`NativeTacticAdapter` wraps an object that has `call()` or `run()` but does not
subclass `Tactic` (a class from older code, say) as a protocol tactic:

```python
from lllm import NativeTacticAdapter

public = NativeTacticAdapter(legacy_brief, description="Write a brief.", run_kwargs={"tone": "precise"})
public.run("Summarize the release")
```

It reads the name and types from the object, passes `run_kwargs` to every call,
forwards the `CallContext` as `context=` when the object's method accepts it,
and with `include_context_metadata=True` passes safe call metadata
(`lllm_request_id`, `lllm_trace_id`, refs, endpoint, tags) as `metadata=`. A
`Tactic` does not need it.

---

## Design notes

- **Tactics are stateless.** Per-call data lives on `TacticCallSession`, not on
  the tactic, so tactics are safe to call concurrently.
- **Agents are per call.** Each `call()` and each stream builds fresh agents
  from specs. The expensive objects (invoker, registry, prompts) are shared;
  only the `Agent` shells and their dialogs are new. Protocol hooks run on the
  tactic itself and do not get agents of their own (see
  [`call()` or a protocol hook](#call-or-a-protocol-hook)).
- **Tracking is transparent.** `_TrackedAgent` intercepts `respond()` without
  changing the agent API, and nested tactic calls record themselves into the
  running session.
- **Threads, not processes.** Model calls are I/O-bound; threads share the
  invoker and avoid pickling.
- **One class for the program and the protocol.** A tactic you can call in a
  notebook is the tactic you serve, package, and use as a tool.
