# 5. Native Tactics

**Goal:** write a program of agents as a `Tactic`, declare its agents in
configuration, type its input and output with Pydantic, inspect per-call
sessions, compose and extend tactics, run them in batches, and finally serve one
over HTTP.

A **tactic** is LLLM's unit of agentic work: the program. A `Tactic` subclass:

- implements `call()` with full control over every agent turn, using the
  `open` / `receive` / `respond` primitives from the earlier lessons;
- declares the agents it needs in `agent_group`, and gets those agents built
  from a config dictionary, so models, prompts, tools, proxies, and limits are
  declared as data rather than wired in code;
- runs every call on fresh agents and records a session with each agent call,
  retry, tool round, sub-tactic call, and cost.

Think of it as an `nn.Module` for agents: composable, isolated between calls,
and easy to test. Every tactic is also a typed, service-ready unit, so the same
class can be served over HTTP (Step 11).

Where Pydantic fits: `input_type` and `output_type` declare what a tactic takes
and returns. Every call validates the task before `call()` runs and the result
after it returns, and the same types describe the tactic when it is served.

## Prerequisites

Finish [Lesson 4](04-tools.md). Calling a tactic sends requests to your model
provider; defining tactics and building them from config does not.

---

## Step 1: Define a tactic

```python
from lllm import Tactic


class SummarizerTactic(Tactic):
    name = "summarizer"
    agent_group = ["writer"]

    def call(self, task: str) -> str:
        writer = self.agents["writer"]
        writer.open("session")
        writer.receive(f"Summarize this text in 3 bullet points:\n\n{task}")
        return writer.respond().content
```

A native tactic defines:

- `name`: a unique identifier used for registration, sessions, and logs;
- `agent_group`: the agent names it needs, each of which must appear in the
  config;
- `call(task, ...)`: the orchestration logic. `self.agents` maps each name in
  `agent_group` to a ready-to-use `Agent`.

A tactic without `agent_group` is plain Python: `call()` can do anything, and it
still gets typing, sessions, composition, and serving. `NativeTactic` is a
subclass of `Tactic` that only adds a check that `agent_group` is set.

---

## Step 2: Configure the agents

A tactic receives a config dictionary when it is constructed:

```python
config = {
    "global": {
        "model_name": "gpt-4o",
        "model_args": {"temperature": 0.2},
    },
    "agent_configs": [
        {
            "name": "writer",
            "system_prompt": "You are a concise technical writer.",
            "model_args": {"max_completion_tokens": 1000},
        },
    ],
}

tactic = SummarizerTactic(config)
print(tactic)                                # SummarizerTactic(name='summarizer', agents=['writer'])
print(tactic.agents["writer"].model_args)    # {'temperature': 0.2, 'max_completion_tokens': 1000}
```

`global` holds defaults that are deep-merged into every entry of
`agent_configs`: nested dictionaries such as `model_args` merge key by key, and
the agent's own values win. The same config in YAML:

```yaml
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.2

agent_configs:
  - name: writer
    system_prompt: You are a concise technical writer.
    model_args:
      max_completion_tokens: 1000
```

Keys an agent entry understands:

| Key | Meaning |
|---|---|
| `name` | Agent name, matched against `agent_group` |
| `model_name` | Model string (required, usually set under `global`) |
| `system_prompt` | Inline system prompt text |
| `system_prompt_path` | Path of a registered prompt, used instead of `system_prompt` |
| `model_args` | Model arguments such as `temperature` |
| `api_type` | `completion` (default) or `response` |
| `max_exception_retry`, `max_interrupt_steps`, `max_llm_recall` | Loop limits ([Lesson 2](02-agents-and-dialogs.md)) |
| `context_manager` | Context-window policy ([Lesson 2](02-agents-and-dialogs.md), Step 10) |
| `tools`, `proxy` | Agent tools and proxies ([Lesson 4](04-tools.md)) |
| `skills` | Agent skills to expose to the model |
| `inference_hints` | Serving hints for self-hosted model servers |

Unknown keys are passed to the model as extra arguments, with a warning that
lists the known keys, so typos are easy to spot. At the top level, `invoker`
selects the invoker (default `"litellm"`) and `max_workers` sets the default
thread count for batch calls (Step 8).

---

## Step 3: Call the tactic

```python
result = tactic("Large language models are increasingly used as components of larger programs ...")
print(result)
```

Each call:

1. makes a per-call copy of the tactic and builds fresh agents from the config,
   wrapped so that every `respond()` is recorded;
2. validates the task against `input_type`, if declared (Step 5);
3. runs `call()`;
4. validates the result against `output_type`, if declared;
5. records the outcome, agent calls, sub-tactic calls, costs, and any error with
   its traceback in a `TacticCallSession`;
6. returns the result, or re-raises the exception.

Because agents are rebuilt for every call, nothing leaks from one task into the
next, and concurrent calls are safe.

---

## Step 4: Coordinate several agents

```python
class ResearchTactic(Tactic):
    name = "researcher"
    agent_group = ["finder", "synthesizer"]

    def call(self, task: str) -> str:
        finder = self.agents["finder"]
        synthesizer = self.agents["synthesizer"]

        # Stage 1: gather raw facts
        finder.open("gather")
        finder.receive(f"List 5 key facts about: {task}")
        facts = finder.respond().content

        # Stage 2: synthesize them
        synthesizer.open("synthesize")
        synthesizer.receive(f"Facts:\n{facts}\n\nWrite a short paragraph.")
        return synthesizer.respond().content


research_config = {
    "global": {"model_name": "gpt-4o"},
    "agent_configs": [
        {"name": "finder", "system_prompt": "You gather accurate factual information."},
        {"name": "synthesizer", "system_prompt": "You write clear, concise summaries."},
    ],
}

researcher = ResearchTactic(research_config)
print(researcher("quantum error correction"))
```

Each agent keeps its own dialogs, and `call()` decides exactly what flows
between them.

---

## Step 5: Type the task and the result

Declare Pydantic types and the tactic validates both ends of every call:

```python
from pydantic import BaseModel


class BriefInput(BaseModel):
    topic: str
    audience: str = "engineers"


class BriefOutput(BaseModel):
    title: str
    bullets: list[str]


class BriefTactic(Tactic):
    name = "brief"
    agent_group = ["writer"]
    input_type = BriefInput
    output_type = BriefOutput

    def call(self, task: BriefInput) -> BriefOutput:
        writer = self.agents["writer"]
        writer.open("brief")
        writer.receive(f"Write three one-line bullet points about {task.topic} for {task.audience}.")
        lines = writer.respond().content.splitlines()
        bullets = [line.lstrip("-* ").strip() for line in lines if line.strip()]
        return BriefOutput(title=task.topic.title(), bullets=bullets)


brief = BriefTactic({
    "global": {"model_name": "gpt-4o"},
    "agent_configs": [{"name": "writer", "system_prompt": "You write crisp technical briefs."}],
})

print(brief({"topic": "package refs"}))   # a dict is validated into BriefInput
```

`brief({"audience": "managers"})` raises `lllm.SchemaError` before any agent
runs, because `topic` is required; the Pydantic validation error is attached as
its cause. Validation uses Pydantic's `TypeAdapter`, so plain types such as
`str` or `list[str]` work too.

To make the model's own output typed, combine this with the parsers and
`format=` from [Lesson 3](03-prompts.md).

---

## Step 6: Inspect the call session

```python
session = tactic("Some text to summarize ...", return_session=True)

print(session.state)              # "success"
print(session.delivery)           # the value call() returned
print(session.agent_call_count)   # number of respond() calls
print(session.total_cost.cost)    # dollar cost reported by the provider
print(session.summary())          # compact overview dict
```

`session.agent_sessions` maps each agent name to the `AgentCallSession` of
every `respond()` it made, with its parse retries, tool rounds, and cost.
[Lesson 7](07-logging.md) shows how to log and persist sessions.

---

## Step 7: Compose tactics

Tactics can call other tactics like functions:

```python
class PipelineTactic(Tactic):
    name = "pipeline"
    agent_group = ["coordinator"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.researcher = ResearchTactic(config)   # tracked in self.sub_tactics

    def call(self, task: str) -> str:
        research = self.researcher(task)

        coordinator = self.agents["coordinator"]
        coordinator.open("plan")
        coordinator.receive(f"Based on this research:\n{research}\n\nCreate an action plan.")
        return coordinator.respond().content


pipeline_config = {
    "global": {"model_name": "gpt-4o"},
    "agent_configs": research_config["agent_configs"] + [
        {"name": "coordinator", "system_prompt": "You turn research into concrete plans."},
    ],
}

pipeline = PipelineTactic(pipeline_config)
print(pipeline.sub_tactics.keys())   # dict_keys(['researcher'])

session = pipeline("adopting LLM agents in a research lab", return_session=True)
print(session.summary()["sub_tactic_calls"])     # 1
print(list(session.sub_tactic_sessions))         # ['researcher']
```

- Assigning a tactic to an attribute after `super().__init__()` adds it to
  `self.sub_tactics`.
- Every tactic called inside `call()` is recorded in the caller's session
  automatically, including calls made through `bcall()`, `acall()`, and
  `ccall()`. The parent's `total_cost` includes the sub-tactics' costs.

---

## Step 8: Run many tasks

```python
import asyncio

topics = ["topic A", "topic B", "topic C", "topic D"]

# Thread pool; results come back in input order
results = researcher.bcall(topics, max_workers=4)

# Keep going when some tasks fail: exceptions are returned in place of results
results = researcher.bcall(topics, max_workers=4, fail_fast=False)

# One async call, for use inside an async application
result = asyncio.run(researcher.acall("a single async task"))


# Stream results as they complete
async def stream_results():
    async for index, result in researcher.ccall(topics, max_workers=4):
        print(f"Task {index} done: {result[:60]}")

asyncio.run(stream_results())
```

Each task gets its own fresh agents, so parallel tasks never share dialogs.
`bcall()` also accepts `tags=`, `metadata=`, and `return_sessions=True`.

---

## Step 9: Extend a tactic by inheritance

Subclass a concrete tactic to reuse its `call()` and add stages:

```python
class WritingPipeline(Tactic):
    """Two-agent pipeline: outline, then article."""

    name = "writing_pipeline"
    agent_group = ["outliner", "writer"]

    def call(self, task: str) -> str:
        outliner = self.agents["outliner"]
        writer = self.agents["writer"]

        outliner.open("outline")
        outliner.receive(f"Create a concise 5-point outline about: {task}")
        outline = outliner.respond().content

        writer.open("write")
        writer.receive(f"Expand this outline into a short article:\n\n{outline}")
        return writer.respond().content


class EditedWritingPipeline(WritingPipeline):
    name = "edited_writing_pipeline"
    agent_group = ["outliner", "writer", "editor"]   # adds a third agent

    def call(self, task: str) -> str:
        draft = super().call(task)                     # reuse outline and draft stages

        editor = self.agents["editor"]
        editor.open("edit")
        editor.receive(f"Polish this draft for clarity and concision:\n\n{draft}")
        return editor.respond().content


writing_config = {
    "global": {"model_name": "gpt-4o", "model_args": {"temperature": 0.7}},
    "agent_configs": [
        {"name": "outliner", "system_prompt": "You create concise, structured outlines."},
        {"name": "writer", "system_prompt": "You expand outlines into engaging articles."},
        {"name": "editor", "system_prompt": "You polish prose for clarity and concision."},
    ],
}

print(EditedWritingPipeline(writing_config)("the rise of agentic AI"))
```

A shared base class can carry helpers without being a usable tactic on its own.
Leave `name` off it:

```python
class WritingBase(Tactic):
    """Shared helpers; never registered because it has no name."""

    agent_group = ["writer"]

    def polish(self, text: str) -> str:
        writer = self.agents["writer"]
        writer.open("polish")
        writer.receive(f"Polish this:\n\n{text}")
        return writer.respond().content


class BlogPostTactic(WritingBase):
    name = "blog_post"
    agent_group = ["writer", "seo_reviewer"]

    def call(self, task: str) -> str:
        draft = self.polish(task)
        reviewer = self.agents["seo_reviewer"]
        reviewer.open("seo")
        reviewer.receive(f"Suggest SEO keywords for:\n\n{draft}")
        return reviewer.respond().content
```

---

## Step 10: Register tactics and build them by name

Defining a subclass does not register it anywhere. Register it in a runtime
registry, then build it from config by name:

```python
from lllm import Runtime, build_tactic, register_tactic_class

runtime = Runtime()
register_tactic_class(SummarizerTactic, registry=runtime)

summarizer = build_tactic(config, name="summarizer", registry=runtime)
print(type(summarizer).__name__)   # SummarizerTactic
```

- With `"tactic_type": "summarizer"` in the config, `build_tactic(config,
  registry=runtime)` needs no `name`.
- `class SummarizerTactic(Tactic, registry=runtime): ...` registers the class
  when it is defined.
- Omit `registry=` to use the default runtime registry.

The tactic's agents resolve registered prompts (`system_prompt_path`), tools,
and proxies from the same registry. In an LLLM package, every tactic class with
a `name` in the package's tactic folders is registered when the package loads.
`lllm create my-app` generates such a project, with prompts, configs, a
tactic, and a smoke test ([Lesson 1](01-quick-start.md));
[Lesson 6](06-packages-and-hub.md) covers packages, sharing, and publishing.

---

## Step 11: Serve it

Every tactic is also a service-ready unit with typed `run` / `arun` methods and
an `info()` description built from its name, docstring, and types, so you can
serve it as it is. Put the tactic in a module:

```python
# summarizer_service.py
from lllm import Tactic


class SummarizerTactic(Tactic):
    """Summarize text in three bullet points."""

    name = "summarizer"
    agent_group = ["writer"]
    input_type = str
    output_type = str

    def call(self, task: str) -> str:
        writer = self.agents["writer"]
        writer.open("session")
        writer.receive(f"Summarize this text in 3 bullet points:\n\n{task}")
        return writer.respond().content


CONFIG = {
    "global": {"model_name": "gpt-4o"},
    "agent_configs": [{"name": "writer", "system_prompt": "You are a concise technical writer."}],
}

tactic = SummarizerTactic(CONFIG)
```

From the directory that contains `summarizer_service.py`, check what will be
served, then start the server:

```bash
lllm inspect summarizer_service:tactic
lllm serve summarizer_service:tactic --port 8000
```

`lllm inspect` prints the tactic's name, runtime, description, and
capabilities:

```text
summarizer (native)
Summarize text in three bullet points.
capabilities: acall, arun, call, run
```

The entrypoint can name a tactic instance, as here, or a function that returns
one. Call the service:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input": "LLLM gives you low-level control of every agent turn.", "context": {"trace_id": "demo-1"}}'
```

The response carries the tactic's output, a request id, and the tactic name:

```json
{"output": "...", "request_id": "...", "tactic": "summarizer"}
```

`GET /info` returns the description and schemas, and `GET /health` reports
readiness. To build the app in Python instead, for example to run it under your
own ASGI server:

```python
from lllm import create_tactic_app

app = create_tactic_app(tactic)
```

Inside `call()`, `self.call_context` holds the request's `CallContext` (request
id, trace id, tags, metadata) when the tactic is served, and `None` when you
call it directly; a `call()` that declares a `context` parameter receives it
there too. `NativeTacticAdapter(tactic, ...)` remains available when you want to
serve a tactic under another name or description, attach a `package_ref`, or
pass fixed keyword arguments to `call()` with `run_kwargs`.

For production you can wrap the tactic with call guardrails and logging before
serving it:

```python
from lllm import InMemoryProxyLog, ProxyTactic, SandboxPolicy, SandboxedTactic

log = InMemoryProxyLog()
guarded = ProxyTactic(
    SandboxedTactic(
        tactic,
        policy=SandboxPolicy(timeout_seconds=120, max_input_bytes=20_000),
        name="summarizer",
    ),
    sink=log.append,
    name="summarizer",
)
app = create_tactic_app(guarded)
```

- `SandboxedTactic` rejects oversized inputs and outputs and disallowed request
  metadata keys, and enforces `timeout_seconds` on async calls, which is how the
  service runs tactics. It is a call guard, not OS-level isolation.
- `ProxyTactic` sends one `ProxyRecord` per call (request id, state, latency,
  error) to `sink`; payloads are captured only with `capture_inputs=True` or
  `capture_outputs=True`.

[Lesson 9](09-serving.md) covers serving and composition in depth.

---

## Summary

| Concept | API |
|---|---|
| Define a tactic | `class MyTactic(Tactic): name = ...; agent_group = [...]` |
| Implement it | `def call(self, task)`, using `self.agents["name"]` |
| Declare agents | config `global` + `agent_configs` |
| Call it | `tactic(task)` |
| Typed input and output | `input_type = Model`, `output_type = Model` |
| Session | `tactic(task, return_session=True)` |
| Compose | `self.sub = OtherTactic(config)` in `__init__`; sub-calls are recorded automatically |
| Batch and async | `tactic.bcall(tasks)`, `await tactic.acall(task)`, `tactic.ccall(tasks)` |
| Extend | subclass and call `super().call(task)` |
| Register and build | `register_tactic_class(cls, registry=...)`, `build_tactic(config, name=..., registry=...)` |
| Start a project | `lllm create my-app` |
| Serve | `lllm serve module:tactic`, `create_tactic_app(tactic)` |

See the [Tactic reference](../core/tactic.md) and the
[Config reference](../core/config.md) for details.

**Next:** [6. Packages and the Hub](06-packages-and-hub.md)
