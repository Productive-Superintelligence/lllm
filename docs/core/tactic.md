# Tactics

A `Tactic` is the top-level abstraction in LLLM — the "program" that wires agents (callers) to prompts (functions). It is the uppermost building block of an agentic system, designed to be local, functional, and composable.

The relationship to the rest of the framework:

- **Prompts** define what an agent can do on a single turn (template + parser + tools + handler).
- **Agents** execute prompts via the agent call loop, managing dialogs and retries.
- **Tactics** define how a *group* of agents collaborates to solve a task.

A tactic accepts `str | BaseModel` as input and returns `str | BaseModel` as output. Subclasses narrow these types for their specific interface.

---

## Defining a Tactic

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

Tactics register themselves automatically through `__init_subclass__` — once the class is imported, it's available to `build_tactic`.

---

## Agent Initialization

`agent_group` lists the agent config keys this tactic needs. At construction time, the tactic reads `agent_group_configs` from the config dict and parses each entry into an `AgentSpec`:

```yaml
# config.yaml
agent_group_configs:
  analyzer:
    model_name: o4-mini-2025-04-16
    system_prompt_path: analytica/analyzer_system
    temperature: 0.1
    max_completion_tokens: 20000
  synthesizer:
    model_name: o4-mini-2025-04-16
    system_prompt_path: analytica/synthesizer_system
    temperature: 0.1
    max_completion_tokens: 20000
```

`AgentSpec` separates config parsing from agent construction. Known keys (`model_name`, `system_prompt_path`, `api_type`) are extracted; everything else becomes `model_args` (temperature, max_tokens, etc.). This means config errors (missing `model_name`, unknown prompt path) surface early with clear messages, before any LLM call is made.

---

## Stateless Execution & Per-Call Isolation

A tactic is stateless across calls. Each `__call__` creates a shallow copy of the tactic with fresh agents:

```python
session = tactic("Analyze this paper")                      # call 1
session2 = tactic("Analyze this other paper")               # call 2 — no interference
```

Internally, `_execute` does:

1. `copy.copy(self)` — shares immutable state (config, runtime, invoker, agent specs).
2. Builds fresh `Agent` instances from specs — each with empty `_dialogs`.
3. Wraps agents in `_TrackedAgent` proxies for transparent session recording.
4. Runs `call()` on the copy.

This means concurrent calls (via `bcall` or threads) never share mutable state. Each call gets its own agents, its own dialogs, its own session.

---

## Transparent Session Tracking

Every `agent.respond()` call inside a tactic is automatically recorded into a `TacticCallSession`. This happens via `_TrackedAgent` — a thin proxy that intercepts `respond()` and records the `AgentCallSession`, then delegates everything else to the real agent.

The developer writes normal Agent API code — tracking is invisible:

```python
def call(self, task: str, **kwargs) -> str:
    analyzer = self.agents["analyzer"]
    analyzer.open("work", prompt_args={"task": task})
    result = analyzer.respond()  # auto-recorded into TacticCallSession
    return result.content
```

To access the full session with costs and diagnostics, use `return_session=True`:

```python
session = tactic("Analyze this paper", return_session=True)

print(session.total_cost)           # aggregated across all agents
print(session.agent_call_count)     # how many agent.respond() calls
print(session.summary())            # human-readable overview
print(session.delivery)             # the return value of call()
```

### TacticCallSession

| Field | Type | Description |
|---|---|---|
| `tactic_name` | `str` | Name of the tactic |
| `state` | `str` | `"initial"`, `"running"`, `"success"`, or `"failure"` |
| `agent_sessions` | `Dict[str, List[AgentCallSession]]` | Per-agent call traces |
| `sub_tactic_sessions` | `Dict[str, List[TacticCallSession]]` | Per-sub-tactic call traces |
| `delivery` | `Any` | The return value of `call()` on success |
| `error` | `str \| None` | Error description on failure |

Key properties:

```python
session.agent_cost          # cost from this tactic's agents only
session.sub_tactic_cost     # cost from sub-tactic calls
session.total_cost          # agent_cost + sub_tactic_cost (recursive)
session.agent_call_count    # number of agent.respond() calls
```

---

## Sub-Tactic Composition

Tactics compose like `nn.Module` child modules. Assign a tactic as an attribute and it's automatically tracked:

```python
class ResearchTactic(Tactic):
    name = "research"
    agent_group = ["planner"]

    def __init__(self, config, ckpt_dir, stream=None, runtime=None):
        super().__init__(config, ckpt_dir, stream, runtime)
        self.analyzer = build_tactic(config, ckpt_dir, stream,
                                      name="analytica", runtime=runtime)
        self.searcher = build_tactic(config, ckpt_dir, stream,
                                      name="searcher", runtime=runtime)

    def call(self, task: str, **kwargs) -> str:
        planner = self.agents["planner"]
        planner.open("plan", prompt_args={"task": task})
        plan = planner.respond()

        # Sub-tactic calls — each gets its own isolated agents
        analysis = self.analyzer(plan.content)
        search = self.searcher(task)

        planner.receive(f"Analysis: {analysis}\nSearch: {search}")
        return planner.respond().content
```

To record sub-tactic sessions into the parent's `TacticCallSession`, pass `return_session=True` and record manually:

```python
sub_session = self.analyzer(plan.content, return_session=True)
self._session.record_sub_tactic_call("analyzer", sub_session)
result = sub_session.delivery
```

Access sub-tactics:

```python
tactic.sub_tactics          # dict of name → child Tactic
```

---

## Tactic Inheritance

Concrete tactics can be subclassed just like any Python class, letting you build reusable pipeline bases and extend them with additional stages.

### Extending a concrete tactic

```python
class WritingPipeline(Tactic):
    """Reusable base: outline → draft."""
    name = "writing_pipeline"
    agent_group = ["outliner", "writer"]

    def call(self, task: str) -> str:
        outliner = self.agents["outliner"]
        writer   = self.agents["writer"]

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
        draft = super().call(task)          # reuse parent's outline→draft logic

        editor = self.agents["editor"]
        editor.open("edit")
        editor.receive(f"Polish this draft:\n\n{draft}")
        return editor.respond().content
```

`super().call(task)` works because LLLM builds agents for **all** names in the subclass's `agent_group` before calling `call()`. The parent's `call()` finds `outliner` and `writer` in `self.agents`; the child adds `editor` on top.

### Abstract base tactics with `register=False`

Use `register=False` to create base classes with shared helpers that should not appear in the registry:

```python
class BasePipelineTactic(Tactic, register=False):
    """Common helpers — not registerable."""

    def _run_stage(self, agent_name: str, dialog: str, message: str) -> str:
        agent = self.agents[agent_name]
        agent.open(dialog)
        agent.receive(message)
        return agent.respond().content


class SummaryPipeline(BasePipelineTactic):
    name = "summary_pipeline"
    agent_group = ["extractor", "writer"]

    def call(self, text: str) -> str:
        facts   = self._run_stage("extractor", "extract", f"Extract key facts:\n{text}")
        summary = self._run_stage("writer",    "write",   f"Summarise:\n{facts}")
        return summary
```

### Typed I/O inheritance

Subclasses can also tighten or widen the I/O types:

```python
class BaseAnalysisTactic(Tactic, register=False):
    agent_group = ["analyzer"]

    def call(self, task: str) -> str:
        agent = self.agents["analyzer"]
        agent.open("analyze")
        agent.receive(task)
        return agent.respond().content


class StructuredAnalysisTactic(BaseAnalysisTactic):
    name = "structured_analysis"

    def call(self, task: str) -> AnalysisOutput:
        raw = super().call(task)
        return AnalysisOutput.model_validate_json(raw)
```

See the working examples at [examples/advanced/multi_agent_tactic.py](../../examples/advanced/multi_agent_tactic.py) and [examples/code_review_service/tactics/code_review.py](../../examples/code_review_service/tactics/code_review.py).

---

## Batch & Concurrent Execution

Tactics provide built-in concurrent execution via thread pools. LLM API calls are I/O-bound, so threads are ideal (the GIL is released during network waits). Each task gets its own isolated agents — no lock contention.

### Synchronous Batch

```python
tasks = ["Analyze paper A", "Analyze paper B", "Analyze paper C"]
results = tactic.bcall(tasks, max_workers=3)
# results[0] corresponds to tasks[0], etc.
```

Returns results in the same order as inputs. Exceptions propagate from the first failed task.

### Async Single

```python
result = await tactic.acall("Analyze this paper")
```

Runs `_execute` in the default thread executor so it doesn't block the event loop.

### Async Concurrent (Fastest-First)

```python
async for idx, result in tactic.ccall(tasks, max_workers=3):
    print(f"Task {idx} finished: {result}")
```

Yields `(index, result)` tuples as tasks complete — not in input order. The index lets you match results to inputs.

All three methods accept `return_sessions=True` to get `TacticCallSession` objects instead of plain results.

---

## Typed I/O with BaseModel

For shareable tactics, define typed inputs and outputs so consumers can inspect the schema:

```python
from pydantic import BaseModel

class AnalysisInput(BaseModel):
    topic: str
    depth: int = 3
    include_sources: bool = True

class AnalysisOutput(BaseModel):
    reasoning: str
    conclusion: str
    confidence: float
    sources: list[str] = []

class Analytica(Tactic):
    name = "analytica"
    agent_group = ["analyzer", "synthesizer"]

    def call(self, task: AnalysisInput, **kwargs) -> AnalysisOutput:
        analyzer = self.agents["analyzer"]
        analyzer.open("work", prompt_args={
            "topic": task.topic,
            "depth": task.depth,
        })
        analysis = analyzer.respond()
        parsed = analysis.parsed

        return AnalysisOutput(
            reasoning=parsed["xml_tags"]["reasoning"][0],
            conclusion=parsed["xml_tags"]["answer"][0],
            confidence=float(parsed["xml_tags"].get("confidence", [0.5])[0]),
        )
```

Simple tactics can just use `str` in and `str` out — both work through the same `__call__` wrapper.

---

## Quick Constructor

For prototyping without config files or discovery:

```python
agent = Tactic.quick("You are a helpful assistant.", model="gpt-4o")
agent.open("chat")
agent.receive("What is the capital of France?")
print(agent.respond().content)
```

This returns a raw `Agent` (not a `Tactic`) — the same object a full tactic would construct internally. No YAML, no TOML, no subclass needed.

---

## Registration & Building

Tactics register automatically when their class is defined (via `__init_subclass__`). To build a tactic from config:

```python
from lllm import build_tactic

config = load_yaml_config("config/experiment.yaml")
tactic = build_tactic(config, ckpt_dir="./runs", name="analytica")
result = tactic("Analyze this paper")
```

Or let the config specify the tactic type:

```yaml
# config/experiment.yaml
tactic_type: analytica

agent_group_configs:
  analyzer:
    model_name: o4-mini-2025-04-16
    system_prompt_path: analytica/analyzer_system
    temperature: 0.1
```

```python
config = load_yaml_config("config/experiment.yaml")
tactic = build_tactic(config, ckpt_dir="./runs")  # reads tactic_type from config
```

---

## Design Notes

- **Tactics are stateless.** Per-call data lives on `TacticCallSession`, not on the tactic. This makes tactics safe for concurrent use and easy to reason about.
- **Agents are per-call.** Each execution builds fresh agents from specs. The expensive objects (invoker, runtime, prompts) are shared; only the mutable `Agent` shell is new.
- **Tracking is transparent.** `_TrackedAgent` proxies intercept `respond()` without changing the Agent API. Developers never need to remember to "record" calls.
- **Thread pool, not multiprocessing.** LLM calls are I/O-bound. Threads share the invoker's HTTP connection pool and avoid pickling issues. The GIL is released during network waits.
- **`Tactic` is an ABC.** `call()` is an abstract method — you must override it. The framework raises `TypeError` at instantiation time if you forget, not at call time.