# 7. Logging and Cost Tracking

**Goal:** see what LLLM logs, read the session every call produces, track tokens
and dollars, persist every tactic call with an observer, tag and query runs,
debug failures from their tracebacks, and save full transcripts.

Every turn in LLLM leaves a structured trace. Each `respond()` produces an
`AgentCallSession` that records every model call, parse repair, tool round,
provider retry, and its cost. Each tactic call produces a `TacticCallSession`
that collects the agent sessions, the sessions of sub-tactics it called, the
result or the error with its traceback, and the total cost. An **observer**
hook on the tactic lets you persist these traces wherever you keep experiment
data.

This matters for research work: runs must be inspectable after the fact, costs
must add up across thousands of calls, and a failure in the middle of a batch
should leave enough behind to debug it.

Where Pydantic fits: sessions, messages, and costs are Pydantic models, so you
can read their fields directly, and `session.summary()` and `Message.to_dict()`
give JSON-ready views.

## Prerequisites

Finish [Lesson 5](05-native-tactics.md). The examples use this tactic:

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


config = {
    "global": {"model_name": "gpt-4o"},
    "agent_configs": [{"name": "writer", "system_prompt": "You are a concise technical writer."}],
}
```

---

## Step 1: Turn on Python logging

LLLM's native runtime logs through Python's standard `logging` module under the
`lllm` logger hierarchy. Configure it like any other library:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

tactic = SummarizerTactic(config)
tactic("LLLM records a session for every tactic call.")
```

```text
INFO lllm.runtimes.native.core.tactic: Tactic 'summarizer' started — session_name=summarizer_<hash>_<timestamp>
INFO lllm.runtimes.native.core.tactic: Tactic 'summarizer' completed — cost=<dollars> agent_calls=1
```

What each level shows:

| Level | Messages |
|---|---|
| `DEBUG` | Every tool call with its arguments, successful agent calls, and the traceback of a failed tactic call |
| `INFO` | Tactic start and completion with cost and agent-call count, and one line per failed call (`Tactic 'summarizer' failed: ValueError: ...`) |
| `WARNING` | Parse repairs (`[writer] Handling exception (retry 0/3): ...`) and provider retries |
| `ERROR` | An agent call that uses up its tool rounds without a final reply (`[writer] Agent call failed`) |

An agent call that runs out of parse repairs, provider retries, or rate-limit
retries logs nothing at `ERROR`: it raises the last error, and the tactic's
INFO failure line reports it.

To see tool calls without turning on debug output for every library:

```python
logging.getLogger("lllm").setLevel(logging.DEBUG)
```

---

## Step 2: Read a tactic call session

`return_session=True` returns the `TacticCallSession` instead of the result:

```python
session = tactic("Sessions record every agent call.", return_session=True)

print(session.tactic_name, session.tactic_path, session.state)
print(session.delivery)           # what call() returned
print(session.agent_call_count)   # respond() calls across all agents
print(session.summary())
```

| Field or property | Meaning |
|---|---|
| `tactic_name` | The tactic's name (a `name=` given to the constructor wins over the class `name`) |
| `tactic_path` | Stable identifier used for filtering (Step 9) |
| `state` | `"success"` or `"failure"` (`"running"` while in flight) |
| `delivery` | The value `call()` returned |
| `error`, `error_traceback` | Exception summary and full traceback on failure |
| `agent_sessions` | Agent name to the list of its `AgentCallSession`s |
| `sub_tactic_sessions` | Tactic name to the sessions of sub-tactics called inside `call()` ([Lesson 5](05-native-tactics.md), Step 7) |
| `total_cost` | `InvokeCost` summed over agent and sub-tactic sessions |
| `agent_call_count`, `sub_tactic_call_count` | Counts |
| `summary()` | Compact dict with tactic, state, counts, and cost |

---

## Step 3: Drill into agent calls

Each `respond()` inside the tactic left an `AgentCallSession`:

```python
for agent_name, calls in session.agent_sessions.items():
    for call in calls:
        print(
            agent_name,
            call.state,
            "repairs:", call.exception_retries_count,
            "tool rounds:", len(call.interrupts),
            "provider retries:", call.llm_recalls_count,
            "cost:", call.cost.cost,
        )
```

The dictionaries are keyed by the loop step (one step per tool round):

- `exception_retries[step]`: the parse or validation errors that triggered
  repairs;
- `interrupts[step]`: the tool calls requested in that round;
- `llm_recalls[step]`: provider errors that were retried;
- `invoke_traces[step]`: one `InvokeResult` per model call, including failed
  attempts, with the exact `model_args` sent, any `execution_errors`, the
  returned `message`, and the provider's `raw_response`.

```python
call = session.agent_sessions["writer"][0]
for step, traces in call.invoke_traces.items():
    for trace in traces:
        print(step, trace.model_args, trace.has_errors, trace.message.content[:60])
```

`call.delivery` is the accepted reply, the same `Message` `respond()` returned.

---

## Step 4: Track tokens and cost

Costs are `InvokeCost` objects at three levels:

```python
reply = call.delivery
print(reply.cost)                  # one message
print(session.total_cost)          # one tactic call
print(session.total_cost.cost)     # total dollars as a float
print(session.total_cost.prompt_tokens, session.total_cost.completion_tokens)
```

Printing an `InvokeCost` gives a readable breakdown:

```text
Tokens: <total> (Prompt: <n> [Cached: <n>], Completion: <n> [Reasoning: <n>])
Cost Breakdown: Prompt: $<x>, Completion: $<y> | Total Cost: $<z>
```

| Field | Meaning |
|---|---|
| `prompt_tokens`, `completion_tokens`, `total_tokens` | Token counts reported by the provider |
| `cached_prompt_tokens`, `reasoning_tokens` | Cached-prompt and reasoning token counts, when reported |
| `prompt_cost`, `completion_cost`, `cost` | Dollar amounts |
| `input_cost_per_token`, `output_cost_per_token` | Per-token rates recorded on individual messages; zero in sums |

Dollar amounts come from LiteLLM's model price list. For a model LiteLLM has no
price for, token counts are still recorded and dollar amounts are `0.0`. A
dialog sums its own messages with `dialog.cost`
([Lesson 2](02-agents-and-dialogs.md)).

---

## Step 5: Persist every call with an observer

Pass an `observer` when you construct a tactic. LLLM calls its hooks for every
tactic call, passing the session, the task, and the tags and metadata of that
call:

- `on_tactic_success(*, tactic, session, task, result, span, tags, metadata)`
- `on_tactic_failure(*, tactic, session, task, exception, span, tags, metadata)`
- optionally `observe_tactic(*, tactic, session, task, session_name, tags, metadata)`,
  a context manager wrapped around the call; whatever it yields is passed to
  the other hooks as `span`. Use it for timing or tracing spans.

Every hook is optional, and an exception inside a hook is logged without
breaking the call. This observer appends one JSON line per call:

```python
import contextlib
import json
import time


class JsonlRunLog:
    """Append one JSON record per tactic call."""

    def __init__(self, path: str):
        self.path = path

    @contextlib.contextmanager
    def observe_tactic(self, *, tactic, session, task, session_name, tags, metadata):
        yield {"session_name": session_name, "started": time.time()}

    def on_tactic_success(self, *, tactic, session, task, result, span, tags, metadata):
        self._write(session, span, tags)

    def on_tactic_failure(self, *, tactic, session, task, exception, span, tags, metadata):
        self._write(session, span, tags)

    def _write(self, session, span, tags):
        cost = session.total_cost
        record = {
            "session_name": span["session_name"],
            "seconds": round(time.time() - span["started"], 3),
            "tactic": session.tactic_path,
            "state": session.state,
            "tags": tags or {},
            "agent_calls": session.agent_call_count,
            "cost_usd": cost.cost,
            "prompt_tokens": cost.prompt_tokens,
            "completion_tokens": cost.completion_tokens,
            "error": session.error,
            "traceback": session.error_traceback,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


run_log = JsonlRunLog("runs.jsonl")
tactic = SummarizerTactic(config, observer=run_log)
```

---

## Step 6: Tag runs and query them

Tags are string key-value pairs attached at call time. They reach the observer,
so you can filter on them later:

```python
tactic("First document ...", tags={"env": "dev", "experiment": "prompt-b"})
tactic.bcall(["Second document ...", "Third document ..."], tags={"env": "dev", "experiment": "prompt-b"})
```

`metadata={...}` works the same way for values that are not strings. Querying
the log is ordinary Python:

```python
with open("runs.jsonl", encoding="utf-8") as f:
    runs = [json.loads(line) for line in f]

prompt_b = [run for run in runs if run["tags"].get("experiment") == "prompt-b"]
print(f"{len(prompt_b)} runs, ${sum(run['cost_usd'] for run in prompt_b):.4f} total")

by_tactic = {}
for run in runs:
    by_tactic[run["tactic"]] = by_tactic.get(run["tactic"], 0.0) + run["cost_usd"]
print(by_tactic)
```

The same observer can write to SQLite, a data warehouse, or an experiment
tracker instead of a JSONL file; only `_write()` changes.

---

## Step 7: Debug failed calls

A failing call raises as usual, and the observer still receives the session,
with `state="failure"`, the exception summary, and the full traceback:

```python
from lllm import Tactic


class FlakyTactic(Tactic):
    name = "flaky"

    def call(self, task: str) -> str:
        raise ValueError(f"cannot handle {task!r}")


try:
    FlakyTactic(observer=run_log)("bad input", tags={"env": "dev"})
except ValueError:
    pass

with open("runs.jsonl", encoding="utf-8") as f:
    failures = [run for run in map(json.loads, f) if run["state"] == "failure"]

print(failures[-1]["error"])       # ValueError: cannot handle 'bad input'
print(failures[-1]["traceback"])
```

The `session` passed to `on_tactic_failure` also holds the agent sessions
recorded before the failure, so you can see how far the call got.

---

## Step 8: Save full transcripts

Sessions keep the accepted replies; the complete dialogs live on the agents of
each call. The `tactic` passed to the observer hooks is the per-call copy, so
its agents still hold every dialog they opened:

```python
class TranscriptLog(JsonlRunLog):
    """Also save every dialog of every agent, one JSON file per call."""

    def on_tactic_success(self, *, tactic, session, task, result, span, tags, metadata):
        super().on_tactic_success(
            tactic=tactic, session=session, task=task, result=result,
            span=span, tags=tags, metadata=metadata,
        )
        transcripts = {
            agent_name: {alias: dialog.to_dict() for alias, dialog in agent.dialogs.items()}
            for agent_name, agent in tactic.agents.items()
        }
        with open(f"{span['session_name']}.dialogs.json", "w", encoding="utf-8") as f:
            json.dump(transcripts, f)


tactic = SummarizerTactic(config, observer=TranscriptLog("runs.jsonl"))
tactic("Save the whole conversation.")
```

`dialog.to_dict()` produces JSON-ready data with the messages (including tool
calls and parsed output), the dialog's lineage, and its latest prompt. Load a
saved dialog with `Dialog.from_dict(data)` to review it, replay it, or build
datasets from it ([Lesson 2](02-agents-and-dialogs.md), Step 9).

---

## Step 9: Use stable tactic identities

`session.tactic_path` identifies the tactic in logs. When a tactic is built from
a package registry it has the form `"<package>::<tactic name>"`:

```python
from lllm import Runtime, build_tactic

runtime = Runtime()
runtime.register_tactic("summarizer", SummarizerTactic, namespace="my_project.tactics")

packaged = build_tactic(config, name="my_project.tactics:summarizer", registry=runtime)
print(packaged("Stable ids.", return_session=True).tactic_path)   # my_project::summarizer
```

Loading an LLLM package registers its tactics this way
([Lesson 6](06-packages-and-hub.md)). The identifier does not depend on file
paths inside the package, so log queries keep working when you reorganize a
project; only renaming the package or the tactic's `name` changes it. A tactic
constructed directly uses its `name`.

---

## Summary

| Task | API |
|---|---|
| See runtime logs | `logging.basicConfig(level=logging.INFO)`; `logging.getLogger("lllm")` |
| Session of one call | `tactic(task, return_session=True)` |
| Per-respond details | `session.agent_sessions[name][i]`: retries, interrupts, invoke traces |
| Sub-tactic sessions | `session.sub_tactic_sessions[name]` |
| Cost | `message.cost`, `dialog.cost`, `session.total_cost` |
| Persist every call | `MyTactic(config, observer=obj)` with `on_tactic_success` / `on_tactic_failure` |
| Timing or tracing spans | `observe_tactic(...)` context manager on the observer |
| Tag runs | `tactic(task, tags={...}, metadata={...})`, `tactic.bcall(tasks, tags={...})` |
| Debug failures | `session.error`, `session.error_traceback` in `on_tactic_failure` |
| Save transcripts | `dialog.to_dict()` for each agent of the per-call `tactic` |
| Stable identity | `session.tactic_path`, `"<package>::<name>"` for packaged tactics |

See the [Logging reference](../core/logging.md) for details.

**Next:** [8. Advanced Patterns](08-advanced-patterns.md)
