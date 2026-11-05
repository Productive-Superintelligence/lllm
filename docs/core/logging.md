# Logging & Sessions

The native runtime gives you two orthogonal systems:

- **Session records.** Every tactic call produces a `TacticCallSession` holding
  every agent call, every model invocation, every repair and tool round, and the
  cost. Dialogs keep the full message trees. Attach an *observer* to persist
  them wherever you like.
- **Terminal logging.** Internal diagnostics go through Python's standard
  `logging` module under the `lllm.runtimes.native` logger hierarchy.

---

## Session records

One tactic call is the unit of record. From the top down:

```
TacticCallSession                 one tactic(task) call
├── delivery / error / error_traceback / state
├── agent_sessions[agent_name] -> [AgentCallSession, ...]      one per respond()
│   ├── delivery                  the final Message
│   ├── exception_retries         parse failures per loop step
│   ├── interrupts                tool calls per loop step
│   ├── llm_recalls               provider errors per loop step
│   └── invoke_traces[step] -> [InvokeResult, ...]
│       ├── raw_response          the provider's response object
│       ├── model_args            what was actually sent
│       ├── execution_errors
│       └── message               the Message produced
└── sub_tactic_sessions[name] -> [TacticCallSession, ...]   tactics called during this one
```

Alongside the session, each agent's dialogs (`agent.dialogs`) hold the message
trees, including the working forks the loop created for repairs; see
[Dialog](dialog.md). Together they contain enough to reconstruct what every
agent saw and said on every turn, with the exact model arguments and costs.

```python
session = tactic("Analyze this paper", return_session=True)

session.state                  # "success" | "failure"
session.total_cost             # InvokeCost (tokens and dollars), sub-tactics included
for agent_name, calls in session.agent_sessions.items():
    for call in calls:
        print(agent_name, call.state, call.cost.cost, call.exception_retries_count)
        print(call.delivery.content)
```

On failure the exception is re-raised; to inspect the failed session, persist it
from an observer (below), which receives it with `state="failure"`, `error`, and
`error_traceback` filled in.

---

## Tactic identity: the stable `pkg::name` key

Every session records which tactic produced it in `session.tactic_path`, in the
form:

```
{package_name}::{tactic_name}
```

for example `my_pkg::researcher`.

A runtime path such as `my_pkg.tactics:v2/researcher` encodes where a tactic
lives inside a package, including folders and `under` prefixes. That is useful
for lookup but a poor storage key, because restructuring would orphan old
records. The `pkg::name` form only uses the package name and `Tactic.name`:

| Change | `tactic_path` |
| --- | --- |
| Move the tactic file to another folder | Stable |
| Add or change an `under` prefix | Stable |
| Rename `[package] name` | Changes |
| Rename `Tactic.name` | Changes |

The last two are semantic changes and should be deliberate. Treat
`[package] name` as a permanent identity, like a Python distribution name.

How it is set:

- `build_tactic(config, name=...)` looks the tactic up and records
  `"<package>::<name>"`, where the package is the first segment of the namespace
  it was found in. Building through an alias (`build_tactic(cfg, name="p1:researcher")`)
  records the alias, so build through the canonical package name when you key
  records on `tactic_path`.
- A tactic constructed directly with a registry looks up its own `name` in that
  registry's default namespace and falls back to the bare `Tactic.name`.
- Pass `tactic_path="my_pkg::researcher"` (or `TacticContext(tactic_path=...)`)
  to set it explicitly.

---

## Persisting sessions with an observer

A tactic accepts an `observer`: any object with some of these methods. All are
optional and called with keyword arguments.

| Method | When | Arguments |
| --- | --- | --- |
| `observe_tactic(...)` | Before `call()` runs. Return a context manager (or `None`); the value it yields is passed to the callbacks as `span`. | `tactic`, `session`, `task`, `session_name`, `tags`, `metadata` |
| `on_tactic_success(...)` | After `call()` returns and the output validates. | `tactic`, `session`, `task`, `result`, `span`, `tags`, `metadata` |
| `on_tactic_failure(...)` | When `call()` (or validation) raises. | `tactic`, `session`, `task`, `exception`, `span`, `tags`, `metadata` |

`tactic` is the per-call copy, so `tactic.agents` holds this call's agents and
their dialogs. An exception inside an observer is logged as a warning and never
breaks the tactic call.

### Example: a JSON Lines session log

```python
import json
import time
from contextlib import contextmanager
from pathlib import Path


class JsonlSessionLog:
    """Append one JSON record per tactic call."""

    def __init__(self, path: str):
        self.path = Path(path)

    @contextmanager
    def observe_tactic(self, *, tactic, session, task, session_name, tags, metadata):
        yield {"session_name": session_name, "started": time.time()}

    def on_tactic_success(self, *, tactic, session, task, result, span, tags, metadata):
        self._write(tactic, session, span, tags, metadata)

    def on_tactic_failure(self, *, tactic, session, task, exception, span, tags, metadata):
        self._write(tactic, session, span, tags, metadata)

    def _write(self, tactic, session, span, tags, metadata):
        dialogs = {}
        for agent_name, agent in getattr(tactic, "agents", {}).items():
            for alias, dialog in agent.dialogs.items():
                dialogs[f"{agent_name}/{alias}"] = dialog.to_dict()   # JSON-safe
        record = {
            "session_name": span["session_name"],
            "tactic": session.tactic_path,
            "state": session.state,
            "tags": tags or {},
            "metadata": metadata or {},
            "cost_usd": session.total_cost.cost,
            "agent_calls": session.agent_call_count,
            "latency_s": round(time.time() - span["started"], 3),
            "error": session.error,
            "traceback": session.error_traceback,
            "delivery": session.delivery if session.state == "success" else None,
            "dialogs": dialogs,
        }
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
```

Attach it at construction, or through `build_tactic`, which forwards keyword
arguments to the constructor:

```python
log = JsonlSessionLog("runs/sessions.jsonl")

tactic = MyTactic(config, observer=log)
tactic = build_tactic(resolve_config("default"), observer=log)
```

Every call is now recorded, with the tags and metadata you pass:

```python
result = tactic(
    task,
    tags={"experiment": "exp-001", "dataset": "arxiv-2024", "split": "val"},
    metadata={"user": "alice", "git_sha": "abc1234"},
)
```

Reading it back is ordinary Python:

```python
import json

records = [json.loads(line) for line in open("runs/sessions.jsonl")]

failures = [r for r in records if r["state"] == "failure"]
exp = [r for r in records if r["tags"].get("experiment") == "exp-001"]
total = sum(r["cost_usd"] for r in exp)
print(f"{len(exp)} sessions, ${total:.4f}")
for r in failures:
    print(r["tactic"], r["error"])
    print(r["traceback"])
```

Swap the file for SQLite, a document store, or an experiment tracker by changing
`_write`.

### Example: OpenTelemetry spans

For production tracing, have `observe_tactic` return a span:

```python
from opentelemetry import trace

tracer = trace.get_tracer("my-app")


class OTelObserver:
    def observe_tactic(self, *, tactic, session, task, session_name, tags, metadata):
        attributes = {"lllm.session_name": session_name}
        attributes.update({f"lllm.tag.{k}": v for k, v in (tags or {}).items()})
        return tracer.start_as_current_span(f"tactic {session.tactic_name}", attributes=attributes)

    def on_tactic_success(self, *, span, session, **_):
        span.set_attribute("lllm.agent_calls", session.agent_call_count)
        span.set_attribute("lllm.cost_usd", session.total_cost.cost)

    def on_tactic_failure(self, *, span, exception, **_):
        span.record_exception(exception)
```

When a tactic is called through the protocol (`run()`, `arun()`, or a FastAPI
service), the per-call copy passed to the observer carries the request's
`CallContext` as `tactic.call_context` (request id, trace id, refs, tags,
metadata), so service traces and native sessions line up:

```python
class TraceIdObserver:
    def on_tactic_success(self, *, tactic, session, **_):
        ctx = tactic.call_context                  # None for plain tactic(task) calls
        if ctx is not None:
            print(session.tactic_path, ctx.request_id, ctx.trace_id, ctx.tags)
```

---

## Tags and metadata

`tags` are short `str -> str` values meant for filtering; `metadata` is richer
context you want to keep but not filter on. Both are passed through to the
observer untouched.

### Recommended vocabulary

```python
# Experiment / run tracking
tags = {"experiment": "ablation-001", "run": "run-003", "variant": "no-retrieval"}

# Dataset splits
tags = {"dataset": "pubmed-2024", "split": "test", "subset": "cardiology"}

# Multi-stage pipelines
tags = {"pipeline": "document-qa", "stage": "extraction", "doc_id": "doc-00127"}

# Production / CI
tags = {"env": "prod", "version": "v2.3.1", "requester": "user-alice"}
```

### Composing tags

Build tags incrementally so every level can be filtered on:

```python
BASE = {"project": "doc-qa", "model": "gpt-4o"}

for experiment in ["baseline", "rag-v1", "rag-v2"]:
    for doc_id in document_ids:
        tactic(make_task(doc_id), tags={**BASE, "experiment": experiment, "doc_id": doc_id})
```

### Tags vs. metadata

```python
tactic(
    task,
    tags={"experiment": "exp-001", "split": "val"},
    metadata={"git_sha": "abc1234", "input_file": "/data/val.jsonl", "config": config},
)
```

`bcall`, `acall`, and `ccall` accept `tags` and `metadata` too and apply them to
every task.

---

## Terminal logging

Every native module logs through `logging.getLogger(__name__)`, so all loggers
live under `lllm.runtimes.native`:

| Logger | Covers |
| --- | --- |
| `lllm.runtimes.native` | Everything below |
| `lllm.runtimes.native.core.tactic` | Tactic start and completion with cost (INFO), failures with traceback (ERROR), observer errors |
| `lllm.runtimes.native.core.prompt` | Repair retries and LLM recalls (WARNING), tool calls and their arguments (DEBUG), agent-call failure (ERROR) |
| `lllm.runtimes.native.core.agent` | Loop configuration warnings |
| `lllm.runtimes.native.core.native_config` | Unknown agent config keys, missing or unreachable skills |
| `lllm.runtimes.native.core.config` | Discovery registration problems |
| `lllm.runtimes.native.core.runtime` | Package install, export, removal; resource overwrites (DEBUG) |
| `lllm.runtimes.native.core.dialog` | Context truncation (DEBUG), unresolved prompts when restoring dialogs |
| `lllm.runtimes.native.core.resource` | Lazy resources that fail to load |
| `lllm.runtimes.native.invokers.litellm` | Provider environment checks |
| `lllm.runtimes.native.proxies.base` | Proxy registration |
| `lllm.runtimes.native.sandbox.jupyter` | Notebook server and kernel lifecycle |
| `lllm.runtimes.native.tools.cua` | Computer-use agent events |

Package discovery problems (a missing folder, a module that fails to import, a
dependency without `lllm.toml`) are reported with `warnings.warn(...,
RuntimeWarning)` rather than logging, so they show up even without logging
configured.

### Quick setup

```python
import logging

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logging.getLogger("lllm.runtimes.native").setLevel(logging.INFO)
```

### Level guide

| Level | What you see |
| --- | --- |
| `WARNING` | Repair retries with the parse error, LLM recalls, unknown config keys, missing skills |
| `INFO` | Plus tactic start and completion with cost and agent-call counts, package installs |
| `DEBUG` | Plus every tool call with arguments, successful agent calls, context truncation, resource overwrites |

### Fine-grained control

```python
import logging

logging.getLogger("lllm.runtimes.native").setLevel(logging.WARNING)
logging.getLogger("lllm.runtimes.native.core.tactic").setLevel(logging.INFO)   # just tactic summaries
logging.getLogger("lllm.runtimes.native.core.prompt").setLevel(logging.ERROR)  # mute retry noise

handler = logging.FileHandler("lllm.log")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.getLogger("lllm.runtimes.native").addHandler(handler)
```

Silence everything with
`logging.getLogger("lllm.runtimes.native").setLevel(logging.CRITICAL + 1)`.
