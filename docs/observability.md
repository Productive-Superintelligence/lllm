# Observability

LLLM records every run as plain Python data. Each tactic call produces a
`TacticCallSession` holding every agent call, each model call inside it,
every repair and tool round, and the cost; each agent keeps its dialogs as
full transcripts. Nothing is sent anywhere by default. You read the records
in your own code, and an observer hook saves them wherever you keep runs.

Ask a tactic for its session with `return_session=True`:

```python title="triage.py"
from pydantic import BaseModel

from lllm import Prompt, Tactic, tool


@tool(description="Look up a customer's support plan")
def support_plan(customer: str) -> str:
    return {"acme": "enterprise"}.get(customer.lower(), "free")


class Ticket(BaseModel):
    priority: str
    reason: str


TRIAGE = Prompt(
    path="support/triage",
    prompt="Triage this ticket: {ticket}",
    format=Ticket,
    function_list=[support_plan],
)


class Triage(Tactic):
    name = "triage"
    agent_group = ["triager"]
    output_model = Ticket

    def call(self, ticket: str) -> Ticket:
        triager = self.agents["triager"]
        triager.open("ticket")
        triager.receive_prompt(TRIAGE, {"ticket": ticket})
        return triager.respond().parsed


triage = Triage({
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [
        {"name": "triager", "system_prompt": "You triage tickets."}
    ],
})

session = triage("ACME: our dashboard is down.", return_session=True)
print(session.state, session.delivery)
#> success priority='high' reason='Enterprise customer with an outage.'
call = session.agent_sessions["triager"][0]  # (1)!
for step, traces in call.invoke_traces.items():  # (2)!
    for trace in traces:
        print(step, trace.message.role.value, trace.has_errors)
        #> 0 tool_call False
        #> 1 assistant True
        #> 1 assistant False
print([c.name for c in call.tool_calls], call.exception_retries_count)
#> ['support_plan'] 1
print(session.total_cost)  # (3)!
"""
Tokens: 90 (Prompt: 30 [Cached: 0], Completion: 60 [Reasoning: 0])
Cost Breakdown: Prompt: $0.000005, Completion: $0.000036 | Total Cost: $0.000040
"""
```

1. One `AgentCallSession` per `respond()`, listed under the agent's name.
2. Every model call, grouped by loop step. In step 0 the model asked for
   the tool. In step 1 its first reply failed the `Ticket` format, the agent
   sent the error back, and the second reply parsed.
3. Tokens and dollars for the whole call, summed over all three model calls.

## What a session holds

```text
TacticCallSession                  one tactic call
├── state, delivery, error, error_traceback
├── agent_sessions[agent]          one AgentCallSession per respond()
│   ├── state, delivery            the final Message
│   ├── invoke_traces[step]        one InvokeResult per model call
│   ├── exception_retries[step]    parse errors that caused repairs
│   ├── interrupts[step]           tool calls (tool_calls lists them all)
│   ├── llm_recalls[step]          provider errors that were retried
│   └── cost
└── sub_tactic_sessions[tactic]    sessions of the tactics it called
```

An `InvokeResult` keeps the `message` the model returned, the exact
`model_args` sent, any `execution_errors` (such as the parse error above),
and the provider's `raw_response`. Tactics called inside a tactic record
their sessions into the caller's, so the cost of a whole program adds up in
one place; [Tactics](tactics.md#sessions) lists the session's fields. A
plain agent outside a tactic has no tactic session, but
`agent.respond(return_session=True)` returns the same `AgentCallSession`.

## Cost and token usage

Costs are `InvokeCost` objects at every level: `message.cost` for one model
call, `call.cost` for one `respond()`, `dialog.cost` for the replies a
dialog kept, and `session.total_cost` for a tactic call with its
sub-tactics. They add with `+`.

| Field | Meaning |
| --- | --- |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | Token counts the provider reported |
| `cached_prompt_tokens`, `reasoning_tokens` | Cached prompt and reasoning tokens, when reported |
| `prompt_cost`, `completion_cost`, `cost` | US dollars |
| `input_cost_per_token`, `output_cost_per_token` | The rates, on a single message only |

Dollar amounts come from LiteLLM's model price list. For a model it has no
price for, such as one on a local server, tokens are still counted and the
dollar fields are `0.0`.

## Saving runs with an observer

The session holds each agent call's replies; the dialogs hold everything
each agent saw and said. Every tactic call gets fresh agents, so the place
to read both is an observer: any object with some of these methods, passed
as `observer=` when you build the tactic.

| Method | Called | Receives |
| --- | --- | --- |
| `observe_tactic` | before the call; may return a context manager | `tactic`, `session`, `task`, `session_name`, `tags`, `metadata` |
| `on_tactic_success` | after the result validates | `tactic`, `session`, `task`, `result`, `span`, `tags`, `metadata` |
| `on_tactic_failure` | when the call raises | `tactic`, `session`, `task`, `exception`, `span`, `tags`, `metadata` |

Arguments come by keyword. `tactic` is the call's own copy, so
`tactic.agents` holds this call's agents and their dialogs, and whatever
`observe_tactic`'s context manager yields arrives as `span`. An exception
inside an observer is logged and never breaks the call. This observer
appends one JSON line per call, failed calls included, with the full
transcripts:

```python title="run_log.py"
import json
from pathlib import Path

from lllm import Dialog, Tactic


class RunLog:
    def __init__(self, path):
        self.path = Path(path)

    def on_tactic_success(self, *, tactic, session, tags, **_):
        self.write(tactic, session, tags)

    def on_tactic_failure(self, *, tactic, session, tags, **_):
        self.write(tactic, session, tags)

    def write(self, tactic, session, tags):
        dialogs = {
            f"{name}/{alias}": dialog.to_dict()  # (1)!
            for name, agent in tactic.agents.items()
            for alias, dialog in agent.dialogs.items()
        }
        run = {
            "tactic": session.tactic_path,
            "state": session.state,
            "tags": tags or {},
            "cost": session.total_cost.cost,
            "error": session.error,
            "dialogs": dialogs,
        }
        with self.path.open("a") as file:
            file.write(json.dumps(run) + "\n")


class Summarize(Tactic):
    name = "summarize"
    agent_group = ["writer"]

    def call(self, text: str) -> str:
        if not text.strip():
            raise ValueError("nothing to summarize")
        writer = self.agents["writer"]
        writer.open("summary")
        writer.receive(f"Summarize in five words: {text}")
        return writer.respond().content


summarize = Summarize(
    {
        "global": {"model_name": "gpt-4o-mini"},
        "agent_configs": [
            {"name": "writer", "system_prompt": "You summarize."}
        ],
    },
    observer=RunLog("runs.jsonl"),
)
summarize(
    "Sessions record every agent call and its cost.", tags={"exp": "a"}
)
summarize(
    "Observers save sessions wherever you keep runs.", tags={"exp": "b"}
)
try:
    summarize("   ", tags={"exp": "c"})
except ValueError:
    pass  # (2)!

runs = [json.loads(line) for line in open("runs.jsonl")]
print([(run["tags"]["exp"], run["state"], run["error"]) for run in runs])
"""
[
    ('a', 'success', None),
    ('b', 'success', None),
    ('c', 'failure', 'ValueError: nothing to summarize'),
]
"""
dialog = Dialog.from_dict(runs[1]["dialogs"]["writer/summary"])  # (3)!
print(dialog.overview())
"""
[0. system (developer)]: You summarize.

[1. user (user)]: Summarize in five words: Observers save sessions wherever you keep runs.

[2. writer (assistant)]: Observers save every run.
"""
```

1. `to_dict()` is JSON-ready: every message with its tool calls, parsed
   output, and usage, plus the dialog's lineage. See
   [Dialogs](dialogs.md#saving-and-loading).
2. A failing call raises as usual, and the observer still gets its session,
   with `state="failure"`, the `error`, the full `error_traceback`, and the
   agent calls that finished before the failure.
3. A saved dialog loads back as a `Dialog`, to review, fork, or replay.

`tags` and `metadata` are passed per call, as in `tactic(task, tags={...},
metadata={...})`, and reach the observer untouched; `bcall`, `acall`, and
`ccall` take them too. Use tags for short strings you filter on, such as an
experiment or a dataset split, and metadata for the rest. To write to
SQLite, an experiment tracker, or a tracing backend, change `write`.

## Logging

LLLM logs through Python's `logging` module, under the `lllm` logger. It
adds no handlers, so configure it as you would for any library:

```python title="logging_setup.py"
import logging

logging.basicConfig(format="%(levelname)s %(message)s")
logging.getLogger("lllm").setLevel(logging.INFO)
```

Each tactic call then logs a line when it starts and one when it ends. Here
is `summarize(text, session_name="run-1")` from the example above:

```text
INFO Tactic 'summarize' started — session_name=run-1
INFO Tactic 'summarize' completed — cost=$0.000013 agent_calls=1
```

| Level | Adds |
| --- | --- |
| `WARNING` | Repairs, with the parse error (`[writer] Handling exception (retry 0/3): ...`); provider retries; unknown config keys |
| `INFO` | Tactic start and completion with cost and agent-call count; one line per failed call |
| `DEBUG` | Every tool call with its arguments; the traceback of a failed call |

Without `session_name=`, a call is named `<tactic>_<task hash>_<time>`. An
agent call that uses up its tool rounds logs `[name] Agent call failed` at
`ERROR`. The loggers `lllm.runtimes.native.core.tactic` (tactic calls) and
`lllm.runtimes.native.core.prompt` (the agent loop) can be set separately.
Problems found while discovering a package are Python warnings, so they
show even without logging set up.

## Records at the tactic boundary

Every tactic also has the typed boundary used for [serving](serving.md):
`run()` and `arun()`. With `return_trace=True` they return a `CallResult`
whose `trace` holds the request id, state, timings, and the error.
`proxy_tactic()` wraps any tactic and sends a `ProxyRecord` for each call to
a sink of your choice:

```python title="boundary.py"
from lllm import CallContext, InMemoryProxyLog, Tactic, proxy_tactic


class Shout(Tactic):
    name = "shout"

    def call(self, text: str) -> str:
        return text.upper()


result = Shout().run(
    "hi", context=CallContext(request_id="req-1"), return_trace=True
)
print(result.output, result.trace.request_id, result.trace.state)
#> HI req-1 success

log = InMemoryProxyLog()
shout = proxy_tactic(Shout(), sink=log.append, capture_inputs=True)
shout.run("hello", context=CallContext(request_id="req-2"))
record = log.records[0]
print(record.request_id, record.tactic, record.state, record.input_value)
#> req-2 shout success hello
```

`proxy_tactic()` also takes `before`, `after`, and `on_error` hooks, called
with the value (or the exception) and the `CallContext`; what `before` or
`after` returns replaces the input or output, and `None` keeps it. When a
served request reaches a tactic, the observer's `tactic.call_context` is
that request's `CallContext`, so native sessions line up with service
traces.

For OpenTelemetry or another tracer, return a span from `observe_tactic`;
the span arrives in the other two methods:

```python test="skip" reason="needs opentelemetry"
from opentelemetry import trace

tracer = trace.get_tracer("my-app")


class Tracing:
    def observe_tactic(self, *, session, **_):
        return tracer.start_as_current_span(f"tactic {session.tactic_name}")

    def on_tactic_success(self, *, span, session, **_):
        span.set_attribute("lllm.cost_usd", session.total_cost.cost)

    def on_tactic_failure(self, *, span, exception, **_):
        span.record_exception(exception)
```

## Stable names for runs

`session.tactic_path` names the tactic in your records. A tactic built from
a [package](packages.md) with `build_tactic` gets `"<package>::<name>"`,
such as `"my_pkg::researcher"`, which stays the same when you move files
inside the package. A tactic you construct yourself uses its `name`, unless
you pass `tactic_path=`.
