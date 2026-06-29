# Pydantic AI Compatibility

Goal: keep Pydantic AI as the runtime owner while exposing its agent through
the LLLM `Tactic` boundary.

## Prerequisites

```bash
python -m pip install -e ".[dev]"
```

## Files Used

```text
examples/pydantic_ai_tactic/
  fake_agent.py
  structured_agent.py
  surrounding_features.py
tests/
  test_pydantic_ai_adapter.py
  test_examples.py
```

Executable offline examples live in
`examples/pydantic_ai_tactic/structured_agent.py` and
`examples/pydantic_ai_tactic/surrounding_features.py`.

## Wrap

```python
from pydantic import BaseModel
from lllm.runtimes import PydanticAITactic


class BriefInput(BaseModel):
    topic: str


class BriefOutput(BaseModel):
    title: str


tactic = PydanticAITactic(
    agent,
    input_type=BriefInput,
    output_type=BriefOutput,
)
```

When a Pydantic model reaches the adapter, the default `input_mode="auto"`
sends JSON to the agent. Use `input_mode="dict"` or `input_mode="python"` when
your agent expects those shapes instead.

## Metadata

Context metadata is forwarded when the agent method accepts `metadata`:

```python
from lllm import CallContext

result = tactic.run(
    {"topic": "refs"},
    context=CallContext(trace_id="trace-1", metadata={"caller": "demo"}),
)
```

Runtime-owned features stay runtime-owned. Configure model/provider settings,
instrumentation, eval hooks, durable execution IDs, graph/workflow state, tool
approval, and dependencies on the agent or pass them as normal run kwargs:

The offline surrounding-features example models Logfire/OpenTelemetry-style
instrumentation as agent-owned state and asserts the LLLM wrapper leaves that
state intact.

## Runtime Features

```python
tactic = PydanticAITactic(
    agent,
    input_type=BriefInput,
    output_type=BriefOutput,
    run_kwargs={
        "model_settings": {"temperature": 0},
        "eval_hook": "offline-score",
    },
)

result = tactic.run(
    {"topic": "refs"},
    durable_run_id="run-1",
    graph_node="planner.step",
)
```

If you pass `metadata=` yourself, LLLM does not overwrite it with context
metadata.

Any LLLM tactic can also become a runtime-owned tool:

## Tool Wrapper

```python
from lllm.runtimes import tactic_as_tool

tool = tactic_as_tool(tactic, parameter_mode="kwargs")
output = tool(topic="refs")
```

See `examples/pydantic_ai_tactic/structured_agent.py` and
`examples/pydantic_ai_tactic/surrounding_features.py` for fully offline fake
agents that demonstrate structured input/output, streaming, metadata, tool
wrapping, and runtime-owned surrounding features.

## Verify

```bash
python -m pytest tests/test_pydantic_ai_adapter.py tests/test_examples.py -q
```

Expected output:

```text
... passed
```

Next, serve the wrapped agent with `create_tactic_app()` or export tactic
metadata for a PsiHub package.
