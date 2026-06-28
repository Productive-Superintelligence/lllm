# Pydantic AI Compatibility

Goal: keep Pydantic AI as the runtime owner while exposing its agent through
the LLLM `Tactic` boundary.

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

Context metadata is forwarded when the agent method accepts `metadata`:

```python
from lllm import CallContext

result = tactic.run(
    {"topic": "refs"},
    context=CallContext(trace_id="trace-1", metadata={"caller": "demo"}),
)
```

Any LLLM tactic can also become a runtime-owned tool:

```python
from lllm.runtimes import tactic_as_tool

tool = tactic_as_tool(tactic, parameter_mode="kwargs")
output = tool(topic="refs")
```

See `examples/pydantic_ai_tactic/structured_agent.py` for a fully offline fake
agent that demonstrates structured input/output, streaming, metadata, and tool
wrapping.
