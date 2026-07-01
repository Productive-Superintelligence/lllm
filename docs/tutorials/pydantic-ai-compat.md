# Pydantic AI Compatibility

Goal: understand the compatibility rule and then move to the current
[Pydantic AI Runtime](pydantic-ai-runtime.md) tutorial.

This page is kept for older links. The current tutorial is
[Pydantic AI Runtime](pydantic-ai-runtime.md).

The compatibility rule remains the same: Pydantic AI owns agent execution,
tools, provider settings, streaming, tracing, eval hooks, graphs, and durable
workflow behavior. LLLM wraps the agent as a typed `Tactic` for service and
package boundaries.

## Prerequisites

```bash
python -m pip install -e ".[dev]"
```

## Files Used

```text
examples/pydantic_ai_tactic/
  structured_agent.py
  surrounding_features.py
tests/
  test_pydantic_ai_adapter.py
  test_examples.py
```

## Compatibility Wrapper

```python
from lllm.runtimes import PydanticAITactic

tactic = PydanticAITactic(
    agent,
    input_type=BriefInput,
    output_type=BriefOutput,
)
```

## Verify

```bash
python -m pytest tests/test_pydantic_ai_adapter.py tests/test_examples.py -q
```

Expected output:

```text
... passed
```

Next, follow [Pydantic AI Runtime](pydantic-ai-runtime.md) for the full
step-by-step flow.
