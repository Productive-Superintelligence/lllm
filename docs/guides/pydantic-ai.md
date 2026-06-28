# Pydantic AI

Pydantic AI is the first-class agentic runtime target.

Configure the agent normally, then wrap it:

```python
from lllm.runtimes import PydanticAITactic

tactic = PydanticAITactic(
    agent,
    input_type=BriefInput,
    output_type=BriefOutput,
)
```

The adapter preserves runtime ownership. LLLM forwards input, context metadata,
run kwargs, streaming modes, and output schemas where supported.

Executable offline examples:

- `examples/pydantic_ai_tactic/structured_agent.py`
- `examples/pydantic_ai_tactic/surrounding_features.py`

Live provider credentials can be smoke-checked without sending prompts:

```bash
LLLM_LIVE_PROVIDER_TESTS=1 pytest tests/test_live_providers.py
```
