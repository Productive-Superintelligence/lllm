# Tactics

`Tactic` is LLLM's center model. A tactic is a small, typed, service-ready unit
that does one thing well.

Tactics provide:

- input and output type metadata,
- local `run` and async `arun` calls,
- optional streaming,
- `CallContext` request metadata,
- `TacticInfo` for services, package cards, and agent instructions.

```python
from lllm import Tactic


class EchoTactic(Tactic[EchoInput, EchoOutput]):
    name = "echo"
    input_type = EchoInput
    output_type = EchoOutput

    def _run(self, input_value, *, context=None):
        return EchoOutput(text=input_value.text.upper())
```

The protocol stays runtime-agnostic. A tactic can hide a Pydantic AI agent,
plain Python function, native prompt/dialog workflow, or remote service.
