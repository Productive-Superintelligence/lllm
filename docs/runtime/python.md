# Plain Python

Use the plain Python runtime when the work is deterministic application logic,
a small adapter, or an existing callable that does not need an agent runtime.

```python
from lllm.runtimes import as_tactic


def uppercase(text: str) -> str:
    return text.upper()


tactic = as_tactic(uppercase, name="uppercase")
assert tactic.run("hello") == "HELLO"
```

`CallableTactic` infers the first non-`context` parameter annotation as the
input type and the return annotation as the output type. You can override both
when you need stricter public schemas.

## Context-Aware Callables

If a callable accepts `context`, LLLM forwards the `CallContext`:

```python
from lllm import CallContext
from lllm.runtimes import as_tactic


def label(text: str, *, context=None) -> dict[str, str | None]:
    return {"text": text, "trace_id": context.trace_id if context else None}


tactic = as_tactic(label, name="label")
output = tactic.run("hello", context=CallContext(trace_id="trace-1"))
```

This is useful for logging, audit metadata, request correlation, and service
adapters.

## When To Prefer A Class

Subclass `Tactic` directly when the behavior has configuration, multiple call
paths, streaming, custom metadata, or custom endpoints.

```python
from lllm import Tactic


class UppercaseTactic(Tactic[str, str]):
    name = "uppercase"
    input_type = str
    output_type = str

    def _run(self, input_value, *, context=None):
        return input_value.upper()
```

## Serving

Plain Python tactics use the same service adapter as agent-backed tactics:

```python
from lllm.services import create_tactic_app

app = create_tactic_app(tactic)
```

The caller still sees `/run`, `/stream` when supported, and `/info`.

## Package Shape

```toml
[tactics.uppercase]
entry = "demo.tactics:build_tactic"
input = "str"
output = "str"
runtime = "python"
```

Plain Python is often the best starting point for protocol-level tutorials,
tests, package examples, and adapters around non-agent systems.
