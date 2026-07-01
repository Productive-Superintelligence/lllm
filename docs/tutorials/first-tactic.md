# First Tactic

Goal: wrap a typed Python class as a tactic and serve it over HTTP.

## Prerequisites

```bash
python -m pip install -e ".[dev]"
```

## Files Used

```text
examples/echo_service/
  app.py
  tactics.py
tests/
  test_examples.py
  test_composition.py
```

The same tactic shape is available as an executable example at
`examples/echo_service/tactics.py`.

## Tactic

Create a tactic:

```python
from pydantic import BaseModel
from lllm import Tactic


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


class EchoTactic(Tactic[EchoInput, EchoOutput]):
    name = "echo"
    input_type = EchoInput
    output_type = EchoOutput

    def _run(self, input_value, *, context=None):
        return EchoOutput(text=input_value.text.upper())
```

## Service

Serve it:

```python
from lllm.services import create_tactic_app

app = create_tactic_app(EchoTactic())
```

## Call

Call it:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input":{"text":"hello"}}'
```

Expected response:

```json
{
  "output": {
    "text": "HELLO"
  },
  "request_id": "...",
  "tactic": "echo"
}
```

## Package Ref

Bind a package-style ref to the same local tactic:

```python
from lllm import TacticResolver

resolver = TacticResolver()
resolver.register("psi://demo/echo/tactics/echo", EchoTactic())

assert resolver.run("psi://demo/echo/tactics/echo", {"text": "hi"}).text == "HI"
```

The same ref can later resolve to a service URL from `.psi/config.toml` without
changing callers.

## Verify

```bash
python -m pytest tests/test_examples.py tests/test_composition.py -q
```

Expected output:

```text
... passed
```

Next, read the [Tactic Boundary](../protocol/tactic-boundary.md) for the
protocol details, then try [Pydantic AI Runtime](pydantic-ai-runtime.md) or
[Native Core](native-core.md) when the tactic should be backed by an agent
runtime.
