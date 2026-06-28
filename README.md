# LLLM

LLLM is a small protocol and service layer for reusable agentic tactics.

The center model is `Tactic`: a typed, runtime-agnostic unit that does one
thing well and can be called locally, exposed through FastAPI, described for a
PsiHub package, and composed later through refs and local config.

## Install

```bash
pip install -e ".[dev]"
```

## Smallest Tactic

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


assert EchoTactic().run({"text": "hello"}).text == "HELLO"
```

## Serve It

```python
from lllm.services import create_tactic_app

app = create_tactic_app(EchoTactic())
```

```bash
uvicorn app:app --reload
curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input":{"text":"hello"}}'
```

## Pydantic AI

Pydantic AI remains the runtime owner. Configure the agent normally, then wrap
it:

```python
from lllm.runtimes import PydanticAITactic

tactic = PydanticAITactic(agent, input_type=str, output_type=str)
```

LLLM forwards request metadata where the agent run method accepts `metadata`.

## Create A Project

Generate a runnable tactic/service project:

```bash
lllm create plain my-tactic
cd my-tactic
pip install -e ".[dev]"
pytest
uvicorn app:app --reload
```

Templates:

- `plain`: typed Python `Tactic`.
- `pydantic-ai`: a Pydantic AI-style agent wrapped as a tactic.
- `native`: a native-style object wrapped behind the tactic boundary.

Add package metadata later with `psihub init`.

## Boundaries

- LLLM owns the `Tactic` protocol and service adapter.
- PsiHub owns `psi.toml`, package validation, package cards, local hub storage,
  downloads, and config templates.
- Native runtime ideas live under `lllm.runtimes.native` and do not shape the
  protocol layer.

## Compose Tactics

One tactic can call another directly or through an HTTP service. LLLM keeps this
as ref resolution, not service launching:

```python
from lllm import TacticResolver

resolver = TacticResolver()
resolver.register("psi://demo/echo/tactics/echo", EchoTactic())

result = resolver.run(
    "psi://demo/echo/tactics/echo",
    {"text": "hello"},
)
```

Local config can bind the same ref to a running service:

```toml
[refs."psi://demo/echo/tactics/echo"]
url = "http://127.0.0.1:8000/tactics/echo"
```

```python
resolver = TacticResolver.from_config(".")
tactic = resolver.resolve("psi://demo/echo/tactics/echo")
```
