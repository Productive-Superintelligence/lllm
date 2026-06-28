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
`examples/pydantic_ai_tactic/structured_agent.py` shows structured
input/output, streaming, and tool wrapping with an offline fake agent.

## Parsers

Shared parser utilities live outside runtime adapters:

```python
from lllm.parsers import DefaultTagParser

parser = DefaultTagParser(required_xml_tags=["answer"])
parsed = parser.parse("<answer>Hello</answer>")
```

Native prompts can use the same parser objects, and plain Python or Pydantic AI
wrappers can call them directly around tactic output.

## Native Prompt/Dialog Core

The native namespace preserves prompt and dialog primitives without letting them
shape the `Tactic` protocol:

```python
from lllm.runtimes.native import Dialog, Prompt, Role

system = Prompt(path="agent/system", prompt="You are a {style} assistant.")
dialog = Dialog(owner="agent")
dialog.put_prompt(system, prompt_args={"style": "careful"}, role=Role.SYSTEM)
dialog.put_text("Draft the next checkpoint.")

retry = dialog.fork(last_n=1, first_k=1)
```

Use these pieces for native runtime transcripts, prompt templates, tool schemas,
and forked histories. Wrap executable native agents with `NativeTacticAdapter`
when they need to cross the reusable tactic boundary.

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

## Package Metadata Helpers

LLLM does not own `psi.toml`, but it can export tactic metadata for PsiHub:

```python
from lllm.integrations import tactic_resource

resource = tactic_resource(EchoTactic())
```

Custom endpoint decorators are included in that metadata so package cards can
show domain routes alongside the portable `/run` interface.
