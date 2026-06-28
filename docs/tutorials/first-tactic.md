# First Tactic

Goal: wrap a typed Python class as a tactic and serve it over HTTP.

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

Serve it:

```python
from lllm.services import create_tactic_app

app = create_tactic_app(EchoTactic())
```

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

Bind a package-style ref to the same local tactic:

```python
from lllm import TacticResolver

resolver = TacticResolver()
resolver.register("psi://demo/echo/tactics/echo", EchoTactic())

assert resolver.run("psi://demo/echo/tactics/echo", {"text": "hi"}).text == "HI"
```

The same ref can later resolve to a service URL from `.psi/config.toml` without
changing callers.
