# LLLM

<p align="center">
  <img src="assets/lllm-logo-text-dark.png" alt="LLLM" width="420">
</p>

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

Remote clients normalize base service URLs into `/run` and `/stream`
endpoints. `RemoteTactic.arun()` calls the JSON run endpoint, while
`RemoteTactic.astream()` consumes the service-sent event stream and yields the
same raw data items as local `Tactic.astream()`. Use
`RemoteTactic.aevents()` when you need the full `TacticEvent` envelopes.
`RemoteTactic.fetch_info()` and `RemoteTactic.afetch_info()` retrieve the
service-advertised `TacticInfo` from `/info` without making local `info()` do
network I/O.

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
`examples/pydantic_ai_tactic/surrounding_features.py` shows that normal
runtime-owned kwargs such as model settings, deps, eval hooks, durable IDs,
graph/workflow state, and tool approval pass through the wrapper.

Live provider credentials can be smoke-checked without sending prompts:

```bash
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
LLLM_LIVE_PROVIDER_TESTS=1 pytest tests/test_live_providers.py
```

Those opt-in tests list models using whichever credentials are available:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `TOGETHER_API_KEY`. Together is
included as an expected-soft-failure check because some networks return an
edge-level `403 error code: 1010` before API-key validation.

## Parsers

Shared parser utilities live outside runtime adapters:

```python
from lllm.parsers import DefaultTagParser

parser = DefaultTagParser(required_xml_tags=["answer"])
parsed = parser.parse("<answer>Hello</answer>")
```

Native prompts can use the same parser objects, and plain Python or Pydantic AI
wrappers can call them directly around tactic output.

## Proxies

Proxy utilities live at the `Tactic` boundary, so they can wrap any runtime:

```python
from lllm import InMemoryProxyLog, ProxyTactic

log = InMemoryProxyLog()
proxy = ProxyTactic(EchoTactic(), sink=log.append)
assert proxy.run({"text": "hello"}).text == "HELLO"
```

Use proxy hooks for small call-boundary transforms, observability, or local
guardrails. Payload capture is opt-in with `capture_inputs` and
`capture_outputs`. Proxies mirror wrapped tactic capabilities, including
streaming, and record captured stream chunks after the stream is consumed.

## Sandboxes

Sandbox utilities provide application-level guardrails around a tactic:

```python
from lllm import SandboxPolicy, SandboxedTactic

sandboxed = SandboxedTactic(
    EchoTactic(),
    policy=SandboxPolicy(max_input_bytes=4096, timeout_seconds=2.0),
)
```

Use them for payload budgets, request-metadata allowlists, and async/service
deadlines. They are not OS-level isolation for untrusted code.

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
pip install -e ".[dev,server]"
pytest
uvicorn app:app --reload
```

Or serve the generated tactic entrypoint directly:

```bash
lllm serve my_tactic.tactics:build_tactic --port 8000
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

[refs."psi://demo/echo/tactics/echo".metadata]
policy_url = "http://127.0.0.1:9000"
```

```python
resolver = TacticResolver.from_config(".")
tactic = resolver.resolve("psi://demo/echo/tactics/echo")
```

Tactic refs are strict package resource identifiers:
`psi://org/package/tactics/name` with no query string or fragment.
Shared config may include non-tactic refs from known PSI resource sections,
including `schemas`, `services`, `channels`, `snapshots`, `runs`, `configs`,
`docs`, `examples`, and `assets`, but malformed refs and unknown resource
sections fail validation.
`TacticResolver.from_config()` preserves `[refs."...".metadata]` on tactic URL
bindings; legacy top-level extras still work, and explicit metadata table
values win on duplicate keys. Tactic URL bindings must not also declare a
`store`, `path`, or `object` target. Tactic refs with a concrete target must
use `url`; `store`, `path`, and serialized `object` targets belong to other
layers or direct in-process registration. URL bindings must not include
embedded credentials, and binding metadata must not include raw secret-shaped
keys such as `api_key`, tokens, passwords, `authorization`, or credentials.
Use local credential refs such as `api_key_ref` or auth hooks instead.

Remote service failures raise `RemoteTacticError` with `status_code`,
`error_type`, `message`, `tactic`, `endpoint`, `request_id`, and raw `detail`
fields parsed from the service envelope.
Protocol and schema errors use HTTP 400; unexpected tactic runtime failures use
HTTP 500.

## Package Metadata Helpers

LLLM does not own `psi.toml`, but it can export tactic metadata for PsiHub:

```python
from lllm.integrations import tactic_resource

resource = tactic_resource(EchoTactic())
```

Custom endpoint decorators and tactic examples are included in that metadata so
package cards can show domain routes and concrete calls alongside the portable
`/run` interface. Use `@endpoint.get`, `@endpoint.post`, `@endpoint.put`,
`@endpoint.patch`, or `@endpoint.delete` to declare typed service routes without
changing the tactic protocol.
Runtime adapters such as `as_tactic`, `PydanticAITactic`, and
`NativeTacticAdapter` accept package refs, service refs, descriptions, examples,
and metadata so wrapper-created tactics can keep the same package-facing
contract as subclassed tactics.
