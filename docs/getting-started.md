# Getting Started

This guide gets you from a checkout to a running tactic service. It uses the
offline echo example so the path works without provider keys.

## Install

For local development:

```bash
python -m pip install -e ".[dev,server]"
```

For documentation work:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

Run the package tests before changing protocol or service behavior:

```bash
python -m pytest
```

## Run The Echo Example

The example contains a typed tactic and a FastAPI app:

```text
examples/echo_service/
  app.py
  tactics.py
```

Start the service:

```bash
cd examples/echo_service
uvicorn app:app --reload
```

Call the portable run endpoint:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input":{"text":"hello"}}'
```

Expected response includes:

```json
{
  "output": {
    "text": "HELLO"
  }
}
```

The same tactic can be inspected without starting a server:

```bash
lllm inspect examples.echo_service.tactics:build_tactic --json
```

The JSON includes the tactic name, schemas, capability flags, and metadata that
service clients and package cards can use.

## Serve An Entrypoint

`lllm serve` accepts an import path that returns a tactic:

```bash
lllm serve examples.echo_service.tactics:build_tactic --port 8000
```

Use this for local smoke tests and demos. For application code, creating the
FastAPI app explicitly keeps ownership clearer:

```python
from lllm.services import create_tactic_app
from examples.echo_service.tactics import build_tactic

app = create_tactic_app(build_tactic())
```

## Add Context

Callers can attach request and trace metadata through the run envelope:

```json
{
  "input": {"text": "hello"},
  "context": {
    "request_id": "demo-1",
    "trace_id": "trace-1",
    "metadata": {"caller": "docs"}
  }
}
```

The tactic receives this as `CallContext`. Runtime adapters may forward the
metadata when the underlying runtime supports it, but callers still interact
with the same LLLM envelope.

## Choose A Runtime

- Use a plain `Tactic` for deterministic Python logic or small service
  adapters.
- Use `PydanticAITactic` when a Pydantic AI agent owns model execution.
- Use `lllm.runtimes.native` when prompt/dialog lineage is part of the work.
- Use `RemoteTactic` when the implementation is already running as an HTTP
  service.

Continue with [Protocol](protocol/index.md) for the public boundary,
[Runtime](runtime/index.md) for adapter choices, then
[First Tactic](tutorials/first-tactic.md) for a step-by-step build.
