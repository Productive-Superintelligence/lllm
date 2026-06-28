# Getting Started

## Install

```bash
python -m pip install -e ".[dev,server]"
```

For documentation work:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

## Run Tests

```bash
python -m pytest
```

## Run The Echo Example

```bash
cd examples/echo_service
uvicorn app:app --reload
```

Call it:

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

## Inspect A Tactic

```bash
lllm inspect examples.echo_service.tactics:build_tactic --json
```

## Serve An Entrypoint

```bash
lllm serve examples.echo_service.tactics:build_tactic --port 8000
```
