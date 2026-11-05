# {{project_title}}

FastAPI-ready LLLM service scaffold.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

Set an API key in `.env` or your shell:

```bash
export OPENAI_API_KEY=sk-...
```

## Run a CLI Demo

```bash
uv run python main.py "Summarize this service in one sentence."
```

## Start the API

```bash
uv run python service.py
```

Then call:

```bash
curl -X POST http://localhost:8080/run \
  -H 'Content-Type: application/json' \
  -d '{"task": "Write a concise project summary."}'
```

## Test

```bash
uv run pytest
```
