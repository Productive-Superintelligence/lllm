# {{project_title}}

LLLM proxy scaffold for wrapping an API or local tool surface.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

Set an API key in `.env` or your shell:

```bash
export OPENAI_API_KEY=sk-...
```

## Inspect the Proxy

```bash
uv run python main.py --catalog
```

## Run

```bash
uv run python main.py "Use the sample data proxy to summarize account activity."
```

## Test

```bash
uv run pytest
```
