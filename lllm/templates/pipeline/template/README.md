# {{project_title}}

Multi-agent LLLM pipeline scaffold with planner, writer, and reviewer agents.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

Set an API key in `.env` or your shell:

```bash
export OPENAI_API_KEY=sk-...
```

## Run

```bash
uv run python main.py "Draft a launch plan for a small developer tool."
```

## Test

```bash
uv run pytest
```
