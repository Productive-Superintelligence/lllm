# {{project_title}}

Experiment-oriented LLLM scaffold with reusable prompts, batch scripts, and a default research tactic.

## Setup

```bash
uv sync --extra dev
cp .env.example .env
```

Set an API key in `.env` or your shell:

```bash
export OPENAI_API_KEY=sk-...
```

## Run One Topic

```bash
uv run python main.py "LLM evaluation for data analysis agents"
```

## Run a Batch

```bash
uv run python scripts/batch_topics.py
```

## Test

```bash
uv run pytest
```
