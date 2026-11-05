# {{project_title}}

Minimal LLLM project scaffold.

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
uv run python main.py "What should I build with LLLM?"
```

Without `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python main.py "What should I build with LLLM?"
```

## Test

```bash
uv run pytest
```
