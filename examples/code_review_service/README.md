# Code Review Service

A complete LLLM **package** example. It demonstrates the full project structure — `lllm.toml`, prompt files, tactic files, YAML configs, and a config-inheritance chain — wrapped into a runnable FastAPI HTTP service.

## What it shows

| Concept | Where |
|---------|-------|
| `lllm.toml` package declaration | [`lllm.toml`](lllm.toml) |
| Prompts as module-level objects in `.py` files | [`prompts/system.py`](prompts/system.py), [`prompts/tasks.py`](prompts/tasks.py) |
| Structured output schema on a `Prompt` (`format=`) | [`prompts/tasks.py`](prompts/tasks.py) — `CodeReviewResult` |
| `Tactic` subclass with two agents | [`tactics/code_review.py`](tactics/code_review.py) |
| YAML agent config with `base:` inheritance | [`configs/default.yaml`](configs/default.yaml) → [`configs/pro.yaml`](configs/pro.yaml) |
| Auto-model injection at startup | [`service.py`](service.py) — `_detect_model()` |
| `build_tactic` + `resolve_config` | [`service.py`](service.py) |
| SQLite session logging | [`service.py`](service.py) — `sqlite_store` |
| FastAPI HTTP service + CLI demo mode | [`service.py`](service.py) |

## Package structure

```
code_review_service/
├── lllm.toml               ← declares the package; auto-registers all resources
│
├── service.py              ← entry point: FastAPI service or --demo CLI mode
│
├── prompts/
│   ├── __init__.py
│   ├── system.py           ← system prompts for analyzer + synthesizer agents
│   └── tasks.py            ← task prompts + CodeReviewResult Pydantic schema
│
├── tactics/
│   ├── __init__.py
│   └── code_review.py      ← CodeReviewTactic: two-stage analyze → review pipeline
│
└── configs/
    ├── default.yaml        ← base config (model injected at runtime)
    └── pro.yaml            ← inherits default, stricter temperature + token limits
```

## How it works

On startup `service.py`:

1. Sets `LLLM_CONFIG` to point at `lllm.toml` in this directory.
2. `import lllm` triggers auto-discovery: all `Prompt` objects in `prompts/`, the `CodeReviewTactic` class in `tactics/`, and both YAML configs in `configs/` are registered in the runtime.
3. Detects the active provider from environment variables and injects the model name into the resolved config.
4. Calls `build_tactic(config, log_store=sqlite_store(...))` to create one tactic instance shared across all requests (each call gets fresh agent copies).

Each review request runs a two-stage pipeline:

```
User code
   │
   ▼
[analyzer agent]  ← free-form analysis (system/analyzer prompt + task/analyze prompt)
   │  analysis text
   ▼
[synthesizer agent]  ← structured review (system/synthesizer prompt + task/review prompt)
   │  JSON → CodeReviewResult(summary, issues, suggestions, rating)
   ▼
Response
```

The `CodeReviewResult` schema is defined in `prompts/tasks.py` and stored on `review_task.format`. The tactic accesses it as `review_prompt.format` — no cross-file import required.

## Setup

```bash
cd examples/code_review_service

# Provide at least one API key:
export OPENAI_API_KEY=sk-...        # → uses gpt-4o-mini
export ANTHROPIC_API_KEY=sk-ant-... # → uses claude-haiku-4-5-20251001

# Override the model explicitly (any LiteLLM model ID):
export LLLM_EXAMPLE_MODEL=gpt-4o
```

## Running

### CLI demo (no web server required)

```bash
python service.py --demo
```

Sample output:

```
Model   : gpt-4o-mini
Profile : default

--- Input code ---
def find_duplicates(lst):
    seen = []
    ...

Reviewing… (this may take a few seconds)

--- Review ---
Rating      : 6/10
Summary     : The function is correct but uses O(n²) membership tests...
Issues:
  • `item in seen` is O(n) — use a set for O(1) lookup
  • Appending to `seen` after the check means the first occurrence is never flagged as duplicate
Suggestions:
  • Replace `seen = []` with `seen = set()` and use `seen.add(item)`
  • Add a docstring and type hints
```

### FastAPI service

```bash
pip install fastapi uvicorn
python service.py
# → http://0.0.0.0:8080
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ok", "model": "...", "profile": "..."}` |
| `POST` | `/review` | Reviews the submitted code snippet |

Example request:

```bash
curl -X POST http://localhost:8080/review \
     -H 'Content-Type: application/json' \
     -d '{
           "code": "def add(a, b):\n    return a + b",
           "language": "python"
         }'
```

Example response:

```json
{
  "summary": "Simple, correct function with no issues.",
  "issues": [],
  "suggestions": ["Add type hints.", "Add a docstring."],
  "rating": 8
}
```

### Use the production config profile

`pro.yaml` inherits from `default.yaml` and applies stricter model args:

```bash
LLLM_CONFIG_PROFILE=pro python service.py --demo
```

## Session logs

Every tactic call is persisted to `sessions.db` (created next to `service.py`). Use `lllm.logging` to query it:

```python
from lllm.logging import sqlite_store

store = sqlite_store("examples/code_review_service/sessions.db")
for s in store.list_sessions():
    print(s.session_id[:12], s.state, s.total_cost)
```
