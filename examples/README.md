# LLLM Examples

All examples require an API key for at least one provider:

```bash
export OPENAI_API_KEY=sk-...        # uses gpt-4o / gpt-4o-mini
export ANTHROPIC_API_KEY=sk-ant-... # uses claude-haiku-4-5-20251001
```

---

## Standalone scripts

Run directly from the repo root. No additional setup needed.

### [`basic_chat.py`](basic_chat.py)

The simplest possible usage — `Tactic.quick()` with no configuration.

```bash
python examples/basic_chat.py
```

Covers: one-line chat, getting the agent back, custom system prompts, model selection.

---

### [`multi_turn_chat.py`](multi_turn_chat.py)

Multi-turn conversation using the agent's dialog management API.

```bash
python examples/multi_turn_chat.py
```

Covers: `agent.open()` / `receive()` / `respond()` cycle, history preservation, `fork()` to branch a dialog into two independent threads.

---

### [`tool_use.py`](tool_use.py)

Function calling with the `@tool` decorator.

```bash
python examples/tool_use.py
```

Covers: `@tool` decorator (auto-infers JSON schema from type hints), attaching tools to a `Prompt` via `function_list`, agent automatically calling tools and feeding results back, call diagnostics via `return_session=True`.

---

### [`structured_output.py`](structured_output.py)

Pydantic structured output using `Prompt(format=MyModel)`.

```bash
python examples/structured_output.py
```

Covers: defining a Pydantic output schema, `Prompt(format=...)` for JSON-mode responses, accessing `message.parsed`, hydrating back into the model.

---

## Advanced scripts

Located in [`advanced/`](advanced/). Each script auto-detects your API key and selects a default model. Override with:

```bash
export LLLM_EXAMPLE_MODEL=gpt-4o-mini   # any LiteLLM model ID
```

---

### [`advanced/multi_agent_tactic.py`](advanced/multi_agent_tactic.py)

A custom `Tactic` subclass orchestrating two agents in a pipeline.

```bash
python examples/advanced/multi_agent_tactic.py
```

Covers: subclassing `Tactic`, declaring `agent_group`, building from a config dict with inline `system_prompt`, passing data between agents, inspecting session cost via `return_session=True`.

---

### [`advanced/session_logging.py`](advanced/session_logging.py)

Persistent session logging with a SQLite `LogStore`.

```bash
python examples/advanced/session_logging.py
```

Covers: `sqlite_store()`, automatic persistence after every tactic call, `list_sessions()`, `load_session_record()`, inspecting cost and delivery.

---

### [`advanced/batch_processing.py`](advanced/batch_processing.py)

Processing many tasks at once with `bcall()` and `ccall()`.

```bash
python examples/advanced/batch_processing.py
```

Covers: `tactic.bcall(tasks, max_workers=N)` for ordered blocking batch, `fail_fast=False` to collect errors instead of raising, `tactic.ccall(tasks)` async generator that yields results as they complete (out-of-order).

---

## Full package example — Code Review Service

[`code_review_service/`](code_review_service/) is a complete LLLM **package** — the kind of structure you would use for a real project. It wraps a two-agent review pipeline as a FastAPI HTTP service.

```
code_review_service/
├── lllm.toml               ← package declaration; auto-discovers all resources
├── service.py              ← FastAPI service + --demo CLI mode
├── prompts/
│   ├── system.py           ← system prompts (auto-registered on import)
│   └── tasks.py            ← task prompts + CodeReviewResult Pydantic schema
├── tactics/
│   └── code_review.py      ← CodeReviewTactic (two-stage pipeline)
└── configs/
    ├── default.yaml        ← base config (model injected at startup)
    └── pro.yaml            ← inherits default, stricter model_args
```

```bash
cd examples/code_review_service

# CLI demo — no web server required
python service.py --demo

# FastAPI on :8080
pip install fastapi uvicorn
python service.py

# Use the production config
LLLM_CONFIG_PROFILE=pro python service.py --demo
```

See [`code_review_service/README.md`](code_review_service/README.md) for full documentation.

---

## Concepts by example

| Concept | Example |
|---------|---------|
| Zero-config single-agent chat | `basic_chat.py` |
| Multi-turn conversation + dialog fork | `multi_turn_chat.py` |
| `@tool` decorator + function calling | `tool_use.py` |
| Pydantic structured output | `structured_output.py` |
| Custom `Tactic` subclass | `advanced/multi_agent_tactic.py` |
| Session persistence + querying | `advanced/session_logging.py` |
| Batch + async concurrent execution | `advanced/batch_processing.py` |
| `lllm.toml` package structure | `code_review_service/` |
| Prompt files auto-discovered from disk | `code_review_service/prompts/` |
| YAML config with `base:` inheritance | `code_review_service/configs/` |
| Service / API wrapper | `code_review_service/service.py` |
