# Contributing To LLLM

LLLM is the protocol and service layer for reusable agentic tactics.

## Development Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev]"
```

Run:

```bash
.venv/bin/python -m pytest
```

## Design Rules

- Keep `Tactic` runtime-agnostic and service-ready.
- Keep native runtime features under `lllm.runtimes.native`.
- Let Pydantic AI own provider, tool, eval, workflow, and observability behavior.
- Keep FastAPI routes boring, typed, and compatible with OpenAPI.
- Keep PsiHub package metadata optional and integration-scoped.
