# Contributing To LLLM

LLLM is a lightweight framework for advanced agentic systems: a native
runtime for agents, prompts, and dialogs, a typed tactic boundary every
runtime shares, a package hub, and the experimental SSSN (network) and AAAX
(shell) modules.

## Development setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev,docs]"
.venv/bin/python -m pytest
.venv/bin/mkdocs build --strict
```

Install `".[all]"` to exercise every adapter, and set provider keys only
for the opt-in live tests (`pytest -m live`).

## Layout

| Path | What lives there |
| --- | --- |
| `lllm/protocol/` | The runtime-neutral tactic boundary: refs, context, validation. Imports no runtime. |
| `lllm/runtimes/native/` | The native runtime: Tactic (the program), Agent, Prompt, Dialog, invokers, proxies, sandbox. |
| `lllm/runtimes/` | Adapters: plain Python, Pydantic AI. |
| `lllm/services/` | FastAPI apps, remote tactics, tactic refs. |
| `lllm/hub/` | The `lllm.toml` package format, validation, local hub, cards, catalogs. |
| `lllm/network/`, `lllm/shell/` | SSSN and AAAX (experimental). |
| `lllm/templates/`, `lllm/create/` | `lllm create` project templates and scaffolds. |
| `docs/` | lllm.one, the docs site. The hub website at lllm.one/hub is the separate `lllm-hub` repository. |

## Design rules

- `lllm.Tactic` is the program. Keep `call()` programs and `_run()` hooks
  equally first-class, and keep every tactic servable without adapters.
- Keep the native runtime low-level and transparent: every agent turn
  (prompt, parser, handlers, tools, dialog state) should stay inspectable
  and overridable.
- `lllm.protocol` stays runtime-neutral: it must not import a runtime,
  services, the hub, or the experimental modules.
- `import lllm` must work with only the base dependencies; optional
  dependencies are imported lazily.
- Let adapters own their runtime: Pydantic AI keeps its tools, models, and
  validation.
- Keep FastAPI routes boring, typed, and OpenAPI-compatible.
- One `lllm.toml` serves the native runtime and the hub; new manifest
  sections must not break either reader.
- Every user-facing behavior needs a test, and every docs snippet should run.
