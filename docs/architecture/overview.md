# Architecture Overview

LLLM ships as a Python package (`lllm/`) plus reusable templates (`template/`). The repository can be used directly as a library (install via `pip install git+...`) or cloned as a project starter.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `lllm/` | Core runtime (agents, prompts, logging, proxies, sandbox, CLI helpers). |
| `template/` | Scaffolds consumed by `lllm create --name <system>`. Includes a minimal `init_template` and a richer `example` template with ready-made proxies. |
| `README.md` | Quick-start instructions for installing the package and enabling auto-discovery. |
| `requirements.txt` / `pyproject.toml` | Packaging metadata when installing as a library. |

Inside `lllm/` the high-level structure is:

- `llm.py` – agent orchestration, dialog state machine, and LLM caller/responder implementations.
- `models.py` – dataclasses for `Prompt`, `Message`, `Function`, `FunctionCall`, and MCP tooling.
- `proxies.py` – API proxy registry plus the `Proxy` runtime that normalizes third-party APIs.
- `sandbox.py` – programmable Jupyter notebook sandbox for code-execution agents.
- `config.py` & `discovery.py` – `lllm.toml` resolution and auto-registration of prompts/proxies.
- `log.py` – replayable logging backends (`LocalFileLog`, `NoLog`).
- `const.py` – model metadata, pricing logic, enumerations, and helpers like tokenizers.
- `utils.py` – filesystem helpers, caching, Streamlit-friendly wrappers, and API utilities.
- `cli.py` – implementation of the `lllm` CLI entry-point.

## Runtime Flow

1. **Configuration & discovery** – `lllm.auto_discover()` runs on import. It reads `lllm.toml` (or `$LLLM_CONFIG`) to find prompt and proxy folders, imports every module inside, and registers any `Prompt` or `BaseProxy` subclasses it encounters. Environment variables let you opt-out (`LLLM_AUTO_DISCOVER=0`).
2. **System bootstrap** – A system (see `template/init_template/system/system.py`) constructs an agent via `lllm.core.agent.build_agent`, passing the YAML configuration, checkpoint directory, and a stream/logger implementation.
3. **Agent call loop** – `Agent.call` seeds a dialog with its system prompt, loads an invocation prompt, and drives the deterministic agent-call state machine until it yields a parsed response or an exception.
4. **Prompt handlers** – If the response contains errors, exception handlers rewrite the dialog and retry. If the response triggers function calls, interrupt handlers inject tool results and continue the loop.
5. **Proxy execution** – When tool/function calls target external APIs, they go through the `Proxy` runtime. Proxy modules describe endpoints declaratively using decorators so that agents can enumerate available tools and call them uniformly.
6. **Sandboxed tooling** – For advanced agents, the sandbox components provide Jupyter kernels and computer-use helpers (see `lllm/sandbox.py` and `lllm/tools/cua.py`). These enable notebook automation or browser automation with a standard interface.
7. **Logging & replay** – Every dialog, frontend event, and replayable artifact is written to a `ReplayableLogBase` collection. Implementations range from no-op logging (`NoLog`) to local file logging with per-session folders.

## Component Relationships

- Prompts depend on parsers, functions, and MCP servers defined in `lllm/models.py`.
- Agents consume prompts and proxies; agents are registered automatically via `Orchestra.__init_subclass__`.
- Proxies can be shared between systems, and their metadata powers auto-generated API catalogs that prompts can embed in instructions.
- Templates wire all of the above into runnable systems and document expected configuration keys.

For a deep dive into the agent state machine, continue to [Agent Call](../core/agent-call.md).
