# Module Reference

This section lists the primary modules in the repository.

## Core Package (`lllm/`)

| File | Description |
| --- | --- |
| `lllm/__init__.py` | Public exports, `_auto_init()` to populate the default runtime from `lllm.toml`. |
| `lllm/core/resource.py` | `ResourceNode` (lazy/eager wrapper), `PackageInfo`, and convenience loaders (`load_prompt`, `load_tactic`, `load_proxy`, `load_config`, `load_resource`). |
| `lllm/core/runtime.py` | `Runtime` — unified `ResourceNode`-based registry with namespace-aware resolution. Named runtimes via `load_runtime` / `get_runtime`. |
| `lllm/core/config.py` | Package loading (`load_package`), TOML parsing, resource discovery, config inheritance (`resolve_config`, `_deep_merge`), `AgentSpec`, `parse_agent_configs`. |
| `lllm/core/templates.py` | Template manifest parsing, source resolution, and rendering used by `lllm create`. |
| `lllm/core/prompt.py` | `Prompt` model — template, parser, tools, handler. `Function`, `FunctionCall`, `MCP` descriptors. |
| `lllm/core/dialog.py` | `Dialog` — append-only message sequence with tree structure. `Message`, `DialogTreeNode`. Serialization uses qualified prompt keys. |
| `lllm/core/agent.py` | `Agent` — dialog management, agent call loop, retry/interrupt handling. |
| `lllm/core/tactic.py` | `Tactic` — top-level abstraction wiring agents to prompts. `TacticCallSession`, `_TrackedAgent`, `build_tactic`, `register_tactic_class`. |
| `lllm/core/const.py` | Enumerations (`Roles`, `Modalities`, `APITypes`), model cards, pricing utilities. |
| `lllm/core/log.py` | Replayable logging base classes (`ReplayableLogBase`, `LocalFileLog`, `NoLog`). |
| `lllm/proxies/base.py` | `BaseProxy`, `ProxyManager` runtime, `ProxyRegistrator` decorator, endpoint metadata. Namespace-aware proxy activation. |
| `lllm/proxies/builtin/` | Built-in proxy implementations (financial data, search, etc.). |
| `lllm/invokers/` | Provider registry and concrete backends implementing `BaseInvoker`. |
| `lllm/sandbox/` | `JupyterSession`, kernel management for code execution. |
| `lllm/tools/cua.py` | Experimental Computer Use Agent (Playwright browser automation). |
| `lllm/cli.py` | `lllm` command-line interface for project creation and package management. |
| `lllm/utils.py` | Filesystem helpers, caching, HTTP utilities, stream wrappers. |

## Templates (`lllm/templates/`)

| Path | Description |
| --- | --- |
| `lllm/templates/minimal/` | Smallest runnable LLLM app. |
| `lllm/templates/pipeline/` | Planner/writer/reviewer multi-agent workflow. |
| `lllm/templates/service/` | FastAPI-ready tactic service. |
| `lllm/templates/proxy/` | API/tool proxy integration scaffold. |
| `lllm/templates/research/` | Experiment workspace with batch scripts and data resources. |
