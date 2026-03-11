# Module Reference

This section lists the primary modules in the repository with short descriptions so you can jump into the right file quickly.

## Core Package (`lllm/`)

| File | Description |
| --- | --- |
| `lllm/__init__.py` | Public package exports and `auto_discover()` invocation. |
| `lllm/llm.py` | Agent framework (dialogs, agent call loop, prompt registry utilities). |
| `lllm/models.py` | Dataclasses for `Prompt`, `Message`, `Function`, `FunctionCall`, MCP descriptors, and parsing helpers. |
| `lllm/const.py` | Enumerations (roles, providers, features), model cards, pricing utilities, and tokenizer helpers. |
| `lllm/providers/` | Provider registry plus concrete backends (currently OpenAI chat/responses) implementing `BaseProvider`. |
| `lllm/log.py` | Replayable logging base classes plus `LocalFileLog` and `NoLog`. |
| `lllm/utils.py` | Filesystem helpers, caching, HTML utilities, frontend wrappers, and HTTP helper functions (`call_api`, `directory_tree`, etc.). |
| `lllm/config.py` | Functions for locating and loading `lllm.toml`, plus environment-variable guards. |
| `lllm/discovery.py` | Auto-discovery of prompts and proxies based on `lllm.toml`; includes module loader helpers. |
| `lllm/proxies.py` | `BaseProxy`, endpoint decorators, runtime `Proxy`, and supporting utilities (auto docs, auto tests, cutoff filtering). |
| `lllm/sandbox.py` | Notebook sandbox with `JupyterSession`, kernel management, and helper enums for programmatic execution. |
| `lllm/tools/cua.py` | Experimental Computer Use Agent for Playwright-driven browser automation. |
| `lllm/cli.py` | Implementation of the `lllm` command-line interface and template renderer. |
| `lllm/README.md` | Additional background on agent calls, prompts, and handlers. |

## Templates & Examples (`template/`)

| Path | Description |
| --- | --- |
| `template/init_template/` | Minimal scaffold used by `lllm create`. Includes `lllm.toml`, config stub, and `SimpleSystem`. |
| `template/example/` | Comprehensive example with system orchestration, prompt definitions, proxy modules, and MCP wiring. |
| `template/example/system/system.py` | Demonstrates experiment/stream management, sync/async call wrappers, and error handling. |
| `template/example/system/agent/agent.py` | Implements the `Vanilla` agent that orchestrates prompts and dialogs. |
| `template/example/system/proxy/modules/*.py` | Realistic proxy implementations (financial data, Google Trends, FRED, Wolfram Alpha, etc.). |

Use this table as a quick index when contributing or debugging.
