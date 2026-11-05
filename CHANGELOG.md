# Changelog

## 0.3.0

One project: LLLM now includes the package hub and the SSSN and AAAX
modules, with a single `lllm` command.

**Tactic**

- `from lllm import Tactic` is the program class again: write `call()`,
  declare `agent_group` for native agents built per call, use
  `Tactic.quick()` for a zero-config agent, `bcall`/`ccall` for batches. It
  subclasses the runtime-neutral `lllm.protocol.Tactic`, so call()-style
  tactics are served, streamed, packaged, and used as tools directly;
  `NativeTacticAdapter` is optional. `input_model`/`output_model` alias
  `input_type`/`output_type`.
- Sub-tactic calls roll up into the caller's session and cost.
- The native vocabulary is importable from `lllm` again (`Prompt`, `Agent`,
  `Dialog`, `tool`, `load_runtime`, `resolve_config`, `build_tactic`,
  `install_package`, `BaseProxy`, ...).

**Native runtime**

- Agents keep their own tools (system prompt, config `tools:`, proxies,
  skills) on every turn; `Prompt(agent_tools=False)` opts out.
- Proxies call their APIs end to end: `CALL_API` resolves documented
  endpoint paths (with `{param}` templates), adds API keys, and sends the
  request; only declared endpoints are reachable.
- Inference hints: agents can send session identity (one dialog, one
  session; forks name their parent), priority, expected output length, and
  KV-cache intent to NVIDIA Dynamo, vLLM, and SGLang.
- Tool schemas follow type annotations; strict mode is chosen automatically.
- Rate-limit retries back off and stop after `max_rate_limit_retries`.
- Fixes: Python 3.10 support, package loading with Python files in custom
  sections or `services/`, prompts without variables, JSON-safe dialogs,
  `load_proxy("pkg:name")`, `export_package` names, Jupyter `CALL_API`.

**Packages and hub**

- `lllm.toml` is the one package manifest: native discovery and hub
  metadata in the same file (formerly `psi.toml`); refs are `lllm://`.
- `lllm.hub`: validation, local hub, cards, agent cards, config templates,
  hub API, and `lllm hub index` for static catalogs (lllm.one/hub).
- `lllm create my-app` brings back the native project templates (minimal,
  pipeline, proxy, research, service); generated projects validate and
  serve as created.

**Modules**

- `lllm.network` (SSSN, experimental): channels, events, subscriptions,
  artifacts, snapshots, stores, and `TacticProcessor`.
- `lllm.shell` (AAAX, experimental): strategies built from lllm services.
- `lllm.Resolver` resolves tactic, channel, and snapshot refs from
  `.lllm/config.toml`.

**CLI**

- One command: `lllm create | init | setup | inspect | serve | hub |
  network | shell`. `lllm serve` and `lllm inspect` accept package paths
  and check declared API keys; `lllm setup` stores them.

## 0.1.0

- Added the runtime-agnostic `Tactic` protocol.
- Added plain Python, Pydantic AI, and native runtime adapters.
- Added FastAPI service wrappers, clients, custom endpoints, and SSE streaming.
- Added parser, proxy, and sandbox utilities outside the native runtime.
- Added optional LLLM hub metadata helpers and `lllm create` scaffolding.
