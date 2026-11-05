# Configuration

The native runtime uses two configuration formats: `lllm.toml` declares the
package (where prompts, tools, proxies, configs, and tactics live), and YAML
files declare the agents. Together they wire up an entire agentic system without
hardcoding paths or model choices in Python. Configuration is declaration: the
shape of the system is data you can read, diff, and swap.

---

## `lllm.toml`: the package manifest

Place `lllm.toml` at the project root. `lllm create my-app` (templates
`minimal`, the default, plus `pipeline`, `proxy`, `research`, `service`, or a
folder containing `lllm-template.toml`) generates a project with this file,
`prompts/`, `configs/default.yaml`, `tactics/`, a `main.py` that calls
`load_runtime()`, and an offline smoke test.

```toml
[package]
name = "my_system"
version = "0.1.0"
description = "Research assistant"

[prompts]
paths = ["prompts"]

[tools]
paths = ["tools"]

[proxies]
paths = ["proxies"]

[configs]
paths = ["configs"]

[tactics]
paths = ["tactics"]

[dependencies]
packages = ["./packages/shared_lib"]
```

### Sections

| Section | Purpose | Default folder |
| --- | --- | --- |
| `[package]` | `name`, `version`, `description`. Every resource is namespaced under `name` (default: the folder name). | — |
| `[prompts]` | Folders scanned for `Prompt` objects. | `prompts/` |
| `[tools]` | Folders scanned for `@tool` `Function` objects; other files there are registered as lazy data resources. | `tools/` |
| `[proxies]` | Folders scanned for `BaseProxy` subclasses. | `proxies/` |
| `[configs]` | Folders of YAML agent configs, loaded lazily. | `configs/` |
| `[tactics]` | Folders scanned for `Tactic` subclasses with a `name`. | `tactics/` |
| `[assets]` | Folders of data files, loaded lazily. | `assets/` |
| `[services]` | Folders of service files: module-level ASGI apps (FastAPI/Starlette) and service spec files (`.json`, `.yaml`, `.toml`) loaded as data. | `services/` |
| `[dependencies]` | `packages = [...]`: other packages to load, each into its own namespace. | — |

A section's `paths` replace its default folder; a section you omit falls back
to the default folder if it exists. Paths are relative to the `lllm.toml`. Any
other table with `paths` is a custom section for packaging data files. Entries
accept `under` prefixes (`"./vendor/prompts under vendor"`) and dependencies
accept `as` aliases (`"./pkgs/shared as shared"`). The full rules (keys,
namespaces, aliases, re-exports, custom sections) are in
[Package Discovery](../packages/discovery.md).

The same file can also describe the package for the LLLM hub: `[package]` keys
such as `org` and `kind`, and named resources such as `[tactics.brief]` with an
`entry`. Native discovery reads only `[package]` `name`/`version`/`description`,
`[dependencies] packages`, each section's `paths`, and custom sections, and it
ignores hub sections (`schemas`, `channels`, `snapshots`, `runs`, `config`,
`docs`, `examples`, `requirements`, `card`). See
[Package Manifest](../packages/manifest.md) and [Hub](../packages/hub.md).

### Loading

Nothing is loaded at import time. Load the package into the default runtime at
startup:

```python
from lllm import load_runtime

runtime = load_runtime()                    # nearest lllm.toml, searching upward from cwd
runtime = load_runtime("path/to/lllm.toml") # explicit file (or folder)
```

`load_runtime()` looks for `$LLLM_CONFIG` first, then `lllm.toml`,
`.lllm.toml`, or `LLLM.toml` in the working directory and its parents. It also
loads drop-in packages from `lllm_packages/` and `~/.lllm/packages/`. See
[Runtime & Registries](runtime.md).

### Environment variables

| Variable | Purpose |
| --- | --- |
| `LLLM_CONFIG` | Path to an `lllm.toml` file or to a folder containing one; overrides the search. |

### Multiple environments

```bash
LLLM_CONFIG=lllm.prod.toml python main.py
```

---

## Agent YAML configs

Agent config files live in `configs/` and describe how a tactic's agents are
built. They are loaded lazily, only when `load_config()` or `resolve_config()`
reads them.

### Structure

```yaml
base: ...              # optional: inherit from another config (no .yaml suffix)

tactic_type: analytica # optional: which registered tactic build_tactic() creates

global:                # defaults merged into every agent in this file
  model_name: gpt-4o
  api_type: completion
  model_args:
    temperature: 0.1
  max_exception_retry: 3
  max_interrupt_steps: 5
  max_llm_recall: 3
  tools:
    - shared_pkg.tools:search
    - shared_pkg.tactics:code_review
    - shared_pkg.proxies:market_data

agent_configs:
  - name: analyzer
    system_prompt_path: analytica/analyzer_system
    model_args:
      max_completion_tokens: 20000

  - name: synthesizer
    model_name: gpt-4o-mini            # overrides global
    system_prompt: "You are a synthesizer."
    model_args:
      temperature: 0.3                 # overrides global
```

### Top-level keys

| Key | Read by | Meaning |
| --- | --- | --- |
| `base` | `resolve_config` | Parent config name to inherit from. |
| `tactic_type` | `build_tactic` | Registered tactic name to build when `name=` is not given. |
| `global` | `Tactic` with `agent_group` | Defaults deep-merged into every agent entry. |
| `agent_configs` | `Tactic` with `agent_group` | List of agent entries, each with a `name`. |
| `invoker` | `Tactic` with `agent_group` | Invoker name (default `"litellm"`); see [Invokers](invokers.md#custom-invokers). |
| `invoker_config` | `Tactic` with `agent_group` | Dict passed to the invoker builder (default: the whole config). |
| `max_workers` | `Tactic` | Default thread-pool size for `bcall` / `ccall` (default 4). |

Anything else at the top level is yours to read in `call()` via `self.config`.

### Per-agent fields

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Agent name; must match an entry in the tactic's `agent_group`. |
| `model_name` | Yes (or from `global`) | Model id, e.g. `gpt-4o`, `anthropic/claude-sonnet-4-5`, `ollama/llama3`. |
| `system_prompt` | One of | Inline system prompt string (may contain `{variables}`). |
| `system_prompt_path` | these two | Registered prompt, e.g. `analytica/system` or `my_pkg:analytica/system`. |
| `api_type` | No | `"completion"` (default) or `"response"` (OpenAI Responses API). |
| `model_args` | No | Provider arguments (`temperature`, `max_tokens`, ...). Deep-merged with `global`. |
| `max_exception_retry` | No | Repair attempts after parse failures. Default 3. |
| `max_interrupt_steps` | No | Tool-call rounds per `respond()`. Default 5. |
| `max_llm_recall` | No | Retries after provider errors. Default 0. |
| `max_rate_limit_retries` | No | Retries after rate-limit errors, with exponential backoff up to 60 s. Default 8. |
| `tools` | No | Tool, tactic-tool, or proxy refs added to the system prompt. |
| `proxy` | No | Proxy settings; see [Proxies & Sandbox](proxy-and-sandbox.md#proxy-config-reference). |
| `skills` | No | Skill entries; see [Agent](agent.md#skills). |
| `context_manager` | No | `{type: ..., max_tokens: ...}`; no `type` means the default manager, `type: null` disables it. See [Dialog](dialog.md#context-manager). |
| `inference_hints` | No | Serving hints sent with each call; see [Inference Hints](inference-hints.md). |
| `extra_settings` | No | Free-form dict kept on the `AgentSpec` for your own code. |

Unrecognized keys are added to `model_args`, with a warning that names them, so
typos are visible.

### Global merge

`global` provides defaults for every agent and each entry overrides it. The
merge is deep: dicts merge recursively, while lists and scalars are replaced. So
you can set `temperature` globally and override only `max_tokens` per agent:

```yaml
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.1

agent_configs:
  - name: creative_writer
    system_prompt: "You are a creative writer."
    model_args:
      temperature: 0.9        # overrides global
      max_tokens: 4000        # added to global model_args
```

Result: `creative_writer` gets `model_name: gpt-4o` and
`model_args: {temperature: 0.9, max_tokens: 4000}`.

For list fields such as `tools` and `skills`, a per-agent list **replaces** the
global list, so one agent can opt into a different capability surface. `proxy`
and `context_manager` blocks are dicts, so they merge.

### Agent tools

`tools` attaches package resources to an agent:

```yaml
global:
  model_name: gpt-4o
  tools:
    - shared_pkg.tools:search
    - shared_pkg.tactics:code_review
    - shared_pkg.tactics:research#summarize
    - shared_pkg.proxies:market_data

agent_configs:
  - name: analyst
    system_prompt_path: system/analyst

  - name: writer
    system_prompt_path: system/writer
    tools: []   # replaces the global tools; writer gets none
```

| Entry | Becomes |
| --- | --- |
| `shared_pkg.tools:search` | A regular `Function` created with `@tool`. |
| `shared_pkg.tactics:code_review` | The tactic's default `@tactictool`. |
| `shared_pkg.tactics:research#summarize` | A named tactic-tool method. |
| `shared_pkg.proxies:market_data` | A proxy: adds `query_api_doc`, `run_python` (interpreter mode), and an API directory to the system prompt. |

This is the config-level form of the decoupled tool style: the config declares
what belongs to the agent's surface, and the implementations come from package
resources. Regular tool and tactic refs stay strings when the agent is built and
are resolved only when the prompt runs; bare names and `pkg:name` shorthand
resolve relative to the system prompt's package, then the default namespace.
Proxy refs must be explicit (`pkg.proxies:name`) because they are resolved at
build time to inject the proxy block.

Keep tool modules declarative at import time: define `@tool` functions and
`BaseProxy` classes there, and construct any agent a tool needs inside the tool
function body. See [Agent](agent.md#tools) for when these tools are offered to
the model.

---

## Inheritance via `base`

Configs inherit with `base`. Inheritance is recursive and uses a **deep merge**:
dict values merge, lists and scalars replace.

```
configs/
  base.yaml
  experiments/
    ablation.yaml
    full.yaml
```

```yaml
# base.yaml
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.1
agent_configs:
  - name: analyzer
    system_prompt_path: research/system
```

```yaml
# experiments/ablation.yaml
base: base                      # inherits configs/base.yaml
global:
  model_args:
    max_tokens: 500             # added to base's model_args
agent_configs:                  # replaces base's list (lists replace, dicts merge)
  - name: analyzer
    system_prompt_path: research/system_no_cot
```

```yaml
# experiments/full.yaml
base: experiments/ablation      # chain: full -> ablation -> base
global:
  model_name: o4-mini           # overrides all the way up
```

```python
from lllm import resolve_config

config = resolve_config("experiments/full")
config["global"]["model_name"]                  # 'o4-mini'
config["global"]["model_args"]["temperature"]   # 0.1  (from base)
config["global"]["model_args"]["max_tokens"]    # 500  (from ablation)
```

The result has no `base` key. Circular inheritance raises `ValueError`.

`base` names are resolved like any config ref: a bare name looks in the
runtime's default namespace (the root package). A config that ships in a
package meant to be used as a dependency should name its parent with the
package prefix (`base: shared_pkg:common`) so it resolves from any root.

### Recommended layout

```
configs/
├── default.yaml            # main config
├── experiments/
│   ├── fast.yaml           # base: default, swaps to smaller models
│   └── ablation.yaml       # base: default, swaps prompts
└── vendor/                 # pinned dependency configs
    └── A.yaml              # base: "A:default" plus overrides
```

---

## Managing dependency configs

When your package depends on other packages, keep the model simple: **each
package configures its own agents**. Your config declares your agents and may
point at prompts from dependencies by namespaced path.

### Assembly config

```yaml
# configs/default.yaml
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.1

agent_configs:
  - name: orchestrator
    system_prompt_path: my_pkg:orchestrator/system

  - name: analyzer
    system_prompt_path: A:analysis/system        # prompt from dependency A
    model_args:
      max_completion_tokens: 20000

  - name: searcher
    system_prompt_path: B:search/system          # prompt from dependency B
    model_name: gpt-4o-mini
```

Prompt *references* point into dependencies, like importing a function.
*Configuration decisions* (model, temperature, retries) stay yours.

### Vendoring configs for sub-tactics

When you compose a dependency's tactic, pull its config and apply your
overrides with `vendor_config`:

```python
from lllm import get_default_runtime
from lllm.runtimes.native.core.config import vendor_config

cfg = vendor_config("A:default", {
    "global": {
        "model_name": "gpt-4o",
        "model_args": {"temperature": 0.05},
    },
})

get_default_runtime().register_config("vendor/A", cfg, namespace="my_pkg.configs")
```

Or as a YAML file:

```yaml
# configs/vendor/A.yaml
base: "A:default"
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.05
```

`vendor_config(source, overrides=None, *, runtime=None, registry=None)` resolves
the full `base` chain, deep-merges your overrides, and returns a standalone dict
with no `base` key:

```python
cfg = vendor_config("A:default")                                         # resolve only
cfg = vendor_config("A:default", {"global": {"model_name": "gpt-4o-mini"}})

import yaml
with open("configs/vendor/A.yaml", "w") as f:
    yaml.dump(cfg, f)                                                    # save to disk
```

---

## Loading configs in Python

```python
from lllm import build_tactic, load_config, resolve_config

cfg = load_config("default")                    # raw, no inheritance
cfg = load_config("my_pkg:experiments/fast")    # package-qualified

cfg = resolve_config("experiments/full")        # with the base chain (recommended)

tactic = build_tactic(resolve_config("default"))
```

Configs can also be plain dicts or Pydantic models passed straight to a tactic:
`MyTactic({"global": {...}, "agent_configs": [...]})`.
