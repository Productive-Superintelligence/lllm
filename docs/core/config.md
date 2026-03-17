# Configuration

LLLM uses two configuration formats: `lllm.toml` declares the package structure and resource locations; YAML files define the agents. Together they let an entire agentic system be wired together without hardcoding paths in Python.

---

## `lllm.toml` — Package Manifest

Place `lllm.toml` at the project root. LLLM finds it automatically by searching upward from the working directory, or you can point to it explicitly via `$LLLM_CONFIG`.

```toml
[package]
name = "my_system"
version = "0.1.0"

[prompts]
paths = ["prompts/"]

[proxies]
paths = ["proxies/"]

[configs]
paths = ["configs/"]

[tactics]
paths = ["tactics/"]

[dependencies]
packages = ["./packages/shared_lib"]
```

### Sections

| Section | Purpose | Default path |
| --- | --- | --- |
| `[package]` | Name, version, description. All resources are namespaced under `name`. | — |
| `[prompts]` | Folders scanned for `Prompt` objects and `.md` files. | `prompts/` |
| `[proxies]` | Folders scanned for `BaseProxy` subclasses. | `proxies/` |
| `[configs]` | Folders scanned for YAML agent config files (loaded lazily). | `configs/` |
| `[tactics]` | Folders scanned for `Tactic` subclasses. | `tactics/` |
| `[dependencies]` | Other packages to load. Each loads into its own namespace. | — |

Custom sections (e.g. `[assets]`, `[schemas]`) are also supported for packaging arbitrary files.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `LLLM_CONFIG` | Absolute path to a `lllm.toml` or folder; overrides auto-detection. |

### Multiple environments

```bash
LLLM_CONFIG=lllm.prod.toml python main.py
```

For the full package system — namespacing, resource URLs, dependency aliasing, re-exporting, custom sections — see [Package System](../architecture/packages.md).

---

## Agent YAML Config

Agent config files live in the `configs/` folder and describe how agents are constructed. Files are loaded lazily — only read when `load_config()` or `resolve_config()` is called.

### Structure

```yaml
base: ...              # optional — inherit from another config (no .yaml suffix)

global:                # defaults merged into every agent in this file
  model_name: gpt-4o
  api_type: completion
  model_args:
    temperature: 0.1
  max_exception_retry: 3
  max_interrupt_steps: 5
  max_llm_recall: 3

agent_configs:
  - name: analyzer
    system_prompt_path: analytica/analyzer_system
    model_args:
      max_completion_tokens: 20000

  - name: synthesizer
    model_name: gpt-4o-mini          # overrides global
    system_prompt: "You are a synthesizer."
    model_args:
      temperature: 0.3               # overrides global
```

### Per-Agent Fields

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Agent identifier — must match an entry in the tactic's `agent_group`. |
| `model_name` | Yes (or from global) | Model identifier (e.g. `gpt-4o`, `claude-opus-4-6`). |
| `system_prompt` | One of | Inline system prompt string. Creates a `Prompt` object directly. |
| `system_prompt_path` | these two | Path to a registered prompt (e.g. `analytica/system` or `my_pkg:research/system`). |
| `api_type` | No | `"completion"` (default) or `"response"` for OpenAI Responses API. |
| `model_args` | No | Model parameters (`temperature`, `max_tokens`, etc.). Merged with global. |
| `max_exception_retry` | No | Max retries on parse/validation errors. Default: 3. |
| `max_interrupt_steps` | No | Max consecutive tool-call interrupts. Default: 5. |
| `max_llm_recall` | No | Max retries on LLM API errors. Default: 0. |
| `extra_settings` | No | Reserved for advanced usage. |

Any unrecognized keys are treated as additional `model_args`.

### Global Merge

The `global` section provides defaults for all agents. Per-agent values override them. For `model_args`, the dicts are **merged** (not replaced) — so you can set `temperature` globally and override only `max_tokens` per agent.

```yaml
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.1

agent_configs:
  - name: creative_writer
    model_args:
      temperature: 0.9        # overrides global
      max_tokens: 4000         # added on top of global model_args
    system_prompt: "You are a creative writer."
```

Result: `creative_writer` gets `model_name: gpt-4o`, `model_args: {temperature: 0.9, max_tokens: 4000}`.

---

## Inheritance via `base`

Configs can inherit from other configs using the `base` key. Inheritance is recursive and uses **deep merge** — dict values merge, lists and scalars replace.

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
base: base                      # inherits from base.yaml
global:
  model_args:
    max_tokens: 500             # added to base's model_args
agent_configs:                  # replaces base's list entirely (lists replace, dicts merge)
  - name: analyzer
    system_prompt_path: research/system_no_cot
```

```yaml
# experiments/full.yaml
base: experiments/ablation      # chain: full → ablation → base
global:
  model_name: o4-mini-2025-04-16   # overrides all the way up
```

```python
from lllm import resolve_config

config = resolve_config("experiments/full")
# config["global"]["model_name"]           == "o4-mini-2025-04-16"
# config["global"]["model_args"]["temperature"] == 0.1    (from base)
# config["global"]["model_args"]["max_tokens"]  == 500    (from ablation)
```

Circular inheritance is detected and raises `ValueError`.

### Recommended layout

```
configs/
├── default.yaml            # main config
├── experiments/
│   ├── fast.yaml           # base: default → swap to mini models
│   └── ablation.yaml       # base: default → swap prompts
└── vendor/                 # pinned dependency configs
    └── A.yaml              # base: "A:default" + overrides
```

---

## Managing Dependency Configs

When your package depends on other packages, keep the config model simple: **each package is responsible for configuring its own dependencies**. Your config should only declare your own agents, referencing prompts from dependencies by their namespaced path.

### Assembly Config Pattern

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

Prompt *references* point into dependencies (that's fine — importing a function). But *configuration decisions* (model, temperature, retries) are owned by you.

### Config Vendoring for Sub-Tactics

When you compose a dependency tactic as a sub-tactic, use `vendor_config` to pull its config and apply your overrides:

```python
from lllm import vendor_config, get_default_runtime

cfg = vendor_config("A:default", {
    "global": {
        "model_name": "gpt-4o",
        "model_args": {"temperature": 0.05},
    },
})

runtime = get_default_runtime()
runtime.register_config("vendor/A", cfg, namespace="my_pkg.configs")
```

Or equivalently, as a YAML file:

```yaml
# configs/vendor/A.yaml
base: "A:default"
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.05
```

`vendor_config` resolves the full `base` chain, applies your overrides via deep merge, and returns a standalone config with no `base` key.

```python
from lllm import vendor_config

cfg = vendor_config("A:default")                              # resolve only
cfg = vendor_config("A:default", {"global": {"model_name": "gpt-4o-mini"}})  # with overrides

import yaml
with open("configs/vendor/A.yaml", "w") as f:
    yaml.dump(cfg, f)                                         # save to disk
```

---

## Loading Configs in Python

```python
from lllm import load_config, resolve_config

# Direct load — no inheritance resolution
cfg = load_config("default")
cfg = load_config("my_pkg:experiments/fast")

# With full base chain resolution (recommended)
cfg = resolve_config("experiments/full")

# build_tactic handles everything automatically
from lllm import build_tactic
tactic = build_tactic(resolve_config("default"), ckpt_dir="./runs")
```
