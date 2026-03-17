# Lesson 6 — Configuration and Auto-Discovery

As a project grows, hard-coded prompts and inline configs become hard to manage. LLLM's config system lets you declare resources in files and have them discovered automatically at startup.

---

## The `lllm.toml` File

Copy the example template to your project root:

```bash
cp lllm.toml.example lllm.toml
```

A minimal config:

```toml
[package]
name = "my_project"
version = "0.1.0"

[prompts]
paths = ["prompts/"]

[configs]
paths = ["configs/"]

[tactics]
paths = ["tactics/"]
```

LLLM scans the listed directories at startup. Any `Prompt` objects found in `.py` files are registered; any `.yaml`/`.yml` files in `configs/` are registered as config resources.

---

## Project Layout

```
my_project/
├── lllm.toml
├── lllm_packages/          # drop third-party packages here (auto-discovered)
├── prompts/
│   ├── greeter.py          # contains Prompt objects
│   └── analyst/
│       └── system.py
├── configs/
│   └── default.yaml        # tactic config
└── tactics/
    └── analyzer.py         # contains Tactic subclasses
```

Any sub-folder of `lllm_packages/` that contains an `lllm.toml` is loaded automatically at startup. See [Package Sharing](../architecture/packages.md#sharing-packages) for `lllm pkg install`, `export`, and `list` commands.

---

## Auto-Discovered Prompts

```python
# prompts/greeter.py
from lllm import Prompt

greeter_system = Prompt(
    path="greeter/system",
    prompt="You are {name}, a friendly assistant.",
)
```

After auto-discovery, load this prompt anywhere:

```python
from lllm import load_prompt

p = load_prompt("greeter/system")
```

The fully-qualified key (including the package namespace) is also accepted:

```python
p = load_prompt("my_project:greeter/system")
```

---

## Agent Config YAML

Define agents in a YAML file instead of inline Python dicts:

```yaml
# configs/default.yaml
tactic_type: analyzer

global:
  model_name: gpt-4o
  model_args:
    temperature: 0.1

agent_configs:
  - name: extractor
    system_prompt_path: analyst/system      # resolves to the registered Prompt
    model_args:
      max_completion_tokens: 4000
  - name: synthesizer
    system_prompt: "You are a concise writer."   # inline system prompt
```

Load and resolve the config at runtime:

```python
from lllm import resolve_config, build_tactic

config = resolve_config("default")
tactic = build_tactic(config, name="analyzer")
```

---

## Config Inheritance with `base`

```yaml
# configs/base.yaml
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.1
    seed: 42

agent_configs:
  - name: writer
    system_prompt: "You are a technical writer."
```

```yaml
# configs/fast.yaml
base: base    # inherits from base.yaml

global:
  model_name: gpt-4o-mini   # overrides model; other fields are kept
```

`resolve_config("fast")` deep-merges `fast.yaml` on top of `base.yaml`. Dict fields are merged recursively; scalars are replaced.

---

## Vendoring a Dependency's Config

When your project depends on another LLLM package, you can vendor its config and apply overrides:

```python
from lllm import vendor_config

cfg = vendor_config("other_pkg:default", overrides={
    "global": {"model_name": "claude-opus-4-6"},
})
```

This materialises the dependency's config into a standalone dict that no longer requires the dependency to be present.

---

## Package Dependencies

Declare LLLM package dependencies in `lllm.toml`:

```toml
[dependencies]
packages = [
    "./shared_prompts as shared",    # load ./shared_prompts, alias its namespace to "shared"
    "../another_pkg",
]
```

Dependent packages are loaded recursively with cycle detection. After loading, their resources are accessible under their package name (or alias):

```python
p = load_prompt("shared:common/system")
```

---

## Skills

> **Experimental.** Skills support follows the [agentskills.io](https://agentskills.io) open standard, which is actively evolving. Both the spec and this implementation may change in future releases.

**Skills** are reusable capability packages you attach to agents via config. They let you add specialised knowledge or workflows to any agent without bloating its system prompt — instructions are only loaded when the model actually needs them.

### Declaring skills in YAML

```yaml
# configs/default.yaml
global:
  model_name: claude-sonnet-4-6
  skills: [pdf, commit]          # all agents get these by default

agent_configs:
  - name: coder
    system_prompt_path: system/coder
    skills: [commit, code-review]  # replaces (not merges) the global list

  - name: writer
    system_prompt_path: system/writer
    # inherits global: skills: [pdf, commit]
```

Entry formats accepted in the `skills` list:

| Format | Example | How it works |
|---|---|---|
| Local name | `pdf` | Scanned from `.agents/skills/` or `~/.agents/skills/`; content injected into system prompt |
| Anthropic skill ID | `skill_01abc123` | Passed to the Anthropic API; content injected server-side |
| URL | `https://example.com/skills/review/SKILL.md` | Downloaded at agent build time |
| `"*"` | `skills: "*"` | Load all locally discovered skills |

### Creating a local skill

A skill is a directory under `.agents/skills/` with a `SKILL.md` file:

```
my_project/
└── .agents/
    └── skills/
        └── data-analysis/
            ├── SKILL.md            # required
            └── references/
                └── schema.md       # optional — loaded on demand
```

Minimal `SKILL.md`:

```markdown
---
name: data-analysis
description: Analyse tabular data, compute statistics, identify trends. Use when working with CSV or numerical datasets.
---

# Data Analysis

Follow these steps when analysing data:
1. ...
```

The `description` is the only thing the model sees before deciding to activate — write it as a trigger, not a title.

### How progressive disclosure works

LLLM injects only skill names and descriptions into the system prompt at startup (~50–100 tokens per skill). A built-in `activate_skill` tool lets the model pull the full instructions on demand:

```
<available_skills>
  <skill name="data-analysis">
    <description>Analyse tabular data... Use when working with CSV...</description>
  </skill>
</available_skills>
```

When the model calls `activate_skill("data-analysis")`, it receives the full `SKILL.md` body and a listing of any resource files. This keeps context lean for agents that have many installed skills.

For full details — Anthropic-hosted skill IDs, `allowed-tools`, best practices — see [Agent Skills reference](../core/agent.md#skills).

---

## Named Runtimes

For running parallel experiments without cross-contamination:

```python
from lllm import load_runtime, get_runtime

# Load a dedicated runtime from a specific config
load_runtime("./configs/exp1/lllm.toml", name="experiment_1")
load_runtime("./configs/exp2/lllm.toml", name="experiment_2")

rt1 = get_runtime("experiment_1")
rt2 = get_runtime("experiment_2")
```

Each named runtime has its own registry. Tactics built against `rt1` only see resources from `rt1`.

---

## Convenience Loaders

```python
from lllm import load_prompt, load_tactic, load_proxy, load_config, load_resource

p   = load_prompt("my_prompt")
t   = load_tactic("my_tactic")
cfg = load_config("default")
```

These all delegate to `get_default_runtime()` so they always see the auto-discovered resources.

---

## Virtual Folder Prefixes (`under`)

When you want resources from a folder to appear under a different path in the registry:

```toml
[prompts]
paths = [
    { path = "prompts/v2/", under = "v2" }
]
```

A file at `prompts/v2/greeter.py` with `path="greeter/system"` will be registered as `v2/greeter/system`.

---

## What Gets Discovered

| File type | Registered as |
|---|---|
| `.py` with `Prompt` instances | prompts |
| `.py` with `BaseProxy` subclasses | proxies |
| `.py` with `Tactic` subclasses | tactics |
| `.yaml` / `.yml` in `configs/` | configs |
| Any other file in a custom section | raw bytes or parsed dict |

---

## Summary

| Task | How |
|---|---|
| Auto-register prompts | Put `Prompt` objects in `.py` files in a scanned folder |
| Define agents via YAML | `agent_configs:` list in a `.yaml` config file |
| Config inheritance | `base: parent_config` in YAML |
| Load a config | `resolve_config("name")` |
| Load a prompt | `load_prompt("path")` |
| Package dependencies | `[dependencies] packages = [...]` in `lllm.toml` |
| Named runtimes | `load_runtime("name", ...)` |
| Attach skills to agents | `skills: [pdf, commit]` in YAML (global or per-agent) |
| Create a local skill | `.agents/skills/<name>/SKILL.md` with frontmatter |

**Next:** [Lesson 7 — Logging and Cost Tracking](07-logging.md)
