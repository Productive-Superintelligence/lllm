# 6. Packages and the Hub

**Goal:** organise a system as a package: prompts in modules, agents declared
in YAML, one `lllm.toml` that the runtime discovers and the hub publishes.
Then validate, serve, and share it.

Everything here runs offline; the last step that calls a model is marked.

## Step 1: generate a package

```bash
lllm create briefs
cd briefs
```

```text
briefs/
  lllm.toml
  prompts/assistant.py
  configs/default.yaml
  tactics/assistant.py
  main.py
  tests/test_smoke.py
  README.md
  pyproject.toml
  .env.example
  .gitignore
```

`lllm.toml` has two halves. The first tells the native runtime what to
discover; the second describes the package for the hub:

```toml
[package]
org = "local"
name = "briefs"
version = "0.1.0"
description = "Briefs"
primary = "tactics.assistant"

[prompts]
paths = ["prompts"]

[configs]
paths = ["configs"]

[tactics]
paths = ["tactics"]

[tactics.assistant]
entry = "main:create_tactic"
description = "A minimal LLLM project: one agent behind a tactic."

[card]
summary = "A minimal LLLM project: one agent behind a tactic."
tags = ["lllm", "minimal"]

[docs.readme]
path = "README.md"
title = "README"
```

## Step 2: discover the package

`load_runtime` reads `lllm.toml` and registers everything it finds under the
package's namespace:

```python
from lllm import load_runtime

runtime = load_runtime("lllm.toml")
print(sorted(runtime.keys()))
```

```text
['briefs.configs:default', 'briefs.prompts:assistant/system', 'briefs.tactics:assistant']
```

The prompt in `prompts/assistant.py` is a plain `Prompt` with `path="system"`;
its key is the module path plus the prompt path. Agents refer to it by that
key, so the prompt text can move or change without touching the code that
uses it.

## Step 3: declare agents in YAML

`configs/default.yaml` declares the tactic's agents instead of wiring them in
code:

```yaml
tactic_type: assistant

global:
  model_name: "gpt-4o-mini"
  model_args:
    temperature: 0.2

agent_configs:
  - name: assistant
    system_prompt_path: assistant/system
```

`global` applies to every agent; each entry in `agent_configs` names an
agent from the tactic's `agent_group` and can override any global value. A
config can also set `base: other_config` to inherit and deep-merge another
config. Resolve it by name:

```python
from lllm import resolve_config

config = resolve_config("default")
print(config["agent_configs"])
```

```text
[{'name': 'assistant', 'system_prompt_path': 'assistant/system'}]
```

## Step 4: build the tactic

`build_tactic` finds the tactic class named by `tactic_type` and builds its
agents from the config. To run it offline, register a canned invoker:

```python
from lllm import build_tactic, register_invoker
from lllm.runtimes.native import Message, Role
from lllm.runtimes.native.core.const import InvokeResult


class Canned:
    def __init__(self, config):
        pass

    def call(self, dialog, model, model_args=None, **kwargs):
        reply = f"[{model}] {dialog.tail.content}"
        return InvokeResult(message=Message(role=Role.ASSISTANT, content=reply, name="assistant"))


register_invoker("canned", Canned, overwrite=True)
config["invoker"] = "canned"

tactic = build_tactic(config)
print(tactic, "->", tactic("Say hello."))
```

```text
AssistantTactic(name='assistant', agents=['assistant']) -> [gpt-4o-mini] Say hello.
```

Drop the `invoker` line to use LiteLLM and a real model; `main.py` does exactly
that in `create_tactic()`.

## Step 5: validate

```bash
lllm hub validate .
```

```text
ok
```

Validation imports `main:create_tactic`, checks the refs and paths, and
reports warnings for anything that would make the package hard to use, such
as a missing card. [The manifest reference](../packages/manifest.md) lists
every check.

## Step 6: serve it (calls a model)

```bash
lllm setup .               # stores any API keys the package declares
lllm serve . --port 8000   # serves the primary tactic: /run, /stream, /info
```

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input": "Say hello in one sentence."}'
```

## Step 7: publish and share

```bash
lllm hub publish . --local
lllm hub list
lllm hub card local/briefs
lllm hub agent-card local/briefs
```

The card is generated for people, the agent card for coding agents that will
use the package. Someone else fetches it with `lllm hub get local/briefs`.

To publish your own read-only hub, write a static catalog any web server can
host (see [The Hub](../packages/hub.md#publish-a-static-hub)):

```bash
lllm hub index ./site/hub
```

```text
indexed 1 package(s) into ./site/hub
```

It writes `site/hub/data/catalog.json` and the package zip,
`site/hub/packages/local/briefs/0.1.0.zip`.

## Summary

| Step | Command or API |
| --- | --- |
| New package | `lllm create NAME` (or `lllm init` for an existing folder) |
| Discover resources | `lllm.load_runtime("lllm.toml")` |
| Load agent configs | `lllm.resolve_config("default")` |
| Build the program | `lllm.build_tactic(config)` |
| Check the package | `lllm hub validate .` |
| Serve it | `lllm setup .`, `lllm serve .` |
| Share it | `lllm hub publish . --local`, `lllm hub index` |

More in [Discovery](../packages/discovery.md), [Configs](../core/config.md),
and [The Hub](../packages/hub.md).

**Next:** [7. Logging and Cost Tracking](07-logging.md)
