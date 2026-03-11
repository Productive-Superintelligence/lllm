# Guide: Building an Agent

This walkthrough explains how to assemble a fully working agent using the components shipped in this repository.

## 1. Bootstrap a System

Use the CLI scaffold or copy the example system.

```bash
lllm create --name research_system
cd research_system
```

The scaffold creates:

- `lllm.toml` with prompt/proxy folders.
- `config/<project>/default.yaml` – edit this first (model name, logging, retries).
- `system/agent/` and `system/proxy/` skeletons.

## 2. Define Prompts

In `system/agent/prompts`, create prompt modules that describe your agent’s behavior.

```python
# system/agent/prompts/researcher.py
from lllm.models import Prompt

analysis = Prompt(
    path="analysis",
    prompt="""
    You are a researcher...
    <analysis>{task}</analysis>
    <answer></answer>
    """,
    xml_tags=["analysis", "answer"],
)
```

Prompts register automatically via auto-discovery, so the agent can load them with `Prompts('researcher')('analysis')`.

## 3. Wire Tooling via Proxies

Implement or reuse proxies under `system/proxy/modules`.

```python
from lllm.proxies import BaseProxy, ProxyRegistrator

@ProxyRegistrator(path="research/web", name="Search", description="Web search API")
class SearchProxy(BaseProxy):
    base_url = "https://api.example.com"
    api_key_name = "apikey"

    @BaseProxy.endpoint(...)
    def search(self, params):
        return params
```

Add the proxy folder to `lllm.toml`. Update your prompt to define `Function` objects and `link_function` to proxy methods so the agent can call them.

## 4. Implement the Agent

Subclass `Orchestrator` (or reuse the example `Vanilla` agent) under `system/agent/agent.py`.

```python
from lllm.llm import Orchestrator, Prompts

class ResearchAgent(Orchestrator):
    agent_type = "research"
    agent_group = ["research"]

    def __init__(self, config, ckpt_dir, stream, **kwargs):
        super().__init__(config, ckpt_dir, stream)
        self.agent = self.agents["research"]
        self.prompts = Prompts("research")

    def call(self, task: str, **kwargs):
        dialog = self.agent.init_dialog()
        dialog.send_message(self.prompts("analysis"), {"task": task})
        response, dialog, _ = self.agent.call(dialog)
        return response.parsed
```

Multiple sub-agents can be grouped by listing more names in `agent_group` and referencing them via `self.agents['name']`.

## 5. Build the System Wrapper

`system/system.py` wires configuration, checkpoint directories, and stream/logging utilities.

```python
from system.agent.agent import build_agent

class ResearchSystem:
    def __init__(self, config, stream=None):
        self.agent = build_agent(config, config['ckpt_dir'], stream)

    def call(self, task, **kwargs):
        return self.agent(task, **kwargs)
```

## 6. Run & Iterate

```python
from system.system import ResearchSystem
from system.utils import load_config

config = load_config("config/research/default.yaml")
system = ResearchSystem(config)
print(system.call("Summarize this paper"))
```

- Increase `max_exception_retry` when working with brittle parsers.
- Use `Proxy().api_catalog` to generate tool-selection prompts.
- Toggle `log_type` to `localfile` while debugging so you can replay entire sessions under `logs/`.
- For coding agents, spin up a `JupyterSession` via `lllm.sandbox` and surface session metadata inside your prompts so the model knows where files live.
