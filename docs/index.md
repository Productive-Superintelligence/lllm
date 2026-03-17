# Welcome to LLLM

<p align="center">
  <img src="assets/LLLM-logo.png" alt="LLLM Logo" width="200"/>
</p>

**LLLM** (Low-Level Language Model) is a lightweight framework for building advanced agentic systems. Go from a 5-line prototype to a production multi-agent system without rewriting anything — just add structure as complexity grows.

---

## Quick Start

```python
from lllm import Tactic

# One-liner — no config, no setup
response = Tactic.quick("What is the capital of France?")
print(response.content)

# Or get the agent for multi-turn chat
agent = Tactic.quick(system_prompt="You are a helpful assistant.", model="gpt-4o")
agent.open("chat")
agent.receive("Hello!")
print(agent.respond().content)
```

No `lllm.toml`, no folder structure, no subclassing. [Full quick start →](getting-started.md)

---

## Explore the Docs

<div class="grid cards" markdown>

-   :material-play-circle:{ .lg .middle } **Getting Started**

    ---

    Install LLLM and build your first agent in 5 lines. Covers single-agent, multi-turn, and the path to a full project.

    [:octicons-arrow-right-24: Quick Start](getting-started.md)

-   :material-map:{ .lg .middle } **Architecture Overview**

    ---

    How the four abstractions — Agent, Prompt, Dialog, Tactic — fit together. Design principles and data flow.

    [:octicons-arrow-right-24: Overview](architecture/overview.md)

-   :material-book-open-variant:{ .lg .middle } **Core Concepts**

    ---

    Deep dives into each abstraction: agent call loop, prompt templates & parsers, dialog state, tactic orchestration, config & packages.

    [:octicons-arrow-right-24: Agent Call](core/agent-call.md)

-   :material-hammer-wrench:{ .lg .middle } **Guides**

    ---

    Step-by-step tutorials: building agents from scratch, structuring a multi-agent project, and growing from prototype to production.

    [:octicons-arrow-right-24: Building Agents](guides/building-agents.md)

-   :material-code-tags:{ .lg .middle } **API Reference**

    ---

    Auto-generated docs for Agent, Tactic, Prompt, Dialog, LogStore, and all public classes.

    [:octicons-arrow-right-24: Core API](reference/core.md)

-   :material-package-variant:{ .lg .middle } **Package System**

    ---

    How `lllm.toml` wires resources together. Namespacing, dependencies, aliasing, and sharing tactics across projects.

    [:octicons-arrow-right-24: Packages](core/packages.md)

</div>

---

## Core Abstractions

| Concept | Role | Analogy |
|---------|------|---------|
| **Agent** | System prompt + base model + call loop | A "caller" |
| **Prompt** | Template + parser + tools + handlers | A "function" |
| **Dialog** | Per-agent conversation state | Internal "mental state" |
| **Tactic** | Wires agents to prompts, orchestrates collaboration | A "program" |

See the [Architecture Overview](architecture/overview.md) for the full picture, including design philosophy and data flow.
