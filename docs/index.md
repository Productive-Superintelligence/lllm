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

## Learning Path

LLLM is designed to grow with you. Here's the recommended reading order:

**1. [Getting Started](getting-started.md)** — Install and run your first agent in 5 lines.

**2. [Architecture Overview](architecture/overview.md)** — How the four abstractions (Agent, Prompt, Dialog, Tactic) fit together, and where the package system fits in.

**3. [Package System](architecture/packages.md)** — The organisational layer that makes projects beyond a single script work cleanly. This is the central concept for any real lllm project.

**4. [Tutorial: Build a Full Package](guides/building-agents.md)** — Walk through from a single agent to a complete multi-agent package with logging. Each step builds on the last.

**5. Advanced Customization** — Plug in your own components once you need them:

- [Invokers](core/invokers.md) — swap or extend the LLM backend (`BaseInvoker`)
- [Logging & Backends](core/logging.md) — session tracking, custom storage backends
- [Proxy & Tools](core/proxy-and-sandbox.md) — build custom tools with the proxy system

---

## Core Abstractions

| Concept | Role | Analogy |
|---------|------|---------|
| **Agent** | System prompt + base model + call loop | A "caller" |
| **Prompt** | Template + parser + tools + handlers | A "function" |
| **Dialog** | Per-agent conversation state | Internal "mental state" |
| **Tactic** | Wires agents to prompts, orchestrates collaboration | A "program" |

See the [Architecture Overview](architecture/overview.md) for the full picture, including design philosophy and data flow.
