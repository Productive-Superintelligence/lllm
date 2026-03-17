# Welcome to LLLM

<p align="center">
  <img src="assets/LLLM-logo.png#only-light" alt="LLLM Logo" width="200"/>
  <img src="assets/LLLM-logo-white.png#only-dark" alt="LLLM Logo" width="200"/>
</p>

**LLLM** (Low-Level Language Model) is a lightweight framework for building advanced agentic systems. Go from a 5-line prototype to a production multi-agent system without rewriting anything — just add structure as complexity grows.

---

## The Mental Model

Everything in LLLM maps to a familiar programming concept:

```
┌──────────────────────────────────────────────────┐
│                    Tactic                        │  ← "the program"
│  (orchestrates agents, owns the task interface)  │
├────────────────┬────────────────┬────────────────┤
│    Agent A     │    Agent B     │    Agent C     │  ← "the callers"
│  (model+loop)  │  (model+loop)  │  (model+loop)  │
├────────────────┴────────────────┴────────────────┤
│               Prompts (functions)                │  ← "the calls"
│       template · parser · tools · proxies        │
├──────────────────────────────────────────────────┤
│               Dialogs (state)                    │  ← "internal state"
│    per-agent message history, fork-able tree     │
└──────────────────────────────────────────────────┘
```

| Concept | Role | Analogy |
|---------|------|---------|
| **Agent** | System prompt + base model + tools/proxies + call loop | A "caller" |
| **Prompt** | Template + parser + tools + handlers | A "function" |
| **Dialog** | Per-agent conversation state | Internal "mental state" |
| **Tactic** | Wires agents to prompts, orchestrates collaboration | A "program" |

---

## Design Principles

**Agentic system as a program.** An agentic system = agents (the callers) + prompts (the functions) + tactics (the program that wires them together). Each agent invocation is a well-defined function call with explicit inputs, outputs, and error handling — no hidden state, no magic.

**Dialog as internal mental state.** Each agent has its own dialog — its own view of the conversation. Agents don't share a global log; they share *information* by passing content between them in the tactic's `call()` method. This makes execution explicit, forkable, and replayable.

**Configuration as declaration.** System shape is declared in TOML/YAML, not hardcoded. What resources exist and how they are wired are expressed as configuration — making systems inspectable, reproducible, and shareable without touching Python.

**Low-level by default.** LLLM stops at `Tactic` as its highest abstraction and stays out of your way. No agent network topology, no automatic planning, no hidden LLM calls. You control the program.

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

## Grows With Your Project

LLLM is designed so that each stage of growth is a clean add-on to the previous one — nothing is rewritten:

| Stage | What you add | What you get |
|-------|-------------|--------------|
| **Prototype** | Nothing — just `Tactic.quick()` | 5-line single agent |
| **Structure** | `lllm.toml` + prompt files | Auto-discovery, no hardcoded paths |
| **Multi-agent** | Subclass `Tactic`, add YAML config | Orchestrated agents, session tracking |
| **Production** | Package export / `lllm pkg install` | Shareable, versioned agent infrastructure |

---

## Learning Path

<div class="grid cards" markdown>

-   :material-map:{ .lg .middle } **Architecture Overview**

    ---

    How the four abstractions fit together, design philosophy, and data flow.

    [:octicons-arrow-right-24: Overview](architecture/overview.md)

-   :material-school:{ .lg .middle } **Tutorials (8 lessons)**

    ---

    From a one-liner chat to a production multi-agent system — step by step. Covers tools, the proxy/interpreter system, tactics, config, logging, and advanced patterns.

    [:octicons-arrow-right-24: Start Lesson 1](tutorials/01-quick-start.md) · [:octicons-arrow-right-24: All lessons](tutorials/index.md)

-   :material-package-variant:{ .lg .middle } **Package System**

    ---

    The organisational layer that makes anything beyond a single script work cleanly — namespacing, discovery, sharing, and the `lllm pkg` CLI.

    [:octicons-arrow-right-24: Package System](architecture/packages.md)

-   :material-book-open-page-variant:{ .lg .middle } **Build a Full Package**

    ---

    Step-by-step guide from a single agent to a complete shareable LLLM package with prompts, tactics, and YAML configs.

    [:octicons-arrow-right-24: Guide](guides/building-agents.md)

-   :material-tune:{ .lg .middle } **Advanced Patterns**

    ---

    Multi-proxy orchestration, dialog forking, parallel batch processing, Jupyter sandbox, and custom execution environments.

    [:octicons-arrow-right-24: Advanced Patterns](tutorials/08-advanced-patterns.md)

-   :material-code-tags:{ .lg .middle } **API Reference**

    ---

    Auto-generated docs for Agent, Tactic, Prompt, Dialog, LogStore, and all public classes.

    [:octicons-arrow-right-24: Core API](reference/core.md)

</div>
