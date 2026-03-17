# Getting Started

## Installation

```bash
pip install lllm-core
```

Set your LLM provider API key:

```bash
export OPENAI_API_KEY=sk-...        # OpenAI
# or
export ANTHROPIC_API_KEY=sk-ant-... # Anthropic
# or any other LiteLLM-supported provider
```

---

## 5-Line Quick Start

No config files. No folder structure. No subclassing.

```python
from lllm import Tactic

agent = Tactic.quick("You are a helpful assistant.", model="gpt-4o")
agent.open("chat")
agent.receive("What is the capital of France?")
print(agent.respond().content)
```

`Tactic.quick()` creates an agent from a plain string system prompt. The `open / receive / respond` pattern maps to: start a conversation, add a message, get a reply.

To use Anthropic instead:

```python
agent = Tactic.quick("You are a helpful assistant.", model="claude-opus-4-6")
```

LiteLLM handles all provider differences automatically.

---

## Single Script

For experiments and one-off scripts, everything can stay inline. No config needed — just run it.

```python
from lllm import Tactic

agent = Tactic.quick("You are a helpful assistant.", model="gpt-4o")
agent.open("session1")
agent.receive("Summarize quantum computing in two sentences.")
print(agent.respond().content)
```

---

## Ready to Grow?

When your prompts get long, you need multiple agents, or you want to reuse components across projects, it's time to move to a proper package.

The [Tutorial: Build a Full Package](guides/building-agents.md) walks through it step by step — from a single file to a complete multi-agent system with logging and advanced customization.

Or understand the model first:

- [Architecture Overview](architecture/overview.md) — how the four abstractions fit together
- [Package System](architecture/packages.md) — the organisational layer explained
