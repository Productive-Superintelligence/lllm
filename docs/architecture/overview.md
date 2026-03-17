# Architecture Overview

LLLM organises an agentic system into four layers. Each layer has a single, clear responsibility and they compose cleanly.

```
┌──────────────────────────────────────────────────┐
│                    Tactic                        │  ← "the program"
│  (orchestrates agents, owns the task interface)  │
├────────────────┬────────────────┬────────────────┤
│    Agent A     │    Agent B     │    Agent C     │  ← "the callers"
│  (model+loop)  │  (model+loop)  │  (model+loop)  │
├────────────────┴────────────────┴────────────────┤
│               Prompts (functions)                │  ← "the calls"
│  template · parser · tools · handlers            │
├──────────────────────────────────────────────────┤
│               Dialogs (state)                    │  ← "mental state"
│  per-agent message history, fork-able tree       │
└──────────────────────────────────────────────────┘
```

---

## The Four Abstractions

### Tactic — the program

A `Tactic` is the top-level unit of an agentic system. It:

- Accepts a task (string or Pydantic model) and returns a result
- Owns a group of agents and wires them together
- Is **stateless** — each `tactic(task)` call spins up fresh agent instances
- Can be subclassed, shared, and reused like a library module

```python
class Analytica(Tactic):
    name = "analytica"
    agent_group = ["analyzer", "synthesizer"]

    def call(self, task: str) -> str:
        ...
```

### Agent — the caller

An `Agent` holds a system prompt and a model. It executes prompts through a **call loop** that handles:

- Retries on LLM errors
- Exception handling (parsing failures, bad output)
- Interrupt handling (tool calls, multi-step reasoning)

An agent is **not** a long-running process — it operates on a `Dialog` (conversation state) that the tactic manages.

```python
agent.open("dialog_alias")    # create/attach a dialog
agent.receive("user message") # append user turn
response = agent.respond()    # run the call loop, return Message
```

### Prompt — the function

A `Prompt` is the specification for a single agent turn:

- **Template** — string or `.md` file with `{variable}` slots
- **Parser** — extracts structured output from the raw LLM text
- **Tools** — callable Python functions linked to the prompt's tool schema
- **Handlers** — `on_exception` and `on_interrupt` prompts for the call loop

Prompts compose: one prompt can `extend()` another, inheriting its tools and parser.

### Dialog — the mental state

A `Dialog` is the per-agent conversation history. Key properties:

- **Append-only** — messages are never mutated after appending
- **Fork-able** — `dialog.fork()` creates a branch at any point, enabling parallel reasoning paths or exception-recovery sub-dialogs
- **Tree structure** — forks form a tree; the agent always works on the active branch

Each agent maintains its own dialog. Agents don't share dialogs — they share *information* by passing content between them in the tactic's `call()` method.

---

## Data Flow

```
tactic(task)
  │
  ├─► Agent A
  │     dialog.put_text(task)
  │     agent.respond()
  │       └─► agent call loop
  │             ├─► llm_invoker.call(dialog)   # API call
  │             ├─► parse output               # Prompt.parser
  │             ├─► handle tool calls          # Prompt.tools
  │             └─► return Message
  │
  └─► Agent B
        dialog.put_text(agent_a_result)
        agent.respond()
          └─► ...
```

---

## Configuration & Discovery

For projects beyond a single script, LLLM uses a `lllm.toml` manifest to declare resources:

```
project/
├── lllm.toml           ← declares prompt/proxy/config folders
├── prompts/            ← .md files auto-register as Prompt resources
├── configs/            ← .yaml files auto-register as agent configs
├── tactics/            ← .py files with Tactic subclasses auto-register
└── proxies/            ← .py files with BaseProxy subclasses auto-register
```

Resources are loaded lazily into a **Runtime** registry and accessed by name:

```python
prompt = runtime.get_prompt("my_prompt")
config = runtime.get_config("my_tactic")
```

Multiple named runtimes can coexist for parallel experiments or isolated tests.

---

## Design Philosophy

LLLM is designed for developers and researchers — in the spirit of PyTorch and Hugging Face: modular, composable, and easy to prototype with — so that agentic systems can be built, shared, and reused like ordinary software modules.

### Core Principles

**Agentic system as a program.** An agentic system = agents (system prompt + base model, the "callers") + prompts (the "functions") + tactics (the "program" that wires them together). Treating each agent invocation as a well-defined function call — with explicit inputs, outputs, and error handling — minimises side effects, maximises compositionality, and makes parallelism straightforward.

**Dialog as internal mental state.** A dialog is the internal view each agent has of the conversation — not a shared global log. Different agents in a task maintain separate dialogs: each agent sees only what it has been told. The `top_prompt` at the head of the dialog also acts as the calling convention for the next turn, making the dialog a kind of function stack.

**Configuration as declaration.** System shape is declared in data (TOML/YAML), not hardcoded. What resources exist (prompts, proxies) and how they are wired (agent configs) are expressed as configuration — making systems inspectable, reproducible, and shareable without touching Python source.

**Low-level by default.** LLLM stops at `Tactic` as its highest abstraction. Higher-level orchestration — systems of systems, agent networks — is left to the application layer. For those patterns, see the [SSSN framework](https://github.com/Productive-Superintelligence/sssn).

### Advanced Capabilities

**Tool calling as programming.** Beyond regular LLM tool calling, the proxy system wraps tools with rich metadata, documentation, and activation filtering — making tools composable and testable rather than just ad-hoc function schemas. A mini-interpreter mode that executes tools as a mini Python script is on the roadmap.

**Tactics as a shared library.** Tactics are reusable modules. Like a Python package, you can import a tactic from a shared library and drop it into your own system. Every prompt, proxy, config, and tactic is an independent, loadable resource — declared in `lllm.toml` and namespaced to avoid collisions.

**Replayable logging.** The logging system records every invocation with enough information to recreate the exact execution context of any run — prompts, model arguments, tool results, costs. This makes debugging, A/B testing, and prompt optimisation tractable on production traces.

---

## Where to Go Next

- [Agent Call](../core/agent-call.md) — the call loop in detail
- [Prompts](../core/prompts.md) — templates, parsers, tools, handlers
- [Dialogs](../core/dialog.md) — message state and forking
- [Tactics](../core/tactic.md) — orchestration patterns
- [Config & Discovery](../core/config.md) — `lllm.toml` and YAML configs
- [Packages](../core/packages.md) — namespacing and dependency management
