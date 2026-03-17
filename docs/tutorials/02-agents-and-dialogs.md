# Lesson 2 — Agents and Dialogs

Every interaction in LLLM is built around two primitives: **`Agent`** and **`Dialog`**.

- An **Agent** owns an LLM identity: its name, system prompt, and model choice.
- A **Dialog** is the append-only conversation history that the agent maintains.

Understanding these two classes unlocks the rest of the framework.

---

## The Three-Step Pattern

```python
agent.open("my_dialog")          # 1. Create a new named dialog
agent.receive("Hello!")          # 2. Append a user message
response = agent.respond()       # 3. Call the LLM and append its reply
print(response.content)
```

Each step corresponds to a method on `Agent`. The string `"my_dialog"` is an **alias** — a human-readable name for this conversation. You can have multiple dialogs on the same agent simultaneously.

---

## Building an Agent

```python
from lllm import Agent, Prompt, Tactic
from lllm.invokers import build_invoker

prompt = Prompt(path="my_bot/system", prompt="You are a helpful assistant.")
invoker = build_invoker({"invoker": "litellm"})

agent = Agent(
    name="assistant",
    system_prompt=prompt,
    model="gpt-4o",
    llm_invoker=invoker,
)
```

Or use `Tactic.quick()` which does all of this for you:

```python
agent = Tactic.quick(system_prompt="You are a helpful assistant.", model="gpt-4o")
```

In real projects you'll define agents in YAML config files instead of inline Python dicts, and attach skills via the `skills:` config key. See [Lesson 6 — Configuration and Auto-Discovery](06-config-and-discovery.md) for the full config format and [Agent Skills](../core/agent.md#skills) for the skills reference.

---

## Multi-Turn Conversations

```python
agent.open("chat")
agent.receive("What is the capital of France?")
print(agent.respond().content)   # "Paris"

agent.receive("And what language do they speak?")
print(agent.respond().content)   # "French"
```

Each `receive` → `respond` cycle appends to the same dialog. The full history is sent to the LLM every time, so the model has context across turns.

---

## Multiple Dialogs on One Agent

An agent can hold multiple named dialogs and switch between them:

```python
agent.open("topic_a")
agent.receive("Tell me about black holes.")
resp_a = agent.respond()

agent.open("topic_b")               # creates a second dialog
agent.receive("Tell me about cats.")
resp_b = agent.respond()

agent.switch("topic_a")             # switch back
agent.receive("How massive is Sagittarius A*?")
resp_continue = agent.respond()
```

This is useful when one agent serves multiple independent conversations, or when you want to explore a branching scenario without losing the main thread.

---

## Dialog Inspection

```python
dialog = agent.current_dialog

# Print all messages
for msg in dialog.messages:
    print(f"[{msg.name}] {msg.content[:80]}")

# Overview (condensed)
print(dialog.overview())

# Token cost so far
print(dialog.cost)
```

`dialog.tail` is the last message (the most recent LLM reply).
`dialog.head` is the first message (the system prompt).

---

## Forking a Dialog

Forking creates a child dialog that shares the same history up to a split point, then diverges. This is useful for exploring "what if" branches:

```python
agent.open("main")
agent.receive("You are interviewing a candidate.")
agent.respond()

# Fork before a critical question
agent.fork("main", "strict_branch")   # child starts at this point
agent.receive("Explain recursion.")
strict_reply = agent.respond()

agent.switch("main")                  # back to the parent
agent.fork("main", "lenient_branch")
agent.receive("Tell me something interesting about yourself.")
lenient_reply = agent.respond()
```

The parent dialog is unchanged; each branch evolves independently.

---

## Closing a Dialog

```python
old_dialog = agent.close("chat")   # removes it from the agent, returns the Dialog object
```

Closed dialogs are returned so you can archive them, pass them elsewhere, or inspect their history.

---

## The `Message` Object

Every `respond()` call returns a `Message`:

```python
msg = agent.respond()

msg.content         # str — the plain text reply
msg.role            # Roles.ASSISTANT
msg.name            # name of the responder (agent.name)
msg.usage           # dict with token counts
msg.cost            # InvokeCost with prompt/completion tokens and dollar cost
msg.is_function_call  # True if the model requested a tool call
msg.parsed          # structured output from a parser (Lesson 3)
```

---

## Context Window Management

Long conversations can exceed a model's context limit. A `ContextManager` solves this by pruning the dialog **before each LLM call** without touching the canonical history.

### Using the built-in truncator

```python
from lllm.core.dialog import DefaultContextManager

cm = DefaultContextManager("gpt-4o")          # context window auto-detected
cm = DefaultContextManager("gpt-4o", max_tokens=32000)  # or set manually

agent = Agent(
    name="assistant",
    system_prompt=prompt,
    model="gpt-4o",
    llm_invoker=invoker,
    context_manager=cm,   # <-- attach here
)
```

The agent will now silently prune old messages on every turn, always keeping the system prompt and the most recent exchanges.

### What gets preserved

- **First message** (system prompt) — always kept.
- **Recent messages** — kept from the tail inward until the token budget is exhausted.
- **Border message** — the oldest kept message is character-truncated with `[...earlier content truncated...]` if it only partially fits.
- **Safety buffer** — 5 000 tokens are reserved below the context limit so you never land right at the edge.

### Writing a custom policy

Subclass `ContextManager`, set `name`, and implement `__call__`:

```python
from lllm.core.dialog import ContextManager, Dialog

class RollingWindowManager(ContextManager):
    name = "rolling_window"

    def __init__(self, model_name: str, max_tokens: int = None, keep_last: int = 10):
        self.model_name = model_name
        self.keep_last = keep_last

    def __call__(self, dialog: Dialog) -> Dialog:
        if len(dialog.messages) <= self.keep_last + 1:
            return dialog  # within limit — no-op
        return dialog.fork(last_n=self.keep_last, first_k=1)
```

Register it so config can find it by name:

```python
runtime.register_context_manager(RollingWindowManager)
```

### Config (YAML)

```yaml
global:
  context_manager:
    type: default        # or "rolling_window" after registering above
    max_tokens: 128000

agent_configs:
  - name: small_agent
    model_name: gpt-4o-mini
    context_manager:
      type: default
      max_tokens: 16000   # tighter cap for the cheaper model

  - name: stateless_agent
    model_name: gpt-4o
    context_manager:
      type: null          # disable — always sends the full dialog
```

---

## Architecture Note: Dialog as Mental State

LLLM treats a dialog as an agent's **internal mental state**. It is:

- **Append-only** — messages are never deleted or edited in place.
- **Owned by one agent** — each dialog belongs to the agent that created it.
- **Forkable** — branching is done via `fork()`, not mutation.

This design makes dialogs easy to reason about, log, and replay.

---

## Summary

| Concept | What it is |
|---|---|
| `Agent` | LLM identity (system prompt + model) |
| `Dialog` | Append-only message history owned by an agent |
| `agent.open(alias)` | Create a new dialog |
| `agent.receive(text)` | Append a user message |
| `agent.respond()` | Call the LLM, append reply, return `Message` |
| `agent.switch(alias)` | Change the active dialog |
| `agent.fork(src, dest)` | Branch a dialog |
| `dialog.messages` | Full message list |
| `DefaultContextManager` | Truncates dialog to fit context window |

**Next:** [Lesson 3 — Prompts and Structured Output](03-prompts.md)
