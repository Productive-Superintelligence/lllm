# 2. Agents and Dialogs

**Goal:** build a native `Agent`, hold multi-turn conversations, run several
dialogs on one agent, branch them with `fork()`, inspect and save them, and keep
long conversations inside the model's context window.

Every interaction in LLLM's native runtime is built from two primitives:

- An **`Agent`** owns an LLM identity: a name, a system prompt, a model, and the
  invoker that talks to the provider.
- A **`Dialog`** is that agent's mental state: an append-only, forkable message
  history. An agent can hold many named dialogs at once.

This is the low-level control LLLM is built around. Nothing is hidden in a
framework loop: you decide which dialog a message goes into, when the model is
called, and where a conversation branches. That is what research workflows need
when they retry from an earlier point, explore alternatives side by side, or
replay a transcript later.

Where Pydantic fits: every message is a Pydantic model (`Message`), and so are
the per-call diagnostics (`AgentCallSession`) and costs (`InvokeCost`). A
transcript is typed data you can dump to JSON and load back.

## Prerequisites

Start with [Lesson 1](01-quick-start.md) if you are new to LLLM. The native
runtime needs the `native` extra and a key for your model provider:

```bash
pip install "lllm-core[native]"
export OPENAI_API_KEY=sk-...   # or the key for any LiteLLM provider you use
```

Steps that call `respond()` send requests to your model provider. Building
agents, opening, forking, inspecting, and saving dialogs run locally.

---

## Step 1: Build an agent

```python
from lllm import Agent, Prompt
from lllm.runtimes.native.invokers import build_invoker

system = Prompt(path="assistant/system", prompt="You are a helpful assistant.")
invoker = build_invoker({"invoker": "litellm"})

agent = Agent(
    name="assistant",
    system_prompt=system,
    model="gpt-4o",
    llm_invoker=invoker,
)
```

- `Prompt` is LLLM's prompt object. Here it is just a template; the next
  lesson gives it parsers, tools, and handlers.
- `build_invoker({"invoker": "litellm"})` returns the LiteLLM invoker, so
  `model` accepts any LiteLLM model string: OpenAI, Anthropic, Gemini, local
  Ollama models, and more.

`Agent` is a dataclass. The fields you will use most:

| Field | Default | Meaning |
|---|---|---|
| `name` | required | The agent's role, used as the responder name on its messages |
| `system_prompt` | required | The `Prompt` that seeds every dialog the agent opens |
| `model` | required | Model identifier passed to the invoker |
| `llm_invoker` | required | The invoker that calls the provider |
| `model_args` | `{}` | Extra model arguments, such as `temperature` |
| `max_exception_retry` | `3` | Retries when the reply fails parsing ([Lesson 3](03-prompts.md)) |
| `max_interrupt_steps` | `5` | Tool-call rounds allowed per `respond()` ([Lesson 4](04-tools.md)) |
| `max_llm_recall` | `0` | Retries when the provider call raises an error other than a rate limit |
| `max_rate_limit_retries` | `8` | Retries after rate-limit errors, waiting exponentially longer (up to 60 seconds) |
| `context_manager` | `None` | Trims the dialog before each call (Step 10) |

For a quick experiment, `Tactic.quick()` from [Lesson 1](01-quick-start.md)
builds the same kind of agent in one line:

```python
from lllm import Tactic

agent = Tactic.quick(system_prompt="You are a helpful assistant.", model="gpt-4o")
```

Without a query, `quick()` returns the agent; with one, it opens a dialog, sends
the query, and returns the reply.

In larger projects you declare agents in configuration instead of constructing
them by hand; [Lesson 5](05-native-tactics.md) shows how.

---

## Step 2: Run one turn: `open`, `receive`, `respond`

```python
agent.open("chat")                                 # 1. create a named dialog
agent.receive("What is the capital of France?")   # 2. append a user message
response = agent.respond()                         # 3. call the model, append its reply
print(response.content)
```

1. `open(alias)` creates a dialog owned by the agent, seeds it with the system
   prompt, and makes it the active dialog. The alias (`"chat"`) is your name for
   this conversation; opening an alias that already exists raises `ValueError`.
2. `receive(text)` appends a user message to the active dialog.
3. `respond()` sends the dialog to the model, appends the reply, and returns it
   as a `Message`. `.content` holds the reply text.

---

## Step 3: Hold a multi-turn conversation

```python
agent.receive("And what language do they speak there?")
print(agent.respond().content)
```

Each `receive` / `respond` cycle appends to the same dialog, and the whole
history is sent on every call, so the model answers the follow-up with the
earlier turn in context.

---

## Step 4: Run several dialogs on one agent

An agent can hold several named dialogs and switch between them:

```python
agent.open("topic_a")               # opening also switches to the new dialog
agent.receive("Tell me about black holes.")
agent.respond()

agent.open("topic_b")
agent.receive("Tell me about cats.")
agent.respond()

agent.switch("topic_a")             # back to the first conversation
agent.receive("How massive is Sagittarius A*?")
agent.respond()

print(agent.active_alias)           # topic_a
print(sorted(agent.dialogs))        # ['chat', 'topic_a', 'topic_b']
```

`receive()` and `respond()` also take `alias=` to target a dialog without
switching to it, for example `agent.respond(alias="topic_b")`.

---

## Step 5: Inspect a dialog

```python
dialog = agent.current_dialog

for message in dialog.messages:
    print(f"[{message.name}] {message.content[:80]}")

print(dialog.overview())   # condensed, numbered view
print(dialog.cost)         # token counts and dollar cost so far
```

`overview()` returns one entry per message, truncated to 100 characters:

```text
[0. system (developer)]: You are a helpful assistant.

[1. user (user)]: Tell me about black holes.

[2. assistant (assistant)]: ...
```

`dialog.head` is the first message (the system prompt) and `dialog.tail` is the
last one (usually the latest reply). `dialog.cost` sums the usage the provider
reported for every message.

---

## Step 6: Fork a dialog

Forking creates a child dialog that starts with a copy of the parent's history
and then evolves independently. Use it to explore "what if" branches without
touching the main thread:

```python
agent.open("main")
agent.receive("You are interviewing a candidate for a backend role.")
agent.respond()

agent.fork("main", "strict_branch")   # copy "main" and switch to the copy
agent.receive("Explain recursion.")
strict_reply = agent.respond()

agent.fork("main", "lenient_branch")  # "main" itself was never changed
agent.receive("Tell me something interesting about yourself.")
lenient_reply = agent.respond()

print(len(agent.dialogs["main"].messages))           # 3
print(len(agent.dialogs["strict_branch"].messages))  # 5
```

`fork(alias, child_alias, last_n=0, first_k=1)` copies the whole history by
default. With `last_n > 0` the child keeps only the first `first_k` messages
plus the last `last_n`, which is handy for retrying from a clean point:

```python
agent.fork("main", "retry", last_n=1, first_k=1)   # system prompt + last message
```

Every dialog records its lineage. Print the tree from any dialog:

```python
print(agent.dialogs["main"].tree_overview())
```

```text
[a033d03f] owner=assistant msgs=3 split@None
  └─ [29cd695c] owner=assistant msgs=3 split@2
  └─ [6a89279f] owner=assistant msgs=5 split@3
    └─ [25480008] owner=assistant msgs=5 split@4
  └─ ...
```

Your named forks appear in the tree, and so do the short-lived working copies
that `respond()` makes for each model call (see "How it works" below). The ids
differ on every run.

---

## Step 7: Close a dialog

```python
old_dialog = agent.close("chat")   # remove it from the agent and return it
print(len(old_dialog.messages))
```

Closing returns the `Dialog` object so you can archive or inspect it. If the
closed dialog was active, the agent has no active dialog until you `open()` or
`switch()` again.

---

## Step 8: Read the reply `Message`

`respond()` returns a `Message`:

```python
agent.receive("Give me one more fact about cats.", alias="topic_b")
message = agent.respond(alias="topic_b")

message.content            # str: the reply text
message.role               # Roles.ASSISTANT
message.name               # "assistant" (the agent's name)
message.model              # the model used for the call
message.usage              # dict of token counts reported by the provider
message.cost               # InvokeCost: tokens and dollar cost for this reply
message.is_function_call   # True when the model asked for tools (Lesson 4)
message.parsed             # parser output (Lesson 3); {} when no parser ran
message.to_dict()          # JSON-ready dict
```

For diagnostics about the whole call, including parse retries and tool rounds,
ask for the session instead:

```python
agent.receive("And one more, please.", alias="topic_b")
session = agent.respond(alias="topic_b", return_session=True)
print(session.state)                    # "success"
print(session.delivery.content)         # the same Message respond() would return
print(session.exception_retries_count)  # 0 when the first reply parsed cleanly
```

[Lesson 7](07-logging.md) covers sessions in detail.

---

## Step 9: Save and restore a dialog

Dialogs serialize to JSON-ready dictionaries that include their message
history, lineage, child dialogs, and the latest prompt (tools are stored as
their schemas):

```python
import json
from lllm import Dialog

with open("main_dialog.json", "w") as f:
    json.dump(agent.dialogs["main"].to_dict(), f)

with open("main_dialog.json") as f:
    restored = Dialog.from_dict(json.load(f))

print(len(restored.messages), len(restored.children))
print(restored.overview())
```

Use saved dialogs to review runs, build datasets, or compare branches offline.
Single messages round-trip the same way with `message.to_dict()` and
`Message.from_dict(data)`.

---

## Step 10: Manage the context window

Long conversations can outgrow a model's context window. A `ContextManager`
trims what is sent to the model **before each call** without touching the
dialog itself.

### Use the built-in truncator

```python
from lllm.runtimes.native.core.dialog import DefaultContextManager

cm = DefaultContextManager("gpt-4o")                     # window size from LiteLLM
cm = DefaultContextManager("gpt-4o", max_tokens=32000)   # or an explicit cap

agent = Agent(
    name="assistant",
    system_prompt=system,
    model="gpt-4o",
    llm_invoker=invoker,
    context_manager=cm,
)
```

Before each model call, `DefaultContextManager` counts tokens with LiteLLM and,
when the dialog is too long:

- always keeps the first message (the system prompt);
- keeps messages from the most recent backwards until the budget runs out;
- cuts the oldest kept message to fit, prefixed with
  `[...earlier content truncated...]`;
- reserves a 5,000-token safety buffer below the limit.

### Write your own policy

Subclass `ContextManager`, set `name`, and implement `__call__`. It receives a
working copy of the dialog and returns the dialog to send:

```python
from typing import Optional

from lllm import Dialog
from lllm.runtimes.native.core.dialog import ContextManager


class RollingWindowManager(ContextManager):
    name = "rolling_window"

    def __init__(self, model_name: str, max_tokens: Optional[int] = None, keep_last: int = 10):
        self.model_name = model_name
        self.keep_last = keep_last

    def __call__(self, dialog: Dialog) -> Dialog:
        if len(dialog.messages) <= self.keep_last + 1:
            return dialog                                       # short enough
        return dialog.fork(last_n=self.keep_last, first_k=1)   # system prompt + recent turns


agent.context_manager = RollingWindowManager("gpt-4o", keep_last=6)
```

Keep the `model_name` and `max_tokens` parameters even if you ignore them:
agents built from configuration construct context managers with those two
keyword arguments.

### Declare it in configuration

When agents are built from a tactic config ([Lesson 5](05-native-tactics.md)),
context management is a config key. A global default applies to every agent and
each agent can override it:

```yaml
global:
  model_name: gpt-4o
  context_manager:
    type: default          # DefaultContextManager
    max_tokens: 128000

agent_configs:
  - name: small_agent
    model_name: gpt-4o-mini
    system_prompt: You answer briefly.
    context_manager:
      type: default
      max_tokens: 16000    # tighter cap for the smaller model

  - name: full_history_agent
    system_prompt: You keep every detail.
    context_manager:
      type: null           # disabled: always send the full dialog

  - name: rolling_agent
    system_prompt: You chat casually.
    context_manager:
      type: rolling_window # a registered custom policy
```

Custom types are looked up in the runtime registry the tactic uses, so register
the class there first:

```python
from lllm import Runtime

runtime = Runtime()
runtime.register_context_manager(RollingWindowManager)   # registered under its `name`
# then build the tactic with registry=runtime (Lesson 5)
```

---

## How it works: dialogs as mental state

LLLM treats a dialog as an agent's internal mental state. It is:

- **Append-only.** Messages are added with `put_*` methods and never edited in
  place.
- **Owned by one agent.** The agent that opens a dialog seeds it with its system
  prompt and records itself as the owner.
- **Forkable.** Branching happens through `fork()`, which records the parent,
  the split point, and the children.

`respond()` builds on these rules. For every model call it forks a working copy
of the dialog, applies the context manager to that copy, and sends it. Failed
attempts and repair messages stay in the working copy; your dialog receives
only accepted replies (plus, when tools are used, the tool calls and their
results). That keeps the canonical history clean and is why working copies show
up in `tree_overview()`.

---

## Summary

| Concept | API |
|---|---|
| Build an agent | `Agent(name, system_prompt, model, llm_invoker)` |
| One-line agent | `Tactic.quick(system_prompt=..., model=...)` |
| Create a dialog | `agent.open(alias)` |
| Append a user message | `agent.receive(text)` |
| Call the model | `agent.respond()` returns a `Message` |
| Switch dialogs | `agent.switch(alias)` |
| Branch a dialog | `agent.fork(alias, child_alias, last_n=0, first_k=1)` |
| Remove a dialog | `agent.close(alias)` returns the `Dialog` |
| Inspect | `dialog.messages`, `dialog.overview()`, `dialog.cost`, `dialog.tree_overview()` |
| Save and load | `dialog.to_dict()`, `Dialog.from_dict(data)` |
| Call diagnostics | `agent.respond(return_session=True)` |
| Trim context | `DefaultContextManager(model, max_tokens=...)` or a `ContextManager` subclass |

See the [Agent reference](../core/agent.md) and the
[Dialog reference](../core/dialog.md) for every method and field.

**Next:** [3. Prompts and Structured Output](03-prompts.md)
