# Dialog and Message

## Overview

A `Dialog` is one agent's internal view of a conversation: its mental state. It
holds an ordered, append-only list of `Message` objects, tracks which `Prompt`
governs the next model turn (`top_prompt`), and forks into branches.

Each dialog is owned by one agent and carries a `DialogTreeNode` that records
its position in a dialog tree. That is what makes branching, speculation, and
session reconstruction from logs possible.

A `Message` is a clean record of one turn. Per-invocation diagnostics (raw
provider responses, retries, the model args actually sent) live on
`InvokeResult` and `AgentCallSession`, not on the message.

```python
from lllm import Dialog, Message, Prompt, Role
```

---

## Message

A `Message` captures what was said in one turn.

| Field | Type | Description |
| --- | --- | --- |
| `role` | `Roles` | `SYSTEM`, `USER`, `ASSISTANT`, `TOOL`, `TOOL_CALL` |
| `content` | `str \| List[Dict]` | Text, a base64 image, or multipart content |
| `name` | `str` | Sender: `"user"`, `"system"`, an agent name, a tool name |
| `function_calls` | `List[FunctionCall]` | Tool calls requested by the model |
| `modality` | `Modalities` | `TEXT` or `IMAGE` (also `AUDIO`, `FUNCTION_CALL`) |
| `logprobs` | `List[TokenLogprob]` | Token log probabilities, when the provider returns them |
| `parsed` | `Dict[str, Any]` | Output of the prompt's parser (or decoded `format` JSON) |
| `model` | `str \| None` | Model that produced the message |
| `usage` | `Dict[str, Any]` | Provider token usage plus LLLM's cost fields |
| `metadata` | `Dict[str, Any]` | Arbitrary metadata (`tool_call_id`, `dialog_id`, `caption`, ...) |
| `api_type` | `APITypes` | `COMPLETION` or `RESPONSE` |
| `vectors` | `List[float]` | Placeholder for embedding vectors |

Computed properties:

```python
message.cost              # InvokeCost: token counts and dollar costs from usage
message.is_function_call  # True if the model requested tool calls
message.sanitized_name    # name safe for provider APIs (alphanumerics, _ and -, max 64 chars)
```

`InvokeCost` has token fields (`prompt_tokens`, `completion_tokens`,
`total_tokens`, `cached_prompt_tokens`, `reasoning_tokens`, audio tokens),
per-token rates, and dollar fields (`prompt_cost`, `completion_cost`, `cost`).
Costs add with `+`; `str(cost)` prints a readable summary.

Messages are Pydantic models: `message.to_dict()` / `Message.from_dict(d)`
round-trip every field. `Roles.SYSTEM` is sent to providers with the
`"developer"` role.

---

## Dialog

### Construction

Inside an agent, dialogs are created by `Agent.open()`, which builds the dialog,
seeds it with the rendered system prompt, and registers it under an alias:

```python
agent.open("planning", prompt_args={"language": "Python"}, session_name="run_001")
```

You can also build one directly, which is useful for tests, offline dataset
generation, or replaying a conversation:

```python
from lllm import Dialog, Prompt, Role

system = Prompt(path="assistant/system", prompt="You are {style}.")

dialog = Dialog(owner="assistant", session_name="demo")
dialog.put_prompt(system, {"style": "concise"}, role=Role.SYSTEM, name="system")
dialog.put_text("Explain dialog trees in one sentence.")
```

`Dialog(session_name=None, top_prompt=None, runtime=None, owner=None)`: the
session name defaults to a timestamp, and the runtime to the default runtime
(used to resolve prompt paths).

### Ownership

Every dialog is owned by the agent that created it. `owner` records the agent
name, and the system message is always that agent's persona. Do not hand one
agent's dialog to another: the second agent would answer under the first one's
system prompt. Pass content between agents instead.

### The `top_prompt` calling convention

`top_prompt` is the active calling convention: the `Prompt` whose parser,
format, tools, and handler govern the next model turn.

| Operation | `top_prompt` after it |
| --- | --- |
| `put_prompt(prompt, ...)` | `prompt` |
| `put_text(text, ...)` | A fresh plain prompt for that text (no parser, no tools of its own) |
| `put_image(image, ...)` | Unchanged |
| `fork()` | Inherited by the child |

When the agent loop starts and `top_prompt` is `None`, it uses the agent's
system prompt. It then adds the agent's own tools (those on its system prompt)
to the top prompt unless that prompt sets `agent_tools=False`, so a plain-text
turn still has the agent's tools; see [Agent](agent.md#where-tools-and-parsers-come-from).
During the loop, tool results are appended with `put_prompt` using the
handler's interrupt prompt, so the contract (tools, parser) carries through the
tool rounds.

### Appending messages

Three primitives cover all cases.

**`put_prompt(prompt, prompt_args=None, name="user", metadata=None, role=Roles.USER)`**
is stateful: it renders the template, appends the message, and advances
`top_prompt`. Use it for any message that carries a calling convention. A string
is resolved as a prompt path through the dialog's runtime.

```python
dialog.put_prompt(my_prompt, {"task": task_str}, name="user")
dialog.put_prompt("my_pkg.prompts:research/agent", {"topic": "transformers"})
```

**`put_text(text, name="user", metadata=None, role=Roles.USER)`** appends raw
text, for unstructured input.

```python
dialog.put_text("What is the capital of France?")
```

**`put_image(image, caption=None, name="user", metadata=None, role=Roles.USER)`**
appends an image message. It accepts a file path (`str` or `Path`), a PIL
`Image`, or a base64 string of a PNG, JPEG, WebP, or GIF; the caption is stored
in `metadata["caption"]`.

```python
dialog.put_image("screenshot.png", caption="Current state of the UI")
```

All three return the new `Message` and go through `dialog.append(message)`,
which stamps `metadata["dialog_id"]`.

### Inspection

```python
dialog.tail        # last Message, or None
dialog.head        # first Message (the system prompt), or None
dialog.messages    # the message list (treat as read-only)
dialog.cost        # InvokeCost summed over all messages
dialog.overview()  # one line per message, contents truncated to 100 chars
dialog.dialog_id   # this dialog's id (from its tree node)
```

```
[0. system (developer)]: You are concise.

[1. user (user)]: Explain dialog trees in one sentence.
```

### Serialization

```python
data = dialog.to_dict()
restored = Dialog.from_dict(data, runtime=runtime)
```

`to_dict()` records the messages, `session_name`, `owner`, the tree node, the
child dialogs (recursively), and the top prompt both as data (`top_prompt`) and
as a registry key (`top_prompt_path`). The result is JSON-safe: the top prompt's
tools are stored as schemas (implementations are not included) and a Pydantic
`format` model as its import path plus JSON schema; see
[Prompts](prompts.md#metadata-and-tracking).

```python
import json

json.dumps(dialog.to_dict())       # ready to store
```

`from_dict()` rebuilds the tree and restores `top_prompt` from the embedded
data, falling back to looking up `top_prompt_path` in the runtime (with a
warning if it is missing). Tool implementations are re-bound by name to
registered `@tool` resources when the prompt next runs.

---

## Dialog tree

### DialogTreeNode

Every `Dialog` owns exactly one `DialogTreeNode`, a lightweight record of its
position in the tree. Nodes hold ids and structure only, no message payloads,
so the topology can be rebuilt from a flat log store.

| Field | Type | Description |
| --- | --- | --- |
| `dialog_id` | `str` | Globally unique hex UUID |
| `owner` | `str \| None` | Agent that owns the dialog |
| `parent_id` | `str \| None` | Parent's `dialog_id` (`None` for a root) |
| `split_point` | `int \| None` | Number of messages copied into this child |
| `last_n` | `int \| None` | Trailing messages kept from the parent |
| `first_k` | `int \| None` | Leading messages kept from the parent |
| `children_ids` | `List[str]` | Direct children's ids |

Nodes also keep live in-process `_parent` / `_children` references for fast
traversal.

```python
node = dialog.tree_node
node.is_root        # True if no parent
node.depth          # number of forks from the root
node.subtree_ids()  # BFS over live children: all reachable ids, self included
node.to_dict()      # serializable form; DialogTreeNode.from_dict(d) restores it
```

On the dialog itself: `dialog.parent`, `dialog.children`, `dialog.is_root`, and
`dialog.depth`.

### Forking

`fork(last_n=0, first_k=1)` creates a child dialog. It deep-copies the selected
messages, links a new `DialogTreeNode` to the parent's, wires the live
parent/child references, and inherits `session_name`, `owner`, `runtime`, and
`top_prompt`.

```python
child = dialog.fork()                    # full copy
child = dialog.fork(last_n=3, first_k=1) # first message (system prompt) + last 3
child = dialog.fork(last_n=5, first_k=2) # first 2 + last 5
```

`last_n` / `first_k` are for context-window management: keep the system prompt
and the recent conversation, drop the middle. `first_k` only applies when
`last_n > 0`, and a `last_n` that covers every message gives a full copy. The
parent is never modified.

The agent loop forks a working dialog at the start of each step, so repair
prompts and failed replies stay out of the canonical dialog; only the
successful reply is appended to it.

### Visualizing the tree

```python
print(dialog.tree_overview())
```

```
[a1b2c3d4] owner=coder msgs=5 split@None
  └─ [e5f6a7b8] owner=coder msgs=5 split@5
  └─ [c9d0e1f2] owner=coder msgs=3 split@3 (last_n=2, first_k=1)
```

After an agent call, the working forks show up here, which is a quick way to
see how many repair attempts a turn needed.

---

## Agent-level dialog access

Most code works with dialogs through the agent's alias system:

```python
agent.open("planning")             # create and activate
agent.receive("What's the plan?")  # append to the active dialog
response = agent.respond()         # run the agent loop

agent.open("execution")            # second dialog
agent.switch("planning")           # switch back

agent.fork("planning", "planning_alt", last_n=2, switch=True)

agent.current_dialog               # the active Dialog
agent.dialogs                      # dict of alias -> Dialog
agent.active_alias                 # current alias
```

See [Agent](agent.md#dialog-management) for the full API.

---

## Example: a complete single-agent turn

```python
from lllm import Tactic


class Summarizer(Tactic):
    name = "summarizer"
    agent_group = ["assistant"]

    def call(self, task: str) -> str:
        agent = self.agents["assistant"]

        # 1. Open a dialog (the system prompt goes in automatically)
        agent.open("summarize", prompt_args={"persona": "helpful assistant"})

        # 2. Send the user's task
        agent.receive(task)

        # 3. Run the agent loop
        response = agent.respond()

        # 4. Inspect
        print(agent.current_dialog.cost)          # total token cost
        print(agent.current_dialog.overview())    # readable message log

        return response.content
```

```python
tactic = Summarizer({
    "agent_configs": [
        {"name": "assistant", "model_name": "gpt-4o-mini",
         "system_prompt": "You are a {persona}. Summarise what you are given."},
    ],
})
print(tactic("LLLM dialogs are append-only trees of messages..."))
```

---

## Context manager

A `ContextManager` is a hook the agent applies to the working dialog **before
each model call**. It can truncate, compress, or rewrite the dialog without
touching the canonical history.

```
[canonical dialog] --fork--> [working dialog] --context_manager--> [pruned dialog] --> model
```

The canonical dialog grows monotonically; the pruned copy is never written back.

### Contract

`ContextManager` and `DefaultContextManager` live in
`lllm.runtimes.native.core.dialog`.

```python
from abc import ABC, abstractmethod
from typing import ClassVar


class ContextManager(ABC):
    name: ClassVar[str]   # registry key: set it on every concrete subclass

    @abstractmethod
    def __call__(self, dialog: Dialog) -> Dialog:
        """Return a dialog that is safe to send to the model."""
```

`__call__` may return the same dialog unchanged or a new one (typically
`dialog.fork(...)`).

### DefaultContextManager

The built-in truncator. It uses `litellm.token_counter` (install
`lllm-core[native]`) to measure the dialog and drops or truncates old messages
until it fits the model's context window.

1. The first message (the system prompt) is always kept.
2. Messages are kept from the **tail** (most recent) inward.
3. The border message, the oldest one that partly fits, is character-truncated
   and prefixed with `[...earlier content truncated...]`.
4. A `SAFETY_BUFFER` of 5,000 tokens is subtracted from the limit so calls never
   land at the edge of the window.

```python
from lllm import Agent
from lllm.runtimes.native.core.dialog import DefaultContextManager

cm = DefaultContextManager("gpt-4o")                     # window size from LiteLLM's model table
cm = DefaultContextManager("gpt-4o", max_tokens=32000)   # explicit cap

agent = Agent(..., context_manager=cm)
```

In config:

```yaml
global:
  context_manager:
    type: default        # built-in, no registration needed
    max_tokens: 128000   # optional; looked up from LiteLLM if omitted

agent_configs:
  - name: cheap_agent
    model_name: gpt-4o-mini
    context_manager:     # deep-merged with global
      max_tokens: 16000

  - name: no_prune_agent
    model_name: gpt-4o
    context_manager:
      type: null         # disabled for this agent
```

A `context_manager` block without `type` means the default manager, so
`{max_tokens: 16000}` alone is enough; `type: null` (or no block at all)
disables pruning. When the model is unknown to LiteLLM and `max_tokens` is not
set, the window defaults to 4,096 tokens.

### Writing a custom ContextManager

1. Subclass `ContextManager` and set `name`.
2. Accept `model_name` and `max_tokens` keyword arguments; the config layer
   constructs the class with both.
3. Register the class on the runtime and reference it by name in config.

```python
from lllm import Dialog
from lllm.runtimes.native.core.dialog import ContextManager


class KeepLastTurns(ContextManager):
    """Keep the system prompt and the last `n` messages."""

    name = "keep_last"

    def __init__(self, model_name: str, max_tokens: int | None = None, n: int = 8):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.n = n

    def __call__(self, dialog: Dialog) -> Dialog:
        if len(dialog.messages) <= self.n + 1:
            return dialog
        return dialog.fork(last_n=self.n, first_k=1)
```

```python
runtime.register_context_manager(KeepLastTurns)   # key comes from KeepLastTurns.name
```

```yaml
global:
  context_manager:
    type: keep_last
```

`register_context_manager` raises `ValueError` if the class has no `name`.
