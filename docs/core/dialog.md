# Dialog and Message

## Overview

`Dialog` is the shared workspace that agents, tools, and users collectively build during an agentic session. It holds an ordered list of `Message` objects, tracks which `Prompt` governs the next LLM turn (`top_prompt`), and provides forking for branching conversations.

Each `Dialog` is owned by a single agent and carries a `DialogTreeNode` that records its position in a dialog tree — enabling branching, speculation, and full session reconstruction from logs.

`Message` is a clean conversational record of a single turn. Per-invocation diagnostics (raw LLM responses, retry attempts, model args) live on `InvokeResult` and `AgentCallSession`, not on the message.

---

## Message

A `Message` captures what was said in one turn — nothing more.

| Field | Type | Description |
|---|---|---|
| `role` | `Roles` | `SYSTEM`, `USER`, `ASSISTANT`, `TOOL`, `TOOL_CALL` |
| `content` | `str \| List[Dict]` | Text or multipart content (images) |
| `name` | `str` | Sender identifier (e.g. `'user'`, `'coder'`, a tool name) |
| `modality` | `Modalities` | `TEXT` or `IMAGE` |
| `function_calls` | `List[FunctionCall]` | Tool calls requested by the LLM |
| `parsed` | `Dict[str, Any]` | Structured output from the prompt's parser |
| `usage` | `Dict[str, Any]` | Raw token usage from the provider |
| `model` | `str` | Model that produced this message |
| `logprobs` | `List[TokenLogprob]` | Log probabilities if requested |
| `metadata` | `Dict[str, Any]` | Arbitrary metadata (e.g. `tool_call_id`, `dialog_id`) |
| `api_type` | `APITypes` | `COMPLETION` or `RESPONSE` |
| `vectors` | `List[float]` | Placeholder for embedding vectors |

Key computed properties:

```python
message.cost              # InvokeCost — token counts and dollar costs
message.is_function_call  # True if the LLM requested tool calls
message.sanitized_name    # name safe for API submission (alphanumeric + _ -)
```

Messages are Pydantic models. `message.to_dict()` / `Message.from_dict(d)` round-trip cleanly with no exclusions — every field serializes.

---

## Dialog

### Construction

Dialogs are created through `Agent.open()`, which creates a dialog, seeds it with the system prompt, and registers it under an alias:

```python
agent.open('planning', prompt_args={'language': 'Python'}, session_name='run_001')
```

Direct construction is reserved for internal use (forking, deserialization).

### Ownership

Every dialog is owned by the agent that created it. The `owner` field records the agent name, and the system prompt message is always from that agent. This prevents accidental cross-agent dialog mix-ups — if `agentB` tried to use `agentA`'s dialog, the system prompt would carry `agentA`'s persona, which is a subtle bug.

### The `top_prompt` calling convention

`top_prompt: Optional[Prompt]` is the active calling convention — the `Prompt` that governs parser, tools, and handler behavior for the next LLM turn. It is set by `put_prompt` and reset to a temporary prompt by `put_text`.

The agent loop snapshots `top_prompt` at call entry:

```python
task_prompt = dialog.top_prompt  # stable for the entire agent call
```

This means `put_prompt` calls during tool result delivery update `dialog.top_prompt` freely without disrupting the loop's handler dispatch.

### Appending messages

Three primitives cover all cases:

**`put_prompt(prompt, prompt_args, name, metadata, role)`** — stateful. Renders the prompt template, appends the message, and advances `top_prompt`. Use for any message that carries a calling convention.

```python
dialog.put_prompt(my_prompt, {'task': task_str}, name='user')
dialog.put_prompt('research/agent', {'topic': 'transformers'})  # path string resolved via runtime
```

**`put_text(text, name, metadata, role)`** — stateless. Appends raw text. Use for unstructured user input.

```python
dialog.put_text("What is the capital of France?")
```

**`put_image(image, caption, name, metadata, role)`** — stateless. Appends an image message. Accepts a file path, `Path`, PIL `Image`, or base64 string.

```python
dialog.put_image('screenshot.png', caption='Current state of the UI')
```

All three route through `dialog.append()`, which stamps `dialog_id` onto the message metadata and writes to the replay log if configured.

### Inspection

```python
dialog.tail       # last Message, or None
dialog.head       # first Message (system prompt)
dialog.messages   # full list, read-only by convention
dialog.cost       # InvokeCost — aggregate across all messages
dialog.overview() # human-readable summary
```

### Serialization

```python
d = dialog.to_dict()
dialog = Dialog.from_dict(d, log_base=log_base, runtime=runtime)
```

`to_dict` records the `tree_node` (dialog tree metadata) and `top_prompt_path`. `from_dict` resolves the prompt path from the runtime registry, setting `top_prompt` to `None` with a warning if not found.

---

## Dialog Tree

### DialogTreeNode

Every `Dialog` owns exactly one `DialogTreeNode` — a lightweight record of its position in the dialog tree. The node carries only ids and structural metadata, no message payloads, so the full tree topology can be reconstructed from a flat log store.

| Field | Type | Description |
|---|---|---|
| `dialog_id` | `str` | Globally unique hex UUID |
| `owner` | `str` | Agent name that created this dialog |
| `parent_id` | `str \| None` | Parent's dialog_id (None for root) |
| `split_point` | `int \| None` | Message count kept from parent |
| `last_n` | `int \| None` | How many trailing messages were preserved |
| `first_k` | `int \| None` | How many leading messages were preserved |
| `children_ids` | `List[str]` | Direct children's dialog_ids |

In addition to the serializable id fields, each node holds live in-process `_parent` and `_children` references for fast tree traversal without deserialization.

```python
node = dialog.tree_node
node.is_root        # True if no parent
node.depth          # number of forks from root
node.subtree_ids()  # BFS: all reachable dialog_ids including self
node.to_dict()      # serialize for logging
```

### Forking

`fork(last_n=0, first_k=1)` creates a child dialog branching from the current one. It deep-copies messages, creates a linked `DialogTreeNode`, and wires both node-level and dialog-level parent/child references.

```python
# Full copy
child = dialog.fork()

# Keep first message (system prompt) + last 3 messages
child = dialog.fork(last_n=3, first_k=1)

# Keep first 2 messages + last 5
child = dialog.fork(last_n=5, first_k=2)
```

The `last_n` / `first_k` parameters are useful for context window management — keeping the system prompt and recent conversation while dropping the middle. If `last_n + first_k >= len(messages)`, all messages are preserved (equivalent to a full copy).

The agent call loop forks at the start of each interrupt iteration so that exception-recovery messages are isolated. Only the final successful response is appended to the canonical dialog.

### Visualizing the tree

```python
print(dialog.tree_overview())
```

```
[a1b2c3d4] owner=coder msgs=5 split@None
  └─ [e5f6g7h8] owner=coder msgs=5 split@5
  └─ [i9j0k1l2] owner=coder msgs=3 split@3 (last_n=2, first_k=1)
```

---

## Agent-Level Dialog Access

While `Dialog` is the low-level data structure, developers typically interact with dialogs through the `Agent`'s alias system:

```python
agent.open('planning')           # create and activate
agent.receive("What's the plan?") # append to active dialog
response = agent.respond()        # run agent loop

agent.open('execution')           # create second dialog
agent.switch('planning')          # switch back

# Fork for exploration
agent.fork('planning', 'planning_alt', last_n=2, switch=True)

# Access dialogs
agent.current_dialog              # the active Dialog object
agent.dialogs                     # dict of alias → Dialog
agent.active_alias                # current alias string
```

See [Agent](./agent.md) for the full agent API.

---

## Example: Complete Single-Agent Turn

```python
class SummarizeAgent(Tactic):
    tactic_type = "summarizer"
    agent_group = ["assistant"]
    
    def call(self, task: str, **kwargs):
        agent = self.agents["assistant"]
        
        # 1. Open dialog (system prompt goes in automatically)
        agent.open('summarize', prompt_args={'persona': 'helpful assistant'})
        
        # 2. Send the user's task
        agent.receive(task)
        
        # 3. Run the agent loop
        response = agent.respond()
        
        # 4. Inspect results
        print(response.parsed)                    # structured output
        print(agent.current_dialog.cost)          # total token cost
        print(agent.current_dialog.overview())    # human-readable message log
        
        return response.content
```

---

## Context Manager

`ContextManager` is an abstract hook that the agent applies to the working dialog **before each LLM call**. It can truncate, compress, rewrite, or otherwise transform the dialog without touching the canonical history.

```
[canonical dialog] ──fork──▶ [working dialog] ──context_manager──▶ [pruned dialog] ──▶ LLM
```

The canonical dialog always grows monotonically. The pruned copy is ephemeral — it is never written back.

### ContextManager contract

```python
class ContextManager(ABC):
    name: ClassVar[str]   # registry key — must be set on every concrete subclass

    @abstractmethod
    def __call__(self, dialog: Dialog) -> Dialog:
        """Return a dialog that is safe to send to the LLM."""
```

`__call__` may return the same `dialog` object unchanged (when no pruning is needed) or a new child dialog.

---

### DefaultContextManager

The built-in truncator. Uses `litellm.token_counter` to measure the dialog's token footprint and drops/truncates old messages until it fits within the model's context window.

**Truncation strategy:**

1. The first message (system / developer prompt) is always preserved.
2. Messages are retained from the **tail** (most recent) inward.
3. The *border* message — the oldest one that partially fits — is character-truncated with a `[...earlier content truncated...]` prefix.
4. A `SAFETY_BUFFER` of 5 000 tokens is subtracted from the context limit so invocations never land right at the model edge.

**Programmatic usage:**

```python
from lllm.core.dialog import DefaultContextManager
from lllm.core.agent import Agent

cm = DefaultContextManager("gpt-4o")               # context window auto-detected from litellm
cm = DefaultContextManager("gpt-4o", max_tokens=32000)  # manual cap

agent = Agent(..., context_manager=cm)
```

**Config (YAML):**

```yaml
# applies to every agent in the tactic
global:
  context_manager:
    type: default        # built-in — no registration needed
    max_tokens: 128000   # optional; auto-detected from litellm if omitted

agent_configs:
  - name: cheap_agent
    model_name: gpt-4o-mini
    context_manager:     # per-agent override (deep-merged with global)
      type: default
      max_tokens: 16000

  - name: no_prune_agent
    model_name: gpt-4o
    context_manager:     # explicitly disable for this agent
      type: null
```

`type: null` (or omitting `context_manager` entirely) disables pruning for that agent.

---

### Writing a custom ContextManager

1. Subclass `ContextManager` and set the `name` class attribute.
2. Register the class with the runtime.
3. Reference it by name in config.

```python
from lllm.core.dialog import ContextManager, Dialog

class SummaryCompressor(ContextManager):
    name = "summary"   # config type key

    def __init__(self, model_name: str, max_tokens: int = None):
        self.model_name = model_name
        self._max_tokens = max_tokens

    def __call__(self, dialog: Dialog) -> Dialog:
        # ... compress older messages into a rolling summary, return new dialog ...
        return compressed_dialog
```

Register it so the config layer can find it:

```python
runtime.register_context_manager(SummaryCompressor)
# reads SummaryCompressor.name ("summary") as the registry key
```

Config:

```yaml
global:
  context_manager:
    type: summary
    max_tokens: 64000
```

`register_context_manager` raises `ValueError` if the class does not define a `name` attribute.