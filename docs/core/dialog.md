# Dialog and Message

## Overview

`Dialog` is the shared workspace (blackboard) that agents, tools, and users collectively build during an agentic session. It holds the ordered list of `Message` objects, tracks which `Prompt` governs the next LLM turn (`top_prompt`), and provides serialization and branching.

`Message` is an immutable record of a single turn — content, role, sender, token usage, parsed output, tool calls, and execution metadata.

---

## Message

A `Message` captures everything about one turn in the conversation.

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
| `model_args` | `Dict[str, Any]` | Model arguments used for this call |
| `logprobs` | `List[TokenLogprob]` | Log probabilities if requested |
| `execution_errors` | `List[Exception]` | Parser or tool errors on this turn |
| `execution_attempts` | `List[Message]` | Previous failed attempts before this message |
| `metadata` | `Dict[str, Any]` | Arbitrary metadata (e.g. `tool_call_id`, `dialog_id`) |
| `api_type` | `APITypes` | `COMPLETION` or `RESPONSE` |

Key computed properties:

```python
message.cost          # InvokeCost — token counts and dollar costs
message.is_function_call  # True if the LLM requested tool calls
message.error_message     # Concatenated string of all execution_errors
message.sanitized_name    # name safe for API submission (alphanumeric + _ -)
```

Messages are Pydantic models. Use `message.to_dict()` / `Message.from_dict(d)` for serialization. `raw_response` and `execution_errors` are excluded from serialization (not reconstructable).

---

## Dialog

### Construction

Dialogs are always started through `Agent.start_dialog()`, which creates an empty dialog and immediately puts the system prompt as the first message:

```python
dialog = agent.start_dialog(prompt_args={'language': 'Python'}, session_name='run_001')
```

Direct construction is reserved for internal use (forking, deserialization).

### The `top_prompt` calling convention

`top_prompt: Optional[Prompt]` is the active calling convention — the `Prompt` that governs parser, tools, and handler behavior for the next LLM turn. It is set exclusively by `put_prompt`. It is `None` after `put_text` (no active contract) or before any prompt has been issued.

The agent loop snapshots `top_prompt` at call entry:

```python
task_prompt = dialog.top_prompt  # stable for the entire agent.call()
```

This means `put_prompt` calls during tool result delivery update `dialog.top_prompt` freely without disrupting the loop's handler dispatch.

### Appending messages

Three primitives cover all cases:

**`put_prompt(prompt, prompt_args, name, metadata, role)`** — stateful. Renders the prompt template, appends the message, and advances `top_prompt`. Use for any message that carries a calling convention: the initial task, handler-returned prompts, tool result prompts.

```python
dialog.put_prompt(my_prompt, {'task': task_str}, name='user')
dialog.put_prompt(task_prompt.on_interrupt(call_state), {'call_results': result}, role=Roles.TOOL, name=fn_name, metadata={'tool_call_id': fn_call.id})
```

`prompt` can be a `Prompt` object or a registered path string — the dialog resolves it via context:

```python
dialog.put_prompt('research/agent', {'topic': 'transformers'})
```

**`put_text(text, name, metadata, role)`** — stateless. Appends raw text, resets `top_prompt` to `None`. Use for unstructured user input with no output contract.

```python
dialog.put_text("What is the capital of France?")
```

**`put_image(image, caption, name, metadata, role)`** — stateless. Appends an image message, does not touch `top_prompt`. `image` accepts a file path, `Path`, PIL `Image`, or a base64 string.

```python
dialog.put_image('screenshot.png', caption='Current state of the UI')
dialog.put_image(pil_image)
```

All three route through `dialog.append()`, which stamps `dialog_id` onto the message metadata and writes to the replay log if one is configured.

### Forking

`fork(n_messages=0)` deep-copies the dialog, creating a new branch that shares `session_name` and `log_base` but has its own `dialog_id` and `parent_dialog` pointer.

```python
working_dialog = dialog.fork()          # full copy
working_dialog = dialog.fork(n_messages=2)  # drop last 2 messages
```

The agent loop forks at the start of each interrupt iteration so that exception-recovery messages are isolated and never appear in the canonical dialog. Only the final successful response is appended to the original dialog.

### Inspection

```python
dialog.tail      # last Message, or None
dialog.system    # first Message (system prompt)
dialog.messages  # full list, read-only by convention
dialog.cost      # InvokeCost — aggregate across all messages
dialog.overview(max_length=100, stream=st)  # human-readable summary
```

### Serialization

```python
d = dialog.to_dict()
dialog = Dialog.from_dict(d, log_base=log_base, context=ctx)
```

`to_dict` records `top_prompt_path` (the registered path of the active prompt). `from_dict` resolves it from the context registry. If the path is not found (e.g. a session from an older code version), `top_prompt` is set to `None` with a warning rather than raising.

---

## Dialog as a tree

Each `Dialog` carries a `dialog_id` (hex UUID) and a `parent_dialog` pointer. Forking creates a directed tree where the root is the session's initial dialog, branches are speculative or recovery contexts, and the canonical path is the sequence of dialogs that received `dialog.append(response)` after successful LLM turns.

This structure is what makes the replay log navigable: every message records its `dialog_id` in metadata, so the full branching history of a session can be reconstructed from the log.

---

## Example: a complete single-agent turn

```python
# 1. Start the dialog (system prompt goes in automatically)
dialog = agent.start_dialog({'persona': 'helpful assistant'})

# 2. Put the user's task
dialog.put_prompt('tasks/summarize', {'document': doc_text}, name='user')

# 3. Optionally attach supplementary content
dialog.put_image(screenshot_path, caption='Referenced figure')

# 4. Run the agent loop
response, dialog, call_state = agent.call(dialog)

# 5. Inspect results
print(response.parsed)   # structured output from the prompt's parser
print(dialog.cost)       # total token cost for the session
print(dialog.overview()) # human-readable message log
```


## Context Manager

TODO
