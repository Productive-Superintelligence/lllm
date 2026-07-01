# Prompt And Dialog

Native `Prompt`, `Message`, and `Dialog` preserve the core v1 mental model:
a prompt describes one behavior contract, messages record model-visible state,
and dialogs keep append-only conversation lineage.

## Prompt Contract

`Prompt` bundles template text, metadata, output parsing, tools, MCP servers,
provider addon args, repair handlers, and a renderer.

| Field | Purpose |
| --- | --- |
| `path` | Stable native prompt path and registry key. |
| `prompt` | Template text. The default renderer uses `str.format(**kwargs)`. |
| `metadata` | Versioning, provenance, experiment labels, or other JSON-like data. |
| `parser` | Callable parser or object with `parse(content, **runtime_args)`. |
| `format` | Structured output hint. Pydantic models are passed as response format when supported. |
| `function_list` | `Function` objects, tool resource refs, or tactic tool refs. |
| `mcp_servers_list` | MCP server declarations exposed as provider tools. |
| `addon_args` | Provider/runtime options such as web search or computer-use config. |
| `handler` | Repair and interrupt prompt generator. |
| `renderer` | Template renderer. Defaults to `StringFormatterRenderer`. |

```python
from lllm.runtimes.native import Prompt

prompt = Prompt(
    path="research/summarize",
    prompt="Summarize {topic} for {audience}.",
    metadata={"revision": "current"},
)

text = prompt(topic="native runtime", audience="maintainers")
assert "native runtime" in text
```

Literal braces in the default renderer must be doubled as `{{` and `}}`.
`prompt.template_vars` records required variables, and `prompt.validate_args`
returns missing variables before rendering.

## Message Model

Native `Message` preserves:

- `role`, `content`, `name`, and sanitized provider-safe names;
- text and image modalities;
- function calls and tool-call metadata;
- model id, API type, parsed output, logprobs, vectors, usage, and metadata;
- cost computed from usage fields.

The invoker writes model-facing results into messages, and the agent loop uses
message roles to distinguish final assistant responses, tool-call requests,
and tool results.

## Dialog Lineage

`Dialog` is append-only conversation state. It owns rendered messages, the top
prompt, runtime reference, session name, and tree metadata.

```python
from lllm.runtimes.native import Dialog, Prompt, Role

prompt = Prompt(path="assistant/system", prompt="You are concise.")

dialog = Dialog(owner="assistant", session_name="demo")
dialog.put_prompt(prompt, role=Role.SYSTEM, name="system")
dialog.put_text("Explain the native dialog model.", role=Role.USER)

child = dialog.fork(last_n=1, first_k=1)

assert child.parent is dialog
assert child.tree_node.parent_id == dialog.dialog_id
assert dialog.tree_node.children_ids == [child.dialog_id]
```

Forking keeps lineage through `DialogTreeNode`, including `parent_id`,
`children_ids`, `split_point`, `first_k`, `last_n`, and subtree traversal
helpers. `overview()` and `tree_overview()` are useful for debugging without
printing full messages.

## Image Messages

`Dialog.put_image()` accepts a base64 string, file path, bytes, or PIL image
when image support is installed. The LiteLLM invoker converts native image
messages into provider-compatible content parts.
