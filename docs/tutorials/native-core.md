# Native Core

Goal: use native prompt and dialog primitives without coupling callers to a
model provider or agent loop.

```python
from lllm.runtimes.native import Dialog, Prompt, Role


system = Prompt(path="agent/system", prompt="You are a {style} assistant.")
task = Prompt(path="agent/task", prompt="Plan the next checkpoint for {project}.")

dialog = Dialog(owner="agent")
dialog.put_prompt(system, prompt_args={"style": "careful"}, role=Role.SYSTEM)
dialog.put_prompt(task, prompt_args={"project": "LLLM v2"})

retry = dialog.fork(last_n=1, first_k=1)
```

`Dialog` keeps append-only messages and serializable lineage metadata. `fork()`
copies the selected context into a child dialog and links the child back to its
parent.

```python
assert retry.parent is dialog
assert retry.depth == 1
assert retry.tree_node.parent_id == dialog.dialog_id
```

Tools are schema records with optional local Python implementations:

```python
from lllm.runtimes.native import FunctionCall, tool


@tool(description="Add two numbers")
def add(left: int, right: int) -> int:
    return left + right


call = add(FunctionCall(name="add", arguments={"left": 2, "right": 3}))
assert call.result == 5
```

Prompts can also carry small parser objects. The default tag parser extracts
XML blocks, fenced markdown blocks, and signal tags without depending on a
model provider:

```python
from lllm.runtimes.native import DefaultTagParser, Prompt


prompt = Prompt(
    path="agent/parse",
    prompt="Return <answer>...</answer> and a ```json block.",
    parser=DefaultTagParser(
        required_xml_tags=["answer"],
        required_md_tags=["json"],
        signal_tags=["DONE"],
    ),
)

parsed = prompt.parse("<answer>Hello</answer>\n```json\n{}\n```\n<DONE>")
assert parsed["signal_tags"]["DONE"] is True
```

When native objects need to be reused outside this runtime, expose them through
`NativeTacticAdapter` so remote callers only depend on the `Tactic` protocol.
