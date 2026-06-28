# Native Runtime

The native runtime is preserved under `lllm.runtimes.native`.

Use it when you want transparent prompt assembly, dialog lineage, parser
objects, tool schemas, or research workflows.

```python
from lllm.runtimes.native import Dialog, Prompt, Role

system = Prompt(path="agent/system", prompt="You are a {style} assistant.")
dialog = Dialog(owner="agent")
dialog.put_prompt(system, prompt_args={"style": "careful"}, role=Role.SYSTEM)
retry = dialog.fork(last_n=1, first_k=1)
```

When native logic needs to cross the public boundary, wrap it with a tactic
adapter instead of leaking native objects into service or package APIs.
