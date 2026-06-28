# PsiHub Metadata

LLLM exposes helper metadata for PsiHub.

```python
from lllm.integrations import tactic_resource

resource = tactic_resource(EchoTactic())
```

Metadata can include:

- tactic name and runtime kind,
- input and output schema refs,
- package and service refs,
- examples,
- custom endpoint metadata,
- descriptions and user metadata.

PsiHub owns `psi.toml`, package validation, local hub storage, generated cards,
agent cards, downloads, and config templates.
