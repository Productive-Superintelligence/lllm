# PsiHub Metadata

LLLM exposes helper metadata for PsiHub.

```python
from lllm.integrations import tactic_resource

resource = tactic_resource(EchoTactic())
```

Metadata can include:

- tactic name and runtime kind,
- input and output JSON schemas,
- package and service refs,
- examples,
- custom endpoint metadata,
- descriptions and user metadata.

`tactic_resource()` filters raw secret-shaped keys from examples and user
metadata before exporting public metadata. Keep local credential refs such as
`api_key_ref`/`apiKeyRef`; do not place raw keys, tokens, passwords,
`authorization`, credentials, or camelCase variants such as `apiKey`,
`accessToken`, and `clientSecret` in exported examples or metadata.

PsiHub owns `psi.toml`, package validation, local hub storage, generated cards,
agent cards, downloads, and config templates.
