# PsiHub Metadata

LLLM exposes helper metadata for PsiHub.

```python
from lllm.integrations import tactic_resource

resource = tactic_resource(EchoTactic())
```

Metadata can include:

- tactic name and runtime kind,
- input and output JSON schemas,
- package refs and service refs,
- examples,
- custom endpoint metadata,
- descriptions and user metadata.

Package refs use `psi://.../tactics/...`; service refs use either
`psi://.../services/...` or absolute HTTP(S) service URLs without embedded
credentials, URL params, queries, fragments, percent escapes, backslashes,
colons, empty path segments, or dot segments.

`tactic_resource()` filters raw secret-shaped keys from examples and user
metadata before exporting public metadata. Keep local credential refs such as
`api_key_ref`/`apiKeyRef`/`apikeyref`; do not place raw keys, tokens,
passwords, cookies, `authorization`, credentials, camelCase variants such as
`apiKey`, `accessToken`, `clientSecret`, and `sessionCookie`, kebab-case
variants such as `set-cookie`, or collapsed lowercase variants such as
`apikey`, `accesstoken`, and `clientsecret` in exported examples or metadata.
Metadata maps must use string keys, including nested maps.

PsiHub owns `psi.toml`, package validation, local hub storage, generated cards,
agent cards, downloads, and config templates.
