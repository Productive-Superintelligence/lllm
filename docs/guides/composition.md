# Composition

Composition is ref resolution, not service launching.

A tactic can be called:

- in process,
- through an HTTP service,
- through a `psi://...` ref resolved from local config.

```python
from lllm import TacticResolver

resolver = TacticResolver()
resolver.register("psi://demo/echo/tactics/echo", EchoTactic())

result = resolver.run("psi://demo/echo/tactics/echo", {"text": "hello"})
```

Local config can bind the same ref to a service:

```toml
[refs."psi://demo/echo/tactics/echo"]
url = "http://127.0.0.1:8000/tactics/echo"
```

`TacticResolver.from_config()` can read a shared `.psi/config.toml` that also
contains non-tactic refs from PsiHub or SSSN. It loads tactic URL bindings and
ignores refs owned by other layers, such as services, channels, snapshots,
docs, examples, and assets. Malformed tactic refs still fail validation.
Tactic ref segments are plain non-empty path segments; avoid whitespace,
percent escapes, `.`, `..`, `/`, `\`, and `:`.
Config file paths and tactic URL bindings are read exactly; do not pad them
with leading or trailing whitespace. Tactic URL bindings must be absolute
HTTP(S) service URLs.

```toml
[refs."psi://demo/echo/tactics/echo"]
url = "http://127.0.0.1:8000/tactics/echo"

[refs."psi://demo/echo/services/api"]
url = "http://127.0.0.1:8000"

[refs."psi://demo/echo/channels/events"]
store = ".sssn"
```

PsiHub validates and documents refs. It does not launch services.
