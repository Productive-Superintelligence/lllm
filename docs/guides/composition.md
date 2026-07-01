# Composition

Composition is ref resolution, not service launching.

A tactic can be called directly in process, through an HTTP service, or through
a `psi://...` ref resolved from local config. LLLM keeps those choices at the
boundary so the implementation can move without changing callers.

```python
from lllm import TacticResolver

resolver = TacticResolver()
resolver.register("psi://demo/echo/tactics/echo", EchoTactic())

result = resolver.run("psi://demo/echo/tactics/echo", {"text": "hello"})
```

## Local Service Binding

The same ref can point at a running service:

```toml
[refs."psi://demo/echo/tactics/echo"]
url = "http://127.0.0.1:8000/tactics/echo"

[refs."psi://demo/echo/tactics/echo".metadata]
policy_url = "http://127.0.0.1:9000"
```

```python
resolver = TacticResolver.from_config(".")
tactic = resolver.resolve("psi://demo/echo/tactics/echo")
```

`TacticResolver.from_config()` reads the shared `.psi/config.toml` shape used
across PSI packages. It loads tactic URL bindings and ignores refs owned by
other layers.

## Mixed Config Files

A single config file may include non-tactic refs from PsiHub or SSSN:

```toml
[refs."psi://demo/echo/tactics/echo"]
url = "http://127.0.0.1:8000/tactics/echo"

[refs."psi://demo/echo/services/api"]
url = "http://127.0.0.1:8000"

[refs."psi://demo/echo/channels/events"]
store = ".sssn"
```

LLLM uses the tactic binding. SSSN can use the channel binding. PsiHub can
generate both. No layer has to launch another layer's services.

All binding keys must still be valid `psi://` resource refs from known PSI resource sections.
Shared config may include `schemas`, `services`, `channels`,
`snapshots`, `runs`, `configs`, `docs`, `examples`, and `assets`, but malformed
refs and unknown resource sections fail validation. Tactic ref segments are
plain non-empty path segments; avoid whitespace, percent escapes, `.`, `..`,
`/`, `\`, `:`, and `;`. Config file paths and tactic URL bindings are read
exactly; do not pad them with leading or trailing whitespace.

## Targets And Metadata

A tactic ref with a concrete target must use `url`.
Tactic URL bindings must be absolute HTTP(S) service URLs, and a tactic URL binding must not also declare a `store`, `path`, or `object` target.
The `store`, `path`, or `object` target shapes belong to other layers or to direct in-process registration.

Service URL paths stay plain: avoid percent escapes, repeated slashes, dot segments, backslashes, and colons.
URL bindings must not include URL params,
query strings, fragments, embedded credentials, empty path segments, or dot
segments in URL paths.

Use `[refs."...".metadata]` for structured binding metadata. Legacy top-level
extra keys are also read as metadata, and the explicit metadata table wins on
duplicate keys. Resolver-owned `ref` and remote `url` fields remain canonical.

Binding metadata must not include raw secret-shaped keys such as
`api_key`/`apiKey`/`apikey`, tokens, `accessToken`/`accesstoken`, passwords,
cookies, `authorization`, or credentials. Use local credential refs such as
`api_key_ref`/`apiKeyRef`/`apikeyref` or auth hooks instead. Metadata maps must
use string keys; direct Python metadata with non-string keys is rejected before
Pydantic can coerce keys into text.

PsiHub validates and documents refs. It does not launch services.
