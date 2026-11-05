# Hub Metadata

Helpers that export what a tactic or channel publishes, in the shape the hub
indexes. Use them to fill `metadata` in [lllm.toml](../packages/manifest.md)
or to build catalog entries.

```python
from lllm.hub import tactic_resource
from lllm.network import Channel, channel_resource, snapshot_resource

resource = tactic_resource(EchoTactic())
channel = channel_resource(Channel(name="events", form="log", description="Raw events."))
```

`tactic_resource(tactic)` returns:

| Key | Content |
| --- | --- |
| `name`, `description`, `runtime` | From `tactic.info()` |
| `capabilities` | `run`, `arun`, `stream`, `call`, ... |
| `endpoints` | Custom `@endpoint` routes: name, method, path, mode, tags |
| `input_schema`, `output_schema` | JSON schemas of the typed input and output |
| `package_ref`, `service_ref` | `lllm://` refs or a service URL |
| `examples`, `metadata` | Public examples and metadata |

`channel_resource` and `snapshot_resource` do the same for SSSN channels and
snapshots, including custom store endpoints.

## Rules for published metadata

Package refs use `lllm://org/package/tactics/name`. Service refs use either
`lllm://org/package/services/name` or an absolute HTTP(S) URL without
embedded credentials, params, queries, fragments, percent escapes,
backslashes, colons, empty segments, or dot segments.

Exported examples and metadata drop secret-shaped keys: API keys, tokens,
passwords, cookies, `authorization`, and credentials, in snake_case,
kebab-case, camelCase, and collapsed forms (`api_key`, `x-api-key`, `apiKey`,
`apikey`, `accessToken`, `clientSecret`, `set-cookie`, `sessionCookie`, ...).
References to secrets are kept: `api_key_ref`, `apiKeyRef`, `apikeyref`.
Metadata maps must use string keys, including nested maps.
