# Hub metadata

Helpers that export what a tactic or channel publishes, in the shape the
hub indexes. Use them to fill `metadata` in a package's
[`lllm.toml`](../packages.md#the-manifest) or to build catalog entries.

```python title="tactic_resource.py"
from pydantic import BaseModel

from lllm import Tactic
from lllm.hub import tactic_resource


class Ticket(BaseModel):
    text: str


class Triage(Tactic):
    """Route a support ticket to a queue."""

    name = "triage"
    input_model = Ticket

    def call(self, ticket: Ticket) -> str:
        return "oncall" if "outage" in ticket.text.lower() else "support"


triage = Triage(
    package_ref="lllm://acme/support/tactics/triage",
    metadata={"owner": "support", "api_key": "sk-123", "token_limit": 5},
)
resource = tactic_resource(triage)
print(resource["name"], resource["runtime"], resource["package_ref"])
#> triage native lllm://acme/support/tactics/triage
print(resource["metadata"])
#> {'owner': 'support', 'token_limit': 5}
```

`tactic_resource(tactic)` returns:

| Key | Content |
| --- | --- |
| `name`, `description`, `runtime` | From `tactic.info()` |
| `capabilities` | `run`, `arun`, `stream`, `call`, ... |
| `endpoints` | Custom `@endpoint` routes: name, method, path, mode, description, tags |
| `input_schema`, `output_schema` | JSON schemas of the typed input and output |
| `package_ref`, `service_ref` | `lllm://` refs or a service URL |
| `examples`, `metadata` | Public examples and metadata, without secret-shaped keys |

`channel_resource(channel)` and `snapshot_resource(snapshot)`, from the
[SSSN](https://sssn.one/) package, do the same for SSSN channels and
snapshots, including custom store endpoints.

## Rules for published metadata

Package refs use `lllm://org/package/tactics/name`. Service refs use either
`lllm://org/package/services/name` or an absolute HTTP(S) URL without
embedded credentials, params, queries, fragments, percent escapes,
backslashes, colons, empty segments, or dot segments.

`tactic_resource` drops secret-shaped keys from examples and metadata: API
keys, tokens, passwords, secrets, cookies, `authorization`, credentials, and
private keys, in snake_case, kebab-case, camelCase, and collapsed forms
(`api_key`, `x-api-key`, `apiKey`, `apikey`, `accessToken`, `clientSecret`,
`set-cookie`, `sessionCookie`, `proxyAuthorization`, `awsCredentials`,
`privateKey`, ...). A token key is `token` or ends in `token`
(`accessToken`, `eos_token`), so `token_limit` is kept. References to secrets
are kept: `api_key_ref`, `apiKeyRef`, `apikeyref`. `channel_resource` and
`snapshot_resource` copy metadata unchanged. Metadata maps must use string
keys, including nested maps.

In `lllm.toml` these keys are errors, not dropped: the package ships the file
as written, so `lllm hub validate` reports `metadata_secret_key` for them in
the `metadata` and extra keys of named resource tables, `[config]`, and `[card]`
([How metadata is checked](../publishing.md#how-metadata-is-checked)).
Validation does not look inside maps under JSON-schema keys (`schema`,
`input_schema`, `output_schema`, or forms such as `inputSchema`), where a
field may be named `api_key`, or inside `[config] defaults` and tactic
`examples`.

Cards, the hub API, and the static catalog drop these keys from resource
metadata, including `[config] defaults` and tactic `examples`. They also drop
any key with a `token` word part, such as `token_limit`, which validation
allows. Inside schema maps, cards and the catalog still drop secret-shaped
keys, but the hub API returns those maps unchanged. The package zip contains
`lllm.toml` as written, so every key validation allows ships in it.
