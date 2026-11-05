# Service API

The FastAPI adapter exposes a tactic through portable endpoints.

| Endpoint | Use |
| --- | --- |
| `GET /health` | `{"ok": true, "tactics": [...]}`. |
| `GET /info` | Return service-advertised `TacticInfo`. |
| `POST /run` | Run the tactic with envelope or raw JSON input. |
| `POST /stream` | Stream tactic output when supported. |
| `GET /tactics`, `GET /tactics/{name}/info` | `TacticInfo` for every tactic, or for one. |
| `POST /tactics/{name}/run`, `POST /tactics/{name}/stream` | Run or stream one tactic. |

`create_tactic_app` mounts all of them. `create_service_app` mounts `/health`
and the `/tactics/...` routes, and `/info`, `/run`, and `/stream` too when it
holds one tactic and `expose_single_tactic_routes=True`. `/openapi.json`
documents the tactic's input schema in the `/run` and `/stream` request
envelopes and its output schema in the `/run` response.

Custom endpoints declared with `@endpoint.*` are mounted alongside these
portable routes. They must use unique method/path pairs and must not shadow
reserved LLLM service routes such as `/run`, `/stream`, `/info`, or
`/tactics/{name}/run`. Endpoint paths, names, and tags must avoid whitespace
and percent escapes; paths must also avoid `//` network-path prefixes, empty
or dot segments, backslashes, colons, semicolon path params, query strings, and
fragments. A `GET` or `DELETE` endpoint called without a body receives the
query parameters as a dict (or `None`) instead of a validated input.

Public info endpoints filter raw secret-shaped keys from tactic examples and
user metadata while preserving local credential refs such as
`api_key_ref`/`apiKeyRef`/`apikeyref`. Snake_case, kebab-case, camelCase, and
collapsed lowercase secret keys such as `api_key`, `x-api-key`, `apiKey`,
`apikey`, `accessToken`, `accesstoken`, `clientSecret`, and `clientsecret` are
treated consistently. Cookie-style keys such as `cookie`, `set-cookie`, and
`sessionCookie` are treated as secret-shaped metadata.
SSE stream event metadata is filtered with the same public boundary before it
is written to the response; event data remains the tactic output.
Metadata maps must use string keys, including nested maps.

Error envelopes are stable across protocol and runtime failures.

```json
{
  "detail": {
    "error": {
      "type": "SchemaError",
      "message": "Value does not match schema Question: ...",
      "tactic": "shout",
      "endpoint": "run",
      "request_id": "...",
      "metadata": {}
    }
  }
}
```

`SchemaError` and `TacticUnsupportedError` are 400 responses, an unknown name
on a `/tactics/{name}/...` route is a 404 `TacticNotFound`, and anything else
a tactic raises is a 500 ([Services](../protocol/services.md#errors)). The
AAAX shell routes report errors in the same envelope.
