# Service API

The FastAPI adapter exposes a tactic through portable endpoints.

| Endpoint | Use |
| --- | --- |
| `GET /info` | Return service-advertised `TacticInfo`. |
| `POST /run` | Run the tactic with envelope or raw JSON input. |
| `POST /stream` | Stream tactic output when supported. |

Custom endpoints declared with `@endpoint.*` are mounted alongside these
portable routes. They must use unique method/path pairs and must not shadow
reserved LLLM service routes such as `/run`, `/stream`, `/info`, or
`/tactics/{name}/run`. Endpoint paths, names, and tags must avoid whitespace
and percent escapes; paths must also avoid `//` network-path prefixes.

Public info endpoints filter raw secret-shaped keys from tactic examples and
user metadata while preserving local credential refs such as
`api_key_ref`/`apiKeyRef`/`apikeyref`. Snake_case, kebab-case, camelCase, and
collapsed lowercase secret keys such as `api_key`, `x-api-key`, `apiKey`,
`apikey`, `accessToken`, `accesstoken`, `clientSecret`, and `clientsecret` are
treated consistently. Cookie-style keys such as `cookie`, `set-cookie`, and
`sessionCookie` are treated as secret-shaped metadata.
SSE stream event metadata is filtered with the same public boundary before it
is written to the response; event data remains the tactic output.

Error envelopes are stable across protocol and runtime failures.

```json
{
  "detail": {
    "error": {
      "type": "SchemaError",
      "message": "Invalid input.",
      "tactic": "echo",
      "endpoint": "run",
      "request_id": "...",
      "metadata": {}
    }
  }
}
```
