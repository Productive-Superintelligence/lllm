# Service API

The HTTP API of a served tactic. `create_tactic_app` and `lllm serve` build
it; [Serving](../serving.md) explains how to use it.

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

## Requests

`POST /run`, `POST /stream`, and custom endpoints read a `RunRequest`
envelope:

| Field | Type | Meaning |
| --- | --- | --- |
| `input` | any JSON | The tactic's input, validated against its input schema |
| `task` | any JSON | Accepted in place of `input` |
| `context` | object or `null` | Request metadata, read into a `CallContext` |

A body is an envelope when it is a JSON object with `input` or `task` and
no keys besides `input`, `task`, and `context`. Any other body, JSON or
text, is the input itself.

| `context` key | Meaning |
| --- | --- |
| `request_id` (or `call_id`) | Generated when missing or empty |
| `trace_id`, `span_id`, `caller` | Strings for tracing |
| `tags` | String-to-string object |
| `metadata` | Object with string keys |
| `package_ref`, `service_ref` | Refs of the caller's package and service |
| `tactic_ref` | Defaults to the tactic's `package_ref` |

The service sets `endpoint` to `run`, `stream`, or the custom endpoint's
name, and ignores other keys. A context that is not an object, or whose
`tags` or `metadata` is not an object, is a `SchemaError`.

## Responses

A successful `/run` returns a `RunResponse`:

```json
{"output": {"queue": "billing"}, "request_id": "req-1", "tactic": "triage"}
```

`/stream` returns `text/event-stream`, one `data:` line per event. Each
event is a `TacticEvent` as JSON: `id`, `timestamp`, `kind` (`message`,
`result`, `error`, or a kind the tactic chooses), `data`, and `metadata`.
A custom endpoint in `mode="stream"` or `mode="events"` sends events as
`TacticEvent` JSON and other items as plain JSON. A failure after the
stream has started is a final `error` event, with `error_type`, `tactic`,
`endpoint`, and `request_id` in its metadata, while the HTTP status stays
200.

## Custom endpoints

Custom endpoints declared with `@endpoint.*` are mounted alongside these
portable routes. They must use unique method/path pairs and must not shadow
reserved LLLM service routes such as `/run`, `/stream`, `/info`, or
`/tactics/{name}/run`. Endpoint paths, names, and tags must avoid whitespace
and percent escapes; paths must also avoid `//` network-path prefixes, empty
or dot segments, backslashes, colons, semicolon path params, query strings, and
fragments. A `GET` or `DELETE` endpoint called without a body receives the
query parameters as a dict (or `None`) instead of a validated input.

## Public metadata

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

## Errors

Error envelopes are stable across protocol and runtime failures.

```json
{
  "detail": {
    "error": {
      "type": "SchemaError",
      "message": "Value does not match schema Ticket: ...",
      "tactic": "triage",
      "endpoint": "run",
      "request_id": "req-8",
      "metadata": {}
    }
  }
}
```

| Status | `error.type` | When |
| --- | --- | --- |
| 400 | `SchemaError` | Invalid input, or a malformed envelope or context |
| 400 | `TacticUnsupportedError` | `/stream` on a tactic that does not stream |
| 404 | `TacticNotFound` | An unknown name on a `/tactics/{name}/...` route |
| 500 | `OutputSchemaError` | The tactic's result does not match its output schema; `message` starts with `The output of tactic '<name>' failed validation:` |
| 500 | The exception's class name | Anything else the tactic raises; `message` is `str(exc)` |

`status_code_for_error(exc)` gives the status for an exception, and
`RemoteTactic` raises `RemoteTacticError` with these fields
([Serving](../serving.md#calling-a-served-tactic-from-python)). The
[AAAX](https://aaax.one/) strategy routes report errors in the same envelope.
