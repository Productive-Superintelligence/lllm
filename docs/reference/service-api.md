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
and percent escapes.

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
