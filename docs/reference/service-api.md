# Service API

The FastAPI adapter exposes a tactic through portable endpoints.

| Endpoint | Use |
| --- | --- |
| `GET /info` | Return service-advertised `TacticInfo`. |
| `POST /run` | Run the tactic with envelope or raw JSON input. |
| `POST /stream` | Stream tactic output when supported. |

Error envelopes are stable across protocol and runtime failures.

```json
{
  "detail": {
    "type": "TacticInputError",
    "message": "Invalid input.",
    "tactic": "echo",
    "endpoint": "run",
    "request_id": "..."
  }
}
```
