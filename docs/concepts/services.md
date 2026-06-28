# Services

LLLM services expose tactics through FastAPI.

The portable path is:

```text
POST /run
GET  /info
POST /stream
```

`/run` accepts either an envelope:

```json
{
  "input": {"text": "hello"},
  "context": {"trace_id": "demo"}
}
```

or raw app JSON when preserving an existing client shape:

```json
{"text": "hello"}
```

Service errors use stable envelopes so callers can inspect `type`, `message`,
`tactic`, `endpoint`, and `request_id` without scraping framework text.
