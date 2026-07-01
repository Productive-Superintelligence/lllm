# Services

LLLM services expose tactics through FastAPI. A service is the network face of
a tactic, not a separate execution model.

The portable endpoints are:

| Endpoint | Use |
| --- | --- |
| `POST /run` | Execute one tactic call and return one output. |
| `POST /stream` | Execute a streaming call as server-sent events. |
| `GET /info` | Return `TacticInfo` without running the tactic. |

`/run` accepts the protocol envelope:

```json
{
  "input": {"text": "hello"},
  "context": {"trace_id": "demo"}
}
```

It can also accept raw app JSON when adapting an existing client shape:

```json
{"text": "hello"}
```

The envelope is preferred for new integrations because it leaves room for
request ids, trace ids, caller metadata, and future routing fields.

## Create An App

```python
from lllm.services import create_tactic_app
from demo.tactics import EchoTactic

app = create_tactic_app(EchoTactic())
```

Run it:

```bash
uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Then call it:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input":{"text":"hello"}}'
```

## Errors

Service errors use stable envelopes so callers can inspect `type`, `message`,
`tactic`, `endpoint`, `request_id`, and `metadata` without scraping framework
text. Validation failures, tactic failures, and endpoint conflicts remain
machine-readable across local and remote calls.

Custom endpoints can be mounted beside the portable API, but they must avoid
reserved LLLM service routes and keep method/path pairs unique. Endpoint paths,
names, and tags are validated as portable text so package metadata and docs can
refer to them safely.

## Remote Clients

`RemoteTactic` normalizes a service base URL into `/run`, `/stream`, and
`/info` calls. Use:

- `RemoteTactic.arun()` for JSON run calls,
- `RemoteTactic.astream()` for streamed data chunks,
- `RemoteTactic.aevents()` for full event envelopes,
- `RemoteTactic.fetch_info()` / `afetch_info()` for remote metadata.

This lets local tactics and HTTP-backed tactics share the same composition
layer.
