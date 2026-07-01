# Native Service Example

This example serves an offline native prompt/dialog workflow through the
FastAPI tactic API.

```bash
pip install -e ".[dev,server]"
uvicorn app:app --app-dir examples/native_service --reload
curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input":{"topic":"native services","audience":"operators"},"context":{"trace_id":"trace-demo"}}'
```

The native tactic records a `Dialog` transcript internally, while
`NativeTacticAdapter` exposes typed input/output schemas, package metadata, and
the service boundary.
