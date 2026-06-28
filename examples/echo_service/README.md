# Echo Service

Run the example:

```bash
uvicorn app:app --reload
```

Call it:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input":{"text":"hello"}}'
```
