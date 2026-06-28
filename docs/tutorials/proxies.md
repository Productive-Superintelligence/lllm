# Proxies

Goal: observe or lightly transform tactic calls without depending on a specific
runtime.

```python
from lllm import CallContext, InMemoryProxyLog, ProxyTactic


log = InMemoryProxyLog()
proxy = ProxyTactic(
    tactic,
    sink=log.append,
    capture_inputs=True,
    capture_outputs=True,
)

result = proxy.run(
    {"text": "hello"},
    context=CallContext(request_id="demo-1"),
)

assert log.records[0].request_id == "demo-1"
```

Proxy hooks run at the `Tactic` boundary, so the same wrapper can sit around
plain Python, Pydantic AI, native, remote, or service-hosted tactics.

```python
def before(value, context):
    context.metadata["seen_by"] = "proxy"
    return value


def after(value, context):
    return value


proxy = ProxyTactic(tactic, before=before, after=after)
```

Payload capture is opt-in. By default, proxy records include timing, state,
request ID, tactic names, errors, and metadata without storing inputs or
outputs.
