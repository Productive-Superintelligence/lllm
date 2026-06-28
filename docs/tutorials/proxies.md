# Proxies

Goal: observe or lightly transform tactic calls without depending on a specific
runtime.

## Prerequisites

```bash
python -m pip install -e ".[dev]"
```

## Files Used

```text
lllm/
  proxy.py
tests/
  test_proxy.py
```

## Wrap

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

## Hooks

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

When the wrapped tactic supports streaming, the proxy mirrors that capability;
captured outputs are recorded as the consumed stream chunks.

## Verify

```bash
python -m pytest tests/test_proxy.py -q
```

Expected output:

```text
... passed
```

Next, add proxy metadata to a package card when a service should advertise the
observability or transformation boundary.
