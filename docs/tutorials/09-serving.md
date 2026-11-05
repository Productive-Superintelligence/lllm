# 9. Serving and Composition

**Goal:** put tactics behind HTTP, call them from anywhere as if they were
local, and wire a system together by refs instead of hard-coded objects.

Everything here runs offline; examples use FastAPI's test client so you can
run them without starting a server.

## Step 1: a tactic worth serving

Any tactic can be served, whether its program is a native `call()` or the
protocol hook `_run`:

```python
from pydantic import BaseModel
from lllm import Tactic, endpoint


class Question(BaseModel):
    text: str


class Shout(Tactic):
    name = "shout"
    input_model = Question

    def call(self, task):
        return task.text.upper()

    def stream(self, task, **kwargs):
        for word in task.text.split():
            yield word.upper()

    @endpoint.post("/whisper", description="Lowercase instead.")
    async def whisper(self, input_value, *, context=None):
        return input_value.text.lower()
```

`stream` adds a streaming surface; `@endpoint` adds extra routes. Both receive
the validated `Question`, exactly like `call`.

## Step 2: serve it

From the command line:

```bash
lllm serve tactics:Shout --port 8000
```

Or build the app yourself:

```python
from fastapi.testclient import TestClient
from lllm.services import create_tactic_app

client = TestClient(create_tactic_app(Shout()))

client.post("/run", json={"input": {"text": "hello there"}}).json()
# {'output': 'HELLO THERE', 'request_id': '...', 'tactic': 'shout'}

client.post("/whisper", json={"input": {"text": "HELLO"}}).json()
# 'hello'

client.get("/info").json()["capabilities"]
# ['acall', 'arun', 'call', 'run', 'stream']
```

| Route | Does |
| --- | --- |
| `POST /run` | One call: `{"input": ..., "context": {...}}` → `{"output": ...}` |
| `POST /stream` | Server-sent events, one per streamed item |
| `GET /info` | Name, description, JSON schemas, capabilities |
| `GET /health` | Liveness |
| your `@endpoint` routes | Custom surfaces on the same tactic |

The request `context` carries `trace_id`, `request_id`, `tags`, and
`metadata`; a `call(self, task, *, context=None)` signature receives it as a
`CallContext`. Errors come back in one envelope with the error type, message,
tactic, and request id. See [Services](../protocol/services.md).

## Step 3: several tactics, one service

```python
from lllm.services import create_service_app


class Count(Tactic):
    name = "count"

    def call(self, task: str) -> int:
        return len(task.split())


service = TestClient(create_service_app([Shout(), Count()]))
service.post("/tactics/count/run", json={"input": "a b c"}).json()["output"]
# 3
[info["name"] for info in service.get("/tactics").json()]
# ['shout', 'count']
```

Each tactic gets `/tactics/<name>/run`, `/stream`, and `/info`.

## Step 4: call a service as a local tactic

`RemoteTactic` speaks the same HTTP API and is itself a tactic, so callers do
not change when an implementation moves behind a URL:

```python
from lllm import RemoteTactic

shout = RemoteTactic("http://127.0.0.1:8000")
shout.run({"text": "remote"})          # 'REMOTE'
shout.fetch_info().capabilities        # what the service advertises
```

`arun`, `astream`, and `aevents` work the same way.

## Step 5: name things by ref

A ref names a resource independently of where it runs:
`lllm://org/package/tactics/name`. Code asks for the ref; a resolver decides
whether it is a local object or a service.

Bind refs to services in `.lllm/config.toml`:

```toml
[refs."lllm://demo/briefs/tactics/shout"]
url = "http://127.0.0.1:8000"

[refs."lllm://demo/briefs/channels/events"]
store = ".lllm/network"
```

and resolve them with one `lllm.Resolver`:

```python
from lllm import Resolver

resolver = Resolver.from_config(".")
resolver.tactic("lllm://demo/briefs/tactics/shout")   # a RemoteTactic
resolver.store("lllm://demo/briefs/channels/events")  # a LocalStore

resolver.register("lllm://demo/briefs/tactics/count", Count())
resolver.run("lllm://demo/briefs/tactics/count", "x y")  # 2
```

Tests register local objects; deployments bind URLs; the code in between
never changes. Channel refs resolve to [SSSN](../sssn/index.md) stores.

## Step 6: serve a whole package

A package folder serves its primary resource, after checking the API keys it
declares:

```bash
lllm setup ./briefs
lllm serve ./briefs --port 8000
lllm inspect ./briefs        # what would be served, and which keys are missing
```

To serve everything a package declares (tactics and channels) as one API, mount
it in an [AAAX strategy](../aaax/index.md): `lllm shell serve ./briefs`.

## Summary

| You want | Use |
| --- | --- |
| HTTP for one tactic | `lllm serve module:attr` or `create_tactic_app(tactic)` |
| Many tactics, one app | `create_service_app([...])` |
| Extra routes | `@endpoint.post("/path")` on the tactic |
| A remote tactic | `RemoteTactic(url)` |
| Wiring by name | `lllm://` refs, `.lllm/config.toml`, `lllm.Resolver` |
| A package | `lllm serve ./package`, `lllm shell serve ./package` |

**Next:** [10. Pydantic AI Agents](10-pydantic-ai.md)
