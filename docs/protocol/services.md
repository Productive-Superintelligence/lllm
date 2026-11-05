# Services

Any tactic can be served over HTTP with the contract it has in Python. A request
carries the input and a `CallContext`, a response carries the validated output,
and `/info` publishes the schemas. `RemoteTactic` turns a service back into a
tactic, so calling code does not change when a tactic moves to another process
or machine.

```bash
pip install "lllm-core[server]"   # FastAPI and Uvicorn, for serving
pip install "lllm-core[client]"   # httpx, for RemoteTactic
```

## Serve a tactic

Put a tactic in a module, `tactics.py`:

```python
from pydantic import BaseModel
from lllm import Tactic, TacticEvent


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    text: str
    trace_id: str | None = None


class Shout(Tactic):
    """Shout the question back."""

    name = "shout"
    input_type = Question
    output_type = Answer

    def call(self, task: Question, *, context=None) -> Answer:
        trace_id = context.trace_id if context else None
        return Answer(text=task.text.upper(), trace_id=trace_id)

    def stream(self, task: Question, *, context=None):
        for word in task.text.split():
            yield word.upper()
        yield TacticEvent.result(self.run(task, context=context))
```

Serve it from the same folder:

```bash
lllm serve tactics:Shout --port 8000
```

```text
INFO:     Started server process [3940928]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Call it:

```bash
curl -s -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input": {"text": "hello"}, "context": {"trace_id": "trace-1"}}'
```

```json
{"output":{"text":"HELLO","trace_id":"trace-1"},"request_id":"fba55530-a1a1-4f28-abfa-535817256e08","tactic":"shout"}
```

The `module:attribute` can name a tactic class or instance, a factory that
returns one, an object with a `run` method, or a handler function. A factory is
called once at load and a handler once per input; the rule that tells them
apart is in [Loading entrypoints](runtimes.md#loading-entrypoints). `--host`
defaults to `127.0.0.1`, `--port` to 8000, and `--log-level` to `info`.

To build the app in code, use `create_tactic_app` and any ASGI server:

```python
# app.py
from lllm import create_tactic_app
from tactics import Shout

app = create_tactic_app(Shout())
```

```bash
uvicorn app:app --port 8000
```

`create_tactic_app(tactic, *, title=None, description=None)` returns a FastAPI
app titled with the tactic's name and description unless you pass your own.

## Routes

| Route | Purpose |
| --- | --- |
| `GET /health` | `{"ok": true, "tactics": [...]}` |
| `GET /info` | The tactic's `TacticInfo`. |
| `POST /run` | Run once and return the output. |
| `POST /stream` | Stream events as server-sent events. |
| `GET /tactics` | `TacticInfo` for every tactic in the app. |
| `GET /tactics/{name}/info` | One tactic's `TacticInfo`. |
| `POST /tactics/{name}/run` | Run one tactic. |
| `POST /tactics/{name}/stream` | Stream one tactic. |

`create_tactic_app` mounts all of them. FastAPI's own `/docs` and
`/openapi.json` are there as usual. `/info` publishes the tactic's input and
output schemas, and `/openapi.json` shows the same schemas as the `input` of
the `/run` and `/stream` request envelopes and the `output` of the `/run`
response. The `/tactics/{name}/...` routes are listed without them.

### Several tactics in one app

`create_service_app` serves a list of tactics (routed by name) or a mapping of
route names to tactics. It mounts `/health` and the `/tactics/...` routes:

```python
from fastapi.testclient import TestClient
from lllm import as_tactic, create_service_app
from tactics import Shout


def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())


app = create_service_app([Shout(), as_tactic(word_count)], title="Text tools")
client = TestClient(app)

print(client.get("/health").json())
print(client.post("/tactics/word_count/run", json={"input": "one two three"}).json())
```

```text
{'ok': True, 'tactics': ['shout', 'word_count']}
{'output': 3, 'request_id': 'e518af99-f181-4587-9fcd-a1fef116b8fa', 'tactic': 'word_count'}
```

Route names must be unique, plain path segments. Pass
`expose_single_tactic_routes=True` to also mount `/info`, `/run`, and `/stream`
when the app holds exactly one tactic.

## Requests and responses

`POST /run` takes an envelope with the input and an optional context:

```json
{
  "input": {"text": "hello"},
  "context": {
    "request_id": "req-1",
    "trace_id": "trace-1",
    "caller": "cli",
    "tags": {"env": "dev"},
    "metadata": {"user": "ada"}
  }
}
```

The `context` object has the shape of a [`CallContext`](tactic.md#callcontext):
`request_id`, `trace_id`, `span_id`, `caller`, `tags`, `metadata`,
`package_ref`, `service_ref`, and `tactic_ref`. The service generates a
`request_id` when there is none, sets `endpoint` to the route (`"run"`,
`"stream"`, or a custom endpoint's name), defaults `tactic_ref` to the tactic's
`package_ref`, and ignores keys it does not know. This is the context the
tactic receives:

```python
from fastapi.testclient import TestClient
from lllm import Tactic, create_tactic_app


class ShowContext(Tactic):
    name = "show-context"
    output_type = dict

    def call(self, task, *, context=None):
        return context.model_dump(exclude_none=True)


app = create_tactic_app(ShowContext(package_ref="lllm://demo/debug/tactics/show-context"))
response = TestClient(app).post("/run", json={
    "input": "hi",
    "context": {"request_id": "req-1", "trace_id": "trace-1", "caller": "cli",
                "tags": {"env": "dev"}, "metadata": {"user": "ada"}, "region": "eu"},
})
print(response.json())
```

```text
{'output': {'request_id': 'req-1', 'caller': 'cli', 'trace_id': 'trace-1', 'tactic_ref': 'lllm://demo/debug/tactics/show-context', 'endpoint': 'run', 'tags': {'env': 'dev'}, 'metadata': {'user': 'ada'}}, 'request_id': 'req-1', 'tactic': 'show-context'}
```

A successful `/run` returns `{"output": ..., "request_id": ..., "tactic": ...}`.

Two looser forms are accepted for clients you cannot change: `task` works in
place of `input`, and a body that is not an envelope is taken as the input
itself, so `{"text": "hello"}` works too. (An envelope is a JSON object with
`input` or `task` and no keys besides `input`, `task`, and `context`.) New
clients should send the envelope; it is the only form that carries a context.

## Streaming

`POST /stream` responds with `text/event-stream`. Each event is one `data:` line
holding a [`TacticEvent`](tactic.md#streaming-and-events) as JSON:

```bash
curl -s -N -X POST http://127.0.0.1:8000/stream \
  -H 'content-type: application/json' \
  -d '{"input": {"text": "hello world"}}'
```

```text
data: {"id": "52f5acbc-174f-4d6f-bdb9-f52ddcefaecb", "timestamp": 1793845808.4317925, "kind": "message", "data": "HELLO", "metadata": {}}

data: {"id": "34f98b46-3109-4105-bcdb-5bfe20faf471", "timestamp": 1793845808.4319072, "kind": "message", "data": "WORLD", "metadata": {}}

data: {"id": "8e506315-e940-4dd2-bb34-5a5c063a77b5", "timestamp": 1793845808.432171, "kind": "result", "data": {"text": "HELLO WORLD", "trace_id": null}, "metadata": {}}
```

The route sends what `tactic.aevents()` yields. For a tactic that implements
`stream()`, that path validates the input before `stream()` runs
([Streaming and events](tactic.md#streaming-and-events)). Once the response
has started, a failure (including invalid input) arrives as a final `error`
event while the HTTP status stays 200:

```text
data: {"id": "458cf822-a0f0-4a81-b345-a02520feec00", "timestamp": 1793845812.2663333, "kind": "error", "data": {"message": "Value does not match schema Question: 1 validation error for Question\ntext\n  Field required [type=missing, input_value={'txt': 'hello'}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.14/v/missing"}, "metadata": {"error_type": "SchemaError", "tactic": "shout", "endpoint": "stream", "request_id": "req-7"}}
```

A tactic that supports neither `stream` nor `events` gets a 400
`TacticUnsupportedError` before any event is sent. Event metadata passes through
the same secret filter as `/info`.

## Errors

Failures return a stable JSON body under FastAPI's `detail` key:

```bash
curl -s -X POST http://127.0.0.1:8000/run \
  -H 'content-type: application/json' \
  -d '{"input": {"text": 5}, "context": {"request_id": "req-42"}}'
```

```json
{
  "detail": {
    "error": {
      "type": "SchemaError",
      "message": "Value does not match schema Question: 1 validation error for Question\ntext\n  Input should be a valid string [type=string_type, input_value=5, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.14/v/string_type",
      "tactic": "shout",
      "endpoint": "run",
      "request_id": "req-42",
      "metadata": {}
    }
  }
}
```

| Status | `error.type` | When |
| --- | --- | --- |
| 400 | `SchemaError` | Invalid input or output, or a malformed envelope or context. |
| 400 | `TacticUnsupportedError` | `/stream` on a tactic that does not stream. |
| 404 | `TacticNotFound` | `/tactics/{name}/...` with an unknown name. |
| 500 | the exception's class name | Anything else the tactic raises; `message` is `str(exc)`. |

`RemoteTactic` turns this body back into a `RemoteTacticError`
([below](#calling-a-service-from-python)). See also the
[Service API reference](../reference/service-api.md).

## Custom endpoints

A tactic can mount its own routes next to the standard ones. Decorate a method
with `endpoint.post`, `endpoint.get`, `endpoint.put`, `endpoint.patch`, or
`endpoint.delete`:

```python
from fastapi.testclient import TestClient
from pydantic import BaseModel
from lllm import Tactic, TacticEvent, create_tactic_app, endpoint


class Ticket(BaseModel):
    text: str


class Route(BaseModel):
    queue: str
    urgent: bool


class Triage(Tactic):
    """Route a support ticket to a queue."""

    name = "triage"
    input_type = Ticket
    output_type = Route

    def call(self, task: Ticket) -> Route:
        urgent = "outage" in task.text.lower()
        return Route(queue="oncall" if urgent else "support", urgent=urgent)

    @endpoint.post("/triage/explain", description="Route a ticket and say why.")
    def explain(self, ticket: Ticket, *, context=None) -> dict:
        route = self.run(ticket, context=context)
        reason = "mentions an outage" if route.urgent else "routine request"
        return {"queue": route.queue, "reason": reason, "endpoint": context.endpoint}

    @endpoint.post("/triage/steps", mode="stream")
    def steps(self, ticket: Ticket, *, context=None):
        yield {"step": "read", "chars": len(ticket.text)}
        yield TacticEvent.result(self.run(ticket, context=context))


client = TestClient(create_tactic_app(Triage()))
print(client.post("/triage/explain", json={"input": {"text": "Total outage in eu-west"}}).json())
print(client.post("/triage/steps", json={"input": {"text": "Printer jam"}}).text)
```

```text
{'queue': 'oncall', 'reason': 'mentions an outage', 'endpoint': 'explain'}
data: {"step": "read", "chars": 11}

data: {"id": "2011ecab-0a65-4527-9946-c563d23cdef4", "timestamp": 1793845911.8305802, "kind": "result", "data": {"queue": "support", "urgent": false}, "metadata": {}}
```

How custom endpoints behave:

- The signature is `endpoint.post(path, *, name=None, mode="run", description="", tags=())`.
  `name` defaults to the method name and becomes `context.endpoint`; the
  description defaults to the docstring.
- The method receives the request's input, read from the same envelope as `/run`
  and validated against the tactic's `input_type`, plus `context=` when it
  accepts one. Sync and async methods both work.
- `endpoint.get` and `endpoint.delete` routes may be called without a body.
  The method then receives the query parameters as a dict of strings (a
  repeated key gives a list), or `None` when there are none, without
  validation. A request that sends a body is read and validated as above.
- With `mode="run"` the return value is the JSON response, with no envelope
  around it. With `mode="stream"` or `mode="events"`, each yielded item becomes
  one `data:` line: events as `TacticEvent` JSON, anything else as plain JSON.
- Routes may not shadow the standard routes (`/run`, `/stream`, `/info`,
  `/tactics/...`), and each method and path pair must be unique across the
  app. Both are checked when the app is built.
- `lllm inspect module:attr --json` lists the endpoints with their method,
  path, mode, description, and tags.

## Calling a service from Python

`RemoteTactic` is a tactic whose implementation is an HTTP service. With the
service from [Serve a tactic](#serve-a-tactic) running:

```python
from lllm import CallContext, RemoteTactic
from tactics import Answer, Question

remote = RemoteTactic(
    "http://127.0.0.1:8000",
    name="shout",
    input_type=Question,
    output_type=Answer,
)

print(repr(remote.run({"text": "hello"}, context=CallContext(trace_id="trace-1"))))
print(list(remote.stream({"text": "hello world"})))
print(remote.fetch_info().capabilities)
```

```text
Answer(text='HELLO', trace_id='trace-1')
['HELLO', 'WORLD', {'text': 'HELLO WORLD', 'trace_id': None}]
('acall', 'arun', 'call', 'run', 'stream')
```

| Method | HTTP call |
| --- | --- |
| `run()`, `await arun()` | `POST {url}/run` with the envelope; returns `output`. |
| `stream()`, `astream()` | `POST {url}/stream`; yields each event's `data`. |
| `aevents()` | `POST {url}/stream`; yields `TacticEvent`s. |
| `fetch_info()`, `await afetch_info()` | `GET {url}/info`; returns the service's `TacticInfo`. |

Things to know:

- The URL is the service base (`http://host:8000`), or a tactic inside a
  multi-tactic app (`http://host:8000/tactics/shout`). A URL that already ends
  in `/run`, `/stream`, or `/info` is normalized.
- Without `name=`, the name comes from the URL: its last path segment
  (ignoring a trailing `/run`, `/stream`, or `/info`), such as `shout` in
  `.../tactics/shout`, or else the host and port, so `http://127.0.0.1:8000`
  is named `127.0.0.1-8000`.
- `run()` and `arun()` validate the input against `input_type` before anything
  is sent, and validate the response into `output_type`; without `output_type`
  you get the JSON as-is. `stream()`, `astream()`, and `aevents()` send the
  input as given and do not validate what comes back. Bad input fails on the
  service and arrives as its `error` event ([Streaming](#streaming)):
  `stream()` and `astream()` yield that event's data like any other item, and
  `aevents()` yields the event with `kind="error"`.
- The context travels in the envelope, so ids, tags, and metadata reach the
  remote tactic.
- `info()` is local (`runtime_kind` is `"http"`); `fetch_info()` asks the
  service.
- `timeout` defaults to 30 seconds. `transport=` and `async_transport=` take
  httpx transports, which is how tests call an app in-process; see
  [Refs and Composition](refs.md#test-against-an-in-process-service).

An error response (status 400 or above) raises `RemoteTacticError`, with the
structured fields as attributes:

```python
from lllm import RemoteTacticError

untyped = RemoteTactic("http://127.0.0.1:8000", name="shout")
try:
    untyped.run({"text": 5})
except RemoteTacticError as exc:
    print(exc.status_code, exc.error_type, exc.tactic, exc.endpoint)
```

```text
400 SchemaError shout run
```

`RemoteTacticError` also carries `message`, `request_id`, and the raw `detail`.
It subclasses `TacticServiceError`.

## The CLI

### lllm serve

```bash
lllm serve module:attribute [--host 127.0.0.1] [--port 8000] [--log-level info]
lllm serve PATH [--resource SECTION.NAME] [--port N] [--skip-key-check]
```

Given a package folder (or its `lllm.toml`), `lllm serve` launches the
package's primary resource (`[package] primary`), or the one named with
`--resource`, such as `tactics.shout` or `services.api`. The port comes from
`--port`, else the resource's `metadata.port` (for a tactic without one, the
port of a service that declares it as its `tactic`), else 8000. Here is a package
folder holding the `tactics.py` above and this manifest:

```toml
[package]
org = "demo"
name = "shout"
version = "0.1.0"
primary = "tactics.shout"

[requirements.api_keys]
OPENAI_API_KEY = "Model access for the agents."

[tactics.shout]
entry = "tactics:Shout"
description = "Shout the question back."

[tactics.shout.metadata]
port = 8100
```

Before starting, `lllm serve` checks the API keys the package declares, in
`[requirements.api_keys]` or a resource's `required_api_keys` metadata. It looks
in the environment, the `--env-file` (default `.env.local`), and the OS keyring
(skip it with `--no-keyring`). In a terminal it prompts for missing keys;
otherwise it stops:

```text
usage: lllm [-h] [--version] COMMAND ...
lllm: error: missing required API key(s):
- OPENAI_API_KEY - Model access for the agents.

Run `lllm setup .` or set the missing names in your environment.
```

`lllm setup .` stores keys in the OS keyring or `.env.local`. It also works for
a package with nothing to launch, such as a library, from its
`[requirements.api_keys]`. `--skip-key-check` starts without checking. The
manifest format is covered in [The lllm.toml manifest](../packages/manifest.md).

A tactic entry loads by the same rule as a `module:attribute` target. A service
entry has no handler form: it names a FastAPI or Starlette app, or something
that can be called without arguments, which is called once to build an app, a
tactic, or an object with a `run` method. If that call returns plain data
instead (a mapping, list, tuple, string, int, float, or bool), a service that
declares `tactic` serves that tactic, and a service without one serves the data
from a fixed JSON app: `GET /health`, plus `POST /run`, which returns the data
as `output`. Any other entry stops `lllm serve` with an error that names the
cause, instead of serving the service's tactic without its auth or middleware.
That includes an entry that fails to import, raises while building, names a
tactic instance, or returns an app wrapped in raw ASGI middleware, `None`, or a
coroutine:

```text
usage: lllm [-h] [--version] COMMAND ...
lllm: error: service api entry service:create_app failed: NameError: name 'add_auth' is not defined
```

Only `services`, `tactics`, `channels`, and `runs` resources can be launched.
Naming another kind, such as `config.default`, as the primary or with
`--resource` is an error that lists what can be launched:

```text
usage: lllm [-h] [--version] COMMAND ...
lllm: error: resource config.default cannot be launched; launchable resources: tactics.shout
```

A served package keeps its modules loaded and its folder on `sys.path` for the
life of the process, so its code can import the package's own modules when it
is called. Packages served in one process, such as the packages of an
[AAAX](../aaax/index.md) strategy, share `sys.modules`, so give their top-level
modules distinct names.

### lllm inspect

`lllm inspect` shows what a target is without starting anything. For an
entrypoint it prints the tactic's contract:

```bash
lllm inspect tactics:Shout
```

```text
shout (native)
Shout the question back.
capabilities: acall, arun, call, run, stream
```

`--json` prints the full contract: `name`, `description`, `runtime`,
`capabilities`, `endpoints`, `input_schema`, `output_schema`, `package_ref`,
`service_ref`, `examples`, and `metadata`, with secret-shaped metadata keys
removed. For a package it shows what `lllm serve` would launch, the state of
each declared key, and the routes:

```bash
lllm inspect .
```

```text
demo/shout:tactics.shout (tactic)
port: 8100
api keys:
  OPENAI_API_KEY: missing - Model access for the agents.
GET /health
GET /info
GET /tactics
GET /tactics/{name}/info
GET,HEAD /docs
GET,HEAD /docs/oauth2-redirect
GET,HEAD /openapi.json
GET,HEAD /redoc
POST /run
POST /stream
POST /tactics/{name}/run
POST /tactics/{name}/stream
```

All commands and flags are in the [CLI reference](../reference/cli.md).

## Next

- [Refs and Composition](refs.md): bind `lllm://` refs to services in
  `.lllm/config.toml` and compose local and remote tactics.
- [Tutorial 9: Serving](../tutorials/09-serving.md): a guided walkthrough.
