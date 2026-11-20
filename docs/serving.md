# Serving

Every tactic is a typed web service. Its input and output models become the
request and response schemas, so the HTTP contract is the Python contract
and there is nothing to keep in sync. `RemoteTactic` turns a service back
into a tactic, so calling code does not change when a tactic moves to
another machine.

```python title="triage.py"
from fastapi.testclient import TestClient
from pydantic import BaseModel

from lllm import Prompt, Tactic, create_tactic_app


class Ticket(BaseModel):
    text: str


class Route(BaseModel):
    queue: str
    reason: str


ROUTE = Prompt(
    path="support/route",
    prompt="Route this ticket to billing, bugs, or general:\n{text}",
    format=Route,
)


class Triage(Tactic):
    """Route a support ticket to a queue."""

    name = "triage"
    agent_group = ["router"]
    input_model = Ticket  # (1)!
    output_model = Route

    def call(self, ticket: Ticket) -> Route:
        router = self.agents["router"]
        router.open("route")
        router.receive_prompt(ROUTE, {"text": ticket.text})
        return router.respond().parsed


triage = Triage(
    {
        "global": {"model_name": "gpt-4o-mini"},
        "agent_configs": [
            {"name": "router", "system_prompt": "You triage tickets."}
        ],
    }
)

client = TestClient(create_tactic_app(triage))  # (2)!
response = client.post(
    "/run",
    json={
        "input": {"text": "I was charged twice this month."},  # (3)!
        "context": {"request_id": "req-1"},
    },
)
print(response.json())
"""
{
    'output': {'queue': 'billing', 'reason': 'Duplicate charge.'},
    'request_id': 'req-1',
    'tactic': 'triage',
}
"""
```

1. The models are the contract. A request is validated against `Ticket`
   before `call()` runs, the result is validated against `Route`, and both
   schemas are published on `/info` and in the OpenAPI docs.
2. `create_tactic_app` builds a FastAPI app. `TestClient` calls it in the
   same process, which is also how you test a service.
3. The request envelope: `input` is the tactic's input, and the optional
   `context` carries request metadata such as ids for tracing.

`lllm serve` builds the same app from the command line and runs it with
Uvicorn. Serving needs the `server` extra (`pip install
"lllm-core[server]"`), and `RemoteTactic` needs `client` (httpx).

```bash
lllm serve triage:triage --port 8000
curl -s localhost:8000/run -H 'content-type: application/json' \
  -d '{"input": {"text": "I was charged twice this month."}}'
```

## Serve from the command line

`lllm serve module:attribute` imports the module from the current directory
and serves the attribute. Name the module, as in `triage:triage`; a file
path such as `triage.py:triage` does not import.

| Attribute | `lllm serve` |
| --- | --- |
| A tactic, or a `Tactic` subclass | Serves it; a class is built with no arguments |
| A FastAPI or Starlette app | Serves the app as it is |
| A function with no parameters | Calls it once at startup and serves the tactic or app it returns |
| Any other function, or an object with `run` | Calls it once per request; its annotations become the schemas |

A factory is how you serve a tactic that needs setup, such as loading a
package and building the tactic from its config, which is what
`create_tactic()` does in projects made by
[`lllm create`](packages.md#start-a-package). `--host` defaults to
`127.0.0.1`, `--port` to 8000, and `--log-level` to `info`.
`lllm inspect triage:triage` prints the contract without starting
anything; `--json` adds the schemas and custom endpoints.

An app target is how you serve your own app, such as
`app = create_tactic_app(triage)` with middleware or routes you added:
`lllm serve app:app` runs it unchanged, and `lllm inspect app:app` lists
its routes. To pick the server and its settings yourself, run the app
with any ASGI server, such as `uvicorn app:app`. The app's title and
description come from the tactic unless you pass `title=` and
`description=`.

## Endpoints and envelopes

| Route | Does |
| --- | --- |
| `POST /run` | Runs the tactic once and returns the output |
| `POST /stream` | Runs it and sends events as they happen ([Streaming](#streaming)) |
| `GET /info` | Name, description, input and output schemas, capabilities |
| `GET /health` | `{"ok": true, "tactics": [...]}` |
| `GET /docs`, `/openapi.json` | FastAPI's interactive docs and OpenAPI document |
| `/tactics/{name}/run`, `/stream`, `/info` | The same, by tactic name ([Several tactics](#several-tactics-in-one-app)) |

`/info` is how clients learn the contract without running anything
(`RemoteTactic.fetch_info()` reads it), and `/openapi.json` carries the
same schemas, so clients generated from it are typed.

`POST /run` takes an envelope and answers with `output`, `request_id`, and
`tactic`. The input stays exactly the tactic's input; the metadata travels
next to it:

```json
{
  "input": {"text": "Total outage in eu-west"},
  "context": {
    "request_id": "req-7",
    "trace_id": "trace-42",
    "caller": "web",
    "tags": {"env": "prod"},
    "metadata": {"user": "ada"}
  }
}
```

The `context` becomes the `CallContext` that a `call()` declaring `context`
receives. The service generates a `request_id` when there is none, sets
`endpoint` to the route (`run`, `stream`, or a custom endpoint's name), and
ignores keys it does not know. Failures share one body, under FastAPI's
`detail` key:

```python title="requests.py"
from fastapi.testclient import TestClient

from lllm import Tactic, create_tactic_app


class Lookup(Tactic):
    name = "lookup"
    input_type = str

    def call(self, order_id: str, *, context=None) -> dict:
        return {"order": order_id, "trace": context.trace_id}


client = TestClient(create_tactic_app(Lookup()))
good = {"input": "A-1001", "context": {"trace_id": "trace-42"}}
print(client.post("/run", json=good).json()["output"])
#> {'order': 'A-1001', 'trace': 'trace-42'}

bad = {"input": 1001, "context": {"request_id": "req-8"}}
response = client.post("/run", json=bad)
error = response.json()["detail"]["error"]
print(response.status_code, error["type"], error["request_id"])
#> 400 SchemaError req-8
print(sorted(error))
#> ['endpoint', 'message', 'metadata', 'request_id', 'tactic', 'type']
```

| Status | `error.type` | When |
| --- | --- | --- |
| 400 | `SchemaError` | The input, the envelope, or the context does not validate |
| 400 | `TacticUnsupportedError` | `/stream` on a tactic that does not stream |
| 404 | `TacticNotFound` | An unknown name in `/tactics/{name}/...` |
| 500 | `OutputSchemaError` | The tactic's result does not match its output schema, a bug in the tactic rather than the request |
| 500 | The exception's class name | Anything else the tactic raises; `message` is `str(exc)` |

For clients you cannot change, a body that is not an envelope is taken as
the input itself, and `task` works in place of `input`; only the envelope
carries a context. The [Service API](api/service-api.md) reference has the
exact shapes.

## Streaming

`POST /stream` answers with server-sent events. Each item the tactic's
`stream()` yields becomes one `data:` line holding a `TacticEvent` as JSON
(`id`, `timestamp`, `kind`, `data`, `metadata`):

```python title="stream.py"
import json

from fastapi.testclient import TestClient

from lllm import Tactic, TacticEvent, create_tactic_app


class Triage(Tactic):
    name = "triage"

    def call(self, text: str) -> str:
        return "oncall" if "outage" in text.lower() else "support"

    def stream(self, text: str, **kwargs):  # (1)!
        yield "reading the ticket"
        yield TacticEvent.result(self.call(text))


client = TestClient(create_tactic_app(Triage()))
body = {"input": "Total outage"}
with client.stream("POST", "/stream", json=body) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line.removeprefix("data: "))
            print(event["kind"], event["data"])
            #> message reading the ticket
            #> result oncall
```

1. `stream()` receives the validated input, like `call()`. Plain values
   become `message` events; `TacticEvent.result()` marks the result.

Once the response has started, a failure (invalid input included) arrives
as a final `error` event, and the HTTP status stays 200. A tactic that
implements neither `stream()` nor `events()` answers `/stream` with a 400
`TacticUnsupportedError`. [Streaming](streaming.md) covers writing streams.

## Custom endpoints

A tactic can add its own routes next to the standard ones. Decorate a
method with `endpoint.post`, `get`, `put`, `patch`, or `delete`:

```python title="endpoints.py"
from fastapi.testclient import TestClient

from lllm import Tactic, create_tactic_app, endpoint


class Triage(Tactic):
    name = "triage"

    def call(self, text: str) -> str:
        return "oncall" if "outage" in text.lower() else "support"

    @endpoint.post("/triage/explain")
    def explain(self, text: str) -> dict:  # (1)!
        """Route a ticket and say why."""
        queue = self.run(text)
        why = "mentions an outage" if queue == "oncall" else "routine"
        return {"queue": queue, "why": why}


client = TestClient(create_tactic_app(Triage()))
body = {"input": "Outage in eu-west"}
print(client.post("/triage/explain", json=body).json())
#> {'queue': 'oncall', 'why': 'mentions an outage'}
```

1. The method gets the input from the same envelope as `/run`, validated
   like the tactic's input, plus `context=` if it accepts one. Its return
   value is the response body, without an envelope.

The decorators take `(path, *, name=None, mode="run", description="",
tags=())`. The name defaults to the method's name and becomes
`context.endpoint`; the description defaults to the docstring. With
`mode="stream"`, each item the method yields is sent as one `data:` line:
a `TacticEvent` as event JSON, anything else as plain JSON. A `GET` or
`DELETE` called without a body passes the query parameters as a dict (or
`None`), without validation. Sync and async methods both work. A route may
not shadow a standard route or repeat a method and path; both are checked
when the app is built.

## Authentication

`create_tactic_app` returns a plain FastAPI app, so you secure it the
FastAPI way: middleware, dependencies, or a gateway in front of it. This
middleware requires a bearer token on every route except `/health`, and a
`RemoteTactic` caller sends one:

```python title="auth.py"
import asyncio
import secrets

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

from lllm import RemoteTactic, RemoteTacticError, Tactic, create_tactic_app


class Triage(Tactic):
    name = "triage"

    def call(self, text: str) -> str:
        return "oncall" if "outage" in text.lower() else "support"


TOKEN = "change-me"  # read it from the environment in production
app = create_tactic_app(Triage())


@app.middleware("http")
async def require_token(request: Request, call_next):
    sent = request.headers.get("authorization", "").encode()
    expected = f"Bearer {TOKEN}".encode()
    if request.url.path != "/health" and not secrets.compare_digest(
        sent, expected
    ):
        return JSONResponse({"detail": "Invalid token."}, status_code=401)
    return await call_next(request)


async def main():
    url = "http://triage.internal:8000"
    transport = httpx.ASGITransport(app=app)  # (1)!
    try:
        await RemoteTactic(url, async_transport=transport).arun("Outage")
    except RemoteTacticError as error:
        print(error.status_code, error.message)  # (2)!
        #> 401 Invalid token.
    triage = RemoteTactic(
        url,
        headers={"authorization": f"Bearer {TOKEN}"},  # (3)!
        async_transport=transport,
    )
    print(await triage.arun("Outage"))
    #> oncall


asyncio.run(main())
```

1. Only for this example: the transport sends requests to the app in this
   process, as in
   [Calling a served tactic from Python](#calling-a-served-tactic-from-python).
2. A body that is not the service's error envelope, such as FastAPI's
   `{"detail": "Invalid token."}`, gives its `detail` text as the message.
3. `headers=` go with every request the tactic sends: `run()`, `arun()`,
   the stream methods, and `fetch_info()`. `auth=` takes any httpx
   authentication the same way, such as `("user", "password")` for basic
   auth.

Keyword arguments to one call still go to httpx, so
`triage.run(text, headers={"x-request-id": "r-1"})` adds a header to that
request only. `lllm serve auth:app` serves the secured app as it is.

In a package, build the secured app in a service entry and make the
service the package's `primary` (or pass `--resource services.api`), so
`lllm serve` launches the app rather than the bare tactic. If the entry
fails to import or raises while building, `lllm serve` stops with the
error instead of falling back to the unsecured tactic:

```toml
[package]
name = "triage"
primary = "services.api"

[services.api]
entry = "service:create_app"   # returns the app with its middleware
tactic = "triage"
```

## Several tactics in one app

`create_service_app([triage, summarize], title="Support tools")` serves
several tactics from one app, each under its name:
`POST /tactics/triage/run`, `POST /tactics/summarize/stream`,
`GET /tactics/summarize/info`, plus `GET /tactics` (every tactic's info)
and `/health`. It takes a list, routed by each tactic's name, or a dict
from route names to tactics; an unknown name answers 404 `TacticNotFound`.
`create_tactic_app` is the one-tactic case, with `/run`, `/stream`, and
`/info` added.

## Calling a served tactic from Python

`RemoteTactic` is a tactic whose implementation is an HTTP service. It
sends the envelope, raises `RemoteTacticError` for an error response, and
otherwise behaves like any tactic:

```python title="remote.py"
import asyncio

import httpx

from lllm import RemoteTactic, RemoteTacticError, Tactic, create_tactic_app


class Triage(Tactic):
    name = "triage"
    input_type = str

    def call(self, text: str) -> str:
        return "oncall" if "outage" in text.lower() else "support"


app = create_tactic_app(Triage())  # running on another machine


async def main():
    triage = RemoteTactic(
        "http://triage.internal:8000",
        async_transport=httpx.ASGITransport(app=app),  # (1)!
    )
    print(triage.tactic_name)
    #> triage.internal-8000
    print(await triage.arun("Total outage in eu-west"))
    #> oncall
    try:
        await triage.arun(42)
    except RemoteTacticError as error:
        print(error.status_code, error.error_type, error.endpoint)
        #> 400 SchemaError run


asyncio.run(main())
```

1. Only for this example: the transport sends requests to the app in this
   process. Against a running service, pass the URL alone, and
   `triage.run(...)` works the same way, synchronously.

`run()` and `arun()` post to `{url}/run`; `stream()` and `astream()` yield
the data of each event from `{url}/stream`, and `aevents()` the events
themselves; `fetch_info()` and `afetch_info()` read `{url}/info`. The URL is
the service base or one tactic of a multi-tactic app
(`http://host:8000/tactics/triage`). Without `name=`, the tactic is named
after the URL's last path segment, or else its host and port. Pass
`input_type` and `output_type` to validate before sending and to get typed
results back; without them, JSON is returned as is. The `timeout` is 30
seconds by default, and `headers=` and `auth=` go with every request
([Authentication](#authentication)). `RemoteTacticError` carries
`status_code`, `error_type`, `message`, `tactic`, `endpoint`, `request_id`,
and `detail`.

A `RemoteTactic` built from a URL alone knows nothing else about the
service: its `info()` has no description or schemas, so a
[tool](tools.md#tactics-as-tools) made from it takes one string `task` and
is described only as `Run <name>.`. `RemoteTactic.from_service()` reads
`/info` once and adopts the service's own contract:

```python title="contract.py"
import asyncio

import httpx
from pydantic import BaseModel

from lllm import RemoteTactic, Tactic, create_tactic_app
from lllm.runtimes.native import tactic_as_function


class Ticket(BaseModel):
    text: str
    priority: int = 3


class Triage(Tactic):
    """Route a support ticket to a queue."""

    name = "triage"
    input_model = Ticket

    def call(self, ticket: Ticket) -> str:
        return "oncall" if "outage" in ticket.text.lower() else "support"


app = create_tactic_app(Triage())  # running on another machine


async def main():
    triage = await RemoteTactic.afrom_service(  # (1)!
        "http://triage.internal:8000",
        async_transport=httpx.ASGITransport(app=app),
    )
    print(triage.tactic_name, "|", triage.info().description)
    #> triage | Route a support ticket to a queue.
    tool = tactic_as_function(triage, parameter_mode="kwargs")
    print(tool.name, sorted(tool.properties), tool.required)
    #> triage ['priority', 'text'] ['text']
    print(await triage.arun({"text": "Total outage"}))
    #> oncall


asyncio.run(main())
```

1. `afrom_service()` is the async form of `from_service()`. Both take the
   constructor's options and use them for the `/info` request too.

The contract arrives as the constructor's `contract=` option, which also
takes a `TacticInfo` you already have. With one, the tactic takes the
service's name unless you pass `name=`, and `info()` reports the service's
description, schemas, and examples wherever the tactic has none of its own,
so `input_type` and `output_type` still win. Its capabilities follow the
service's: `stream` and `events` come together when the service streams at
all, since the stream methods and `aevents()` all read `/stream`, and
neither comes when it does not.

Because it is a tactic, a `RemoteTactic` works wherever a tactic does:
inside another tactic's `call()`, as a [tool](tools.md#tactics-as-tools),
or bound to a ref in `.lllm/config.toml`, so code names it without knowing
where it runs ([Packages](packages.md#local-configuration)). Pass the
caller's `context` to `run()`, and its ids and tags reach the remote
tactic.

## Serving a package

A [package](packages.md) folder, such as one downloaded from the
[hub](publishing.md), serves the resource its manifest names as
`primary`:

```toml title="triage/lllm.toml"
[package]
org = "acme"
name = "triage"
version = "0.1.0"
primary = "tactics.triage"

[tactics.triage]
entry = "main:create_tactic"
description = "Route a support ticket to a queue."
metadata = {port = 8100}

[requirements.api_keys]
OPENAI_API_KEY = "Model access for the router agent."
```

```bash
lllm inspect ./triage   # what would be served, and which keys are missing
lllm setup ./triage     # store the missing keys
lllm serve ./triage     # serve tactics.triage on port 8100
```

- `--resource section.name` serves another resource, such as
  `services.api`, and must name one the manifest declares. Without
  `primary` or `--resource`, the first service, tactic, channel, or run is
  served; only those kinds can be. A run must list exactly one service.
  Channels need [SSSN](https://sssn.one/): a channel resource serves the
  package's whole channel store, in `<package>/.lllm/network` unless
  `--store` says otherwise. `lllm inspect` creates no store.
- The port is `--port`, else the resource's `metadata.port` or a `port`
  written in its table (a tactic without one takes the port of the first
  service that serves it, through `tactic` or `tactics`, and declares one),
  else 8000. The hub's config template binds refs to the same ports.
- Before starting, `lllm serve` checks the API keys the package declares
  ([API keys](packages.md#api-keys)). In a terminal it asks for missing
  ones; otherwise it stops and lists them. `--skip-key-check` skips this.
- A served package keeps its folder on `sys.path` and its modules loaded,
  so its code can import its own modules. Packages served in one process
  share `sys.modules`, so give their top-level modules distinct names.

To serve everything a package declares, tactics and channels, as one API,
mount it in an [AAAX](https://aaax.one/) strategy.
