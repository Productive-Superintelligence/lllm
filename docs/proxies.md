# Proxies

A proxy is an API an agent can read about and call. The agent finds a
directory of the API's endpoints in its system prompt and gets two tools:
`query_api_doc` to read an API's documentation, and `run_python` to run
Python in which `CALL_API` reaches every endpoint. An API with a hundred
endpoints costs two tool schemas instead of a hundred, and the agent can
fetch, join, and compute over the data in one short program. The proxy adds
the API key to each request, so the key never appears in the prompt.

Here a trip planner answers from a weather API:

```python title="trip_weather.py"
from lllm import BaseProxy, ProxyRegistrator, Tactic

FORECASTS = {
    "Tokyo": {"high": 24, "low": 17, "sky": "rain"},
    "Paris": {"high": 19, "low": 11, "sky": "sun"},
}


@ProxyRegistrator(  # (1)!
    path="weather",
    name="Weather API",
    description="Tomorrow's forecast for major cities.",
)
class WeatherProxy(BaseProxy):
    @BaseProxy.endpoint(  # (2)!
        category="forecast",
        endpoint="forecast/{city}",
        description="Tomorrow's forecast for a city.",
        params={"city*": (str, "Tokyo")},
        response={"high": 24, "low": 17, "sky": "rain"},
    )
    def forecast(self, params: dict) -> dict:
        return FORECASTS[params["city"]]


class TripPlanner(Tactic):
    name = "trip_planner"
    agent_group = ["planner"]

    def call(self, question: str) -> str:
        planner = self.agents["planner"]
        planner.open("trip")
        planner.receive(question)
        return planner.respond().content


planner = TripPlanner({
    "global": {"model_name": "gpt-4o-mini"},
    "agent_configs": [{
        "name": "planner",
        "system_prompt": "You help people plan trips.",
        "proxy": {"activate_proxies": ["weather"]},  # (3)!
    }],
})

session = planner(
    "Tokyo or Paris tomorrow: which will be drier?", return_session=True
)
print(session.delivery)
#> Paris will be drier: sunny, 11 to 19°C. Tokyo expects rain.
calls = session.agent_sessions["planner"][0].tool_calls
print([call.name for call in calls])  # (4)!
#> ['query_api_doc', 'run_python']
print(calls[1].arguments["code"])
"""
for city in ["Tokyo", "Paris"]:
    print(city, CALL_API(f"weather/forecast/{city}"))
"""
print(calls[1].result)
"""
Tokyo {'high': 24, 'low': 17, 'sky': 'rain'}
Paris {'high': 19, 'low': 11, 'sky': 'sun'}
"""
```

1. `@ProxyRegistrator` registers the class under the key `weather`, the
   first segment of every `CALL_API` path.
2. An endpoint is a method plus a description of its path and parameters.
   This proxy computes answers itself; an HTTP proxy returns the parameters
   to send instead ([HTTP APIs and keys](#http-apis-and-keys)).
3. The `proxy` block gives the agent the API: the endpoint directory is
   appended to its system prompt, and `query_api_doc` and `run_python` join
   its tools for every turn.
4. The model read the docs, then wrote one program that called the endpoint
   for both cities and printed only what it needed.

## Defining endpoints

Decorate methods of a `BaseProxy` subclass with `@BaseProxy.endpoint`. Each
method receives the call's parameters as one dict, path values included.
`ProxyManager` loads registered proxies and dispatches calls; the agent's
`CALL_API` is a `ProxyManager`, and you can call one from Python too:

```python title="fx_proxy.py"
from lllm import BaseProxy, ProxyManager, ProxyRegistrator

RATES = {("USD", "EUR"): 0.92, ("USD", "JPY"): 151.3}


@ProxyRegistrator(path="fx", name="FX Rates", description="Exchange rates.")
class FXProxy(BaseProxy):
    @BaseProxy.endpoint(
        category="rates",
        endpoint="rates/{base}",  # (1)!
        description="Latest rate from base to quote.",
        params={"base*": (str, "USD"), "quote*": (str, "EUR")},  # (2)!
        response={"base": "USD", "quote": "EUR", "rate": 0.92},
    )
    def rate(self, params: dict) -> dict:
        rate = RATES[params["base"], params["quote"]]
        return {**params, "rate": rate}


fx = ProxyManager(activate_proxies=["fx"])
print(fx.retrieve_api_docs())  # (3)!
"""
## FX Rates (fx)
Exchange rates.
- **rates/{base}** [GET] – Latest rate from base to quote.
    - `base*` (str) e.g. USD
    - `quote*` (str) e.g. EUR
"""
print(fx("fx/rates/USD", {"quote": "JPY"}))  # (4)!
#> {'base': 'USD', 'quote': 'JPY', 'rate': 151.3}
```

1. Paths may contain `/`, `-`, and `{name}` placeholders, which are filled
   from the path the agent calls.
2. Each parameter is `(type, example)`, and a trailing `*` marks it required
   in the docs. LLLM does not validate parameters, so check them in the
   method and raise a clear error: the agent reads it in the traceback.
3. This is the text the agent reads. It is appended to the system prompt,
   and `query_api_doc("fx")` returns the same section.
4. The path as listed (`fx/rates/USD`), the template with the value in the
   params (`fx/rates/{base}`), and the method name (`fx.rate`) all reach
   this endpoint. Parameters can also be keyword arguments, alone or next
   to the dict.

The other arguments of `@BaseProxy.endpoint` are `method` (`"GET"`, or
`"POST"` to send a JSON body), `remove_keys` (keys to drop from each item of
the response), and `name`, `sub_category`, and the `response` example, which
are kept in `fx.api_catalog()`, the structured version of the docs. Only
decorated methods are reachable: any other path raises an error that lists
the valid ones.

## HTTP APIs and keys

For a web API, set `base_url` and the key, and let each endpoint method
return the parameters to send. The proxy builds the URL, adds the key, sends
the request, and returns the decoded JSON:

```python title="github_proxy.py"
import os

from lllm import BaseProxy, ProxyRegistrator


@ProxyRegistrator(path="github", name="GitHub", description="Repo issues.")
class GitHubProxy(BaseProxy):
    base_url = "https://api.github.com"
    api_key_name = "*Authorization"  # (1)!

    def __init__(self, **kwargs):
        super().__init__(use_cache=False, **kwargs)  # (2)!
        token = os.getenv("GITHUB_TOKEN")
        self.api_key = f"Bearer {token}" if token else None

    @BaseProxy.endpoint(
        category="issues",
        endpoint="repos/{owner}/{repo}/issues",
        description="Issues of a repository, newest first.",
        params={
            "owner*": (str, "pydantic"),
            "repo*": (str, "pydantic"),
            "state": (str, "open"),
        },
        response=[{"number": 1, "title": "..."}],
    )
    def issues(self, params: dict) -> dict:
        params.setdefault("state", "open")  # (3)!
        return params


class RecordedGitHub(GitHubProxy):  # (4)!
    def _call_api(self, url, params, endpoint_info, headers):
        return url, params


github = RecordedGitHub()
print(github.call("repos/pydantic/pydantic/issues"))
"""
('https://api.github.com/repos/pydantic/pydantic/issues', {'state': 'open'})
"""
odd = {"owner": "a b", "repo": "c/d"}
print(github.call("repos/{owner}/{repo}/issues", odd))  # (5)!
#> ('https://api.github.com/repos/a%20b/c%2Fd/issues', {'state': 'open'})
```

1. The key goes in a query parameter named `api_key_name`, or in a header
   when the name starts with `*`.
2. GET responses are cached on disk by default (under `~/.lllm/.cache`, or
   `$TMP_DIR/.cache`), keyed by URL and parameters, with no expiry.
   `use_cache=False` turns the cache off, both reads and writes, for data
   that changes. POST responses are never cached.
3. The endpoint method is the place for defaults, renames, and checks.
4. `_call_api(url, params, endpoint_info, headers)` sends the request.
   Override it for APIs that answer in plain text or need an SDK, or, as
   here, to test a proxy without the network.
5. Each placeholder value is URL-encoded as exactly one path segment, and an
   empty value, `.`, or `..` is an error, so a request (and the key it
   carries) cannot leave the documented endpoint.

HTTP errors raise, and so do JSON responses with an `error` key; the agent
sees either as a traceback. A proxy with neither a `base_url` nor its own
`_call_api` returns what its endpoint methods return, like `WeatherProxy`.

## Giving proxies to agents

Add a `proxy` block to an agent's config, or to `global` for every agent.
An agent's block is merged over the global one key by key:

```yaml
global:
  model_name: gpt-4o-mini
  proxy:
    activate_proxies: [fred, fx]
agent_configs:
  - name: analyst
    system_prompt: You analyze macro data.
  - name: quick
    system_prompt: You answer in one sentence.
    proxy:
      timeout: 10
```

| Key | Default | Meaning |
| --- | --- | --- |
| `activate_proxies` | `[]` | Proxies to load, by key (`fx`) or package ref; empty loads every registered proxy |
| `exec_env` | `interpreter` | `interpreter` adds `run_python`; any other value (`jupyter`, `null`) adds only `query_api_doc`, and your tactic runs the code ([Sandbox](sandbox.md)) |
| `timeout` | `60.0` | Seconds per `run_python` call |
| `max_output_chars` | `5000` | Longer `run_python` output loses its middle; `0` turns this off |
| `truncation_indicator` | `... (truncated)` | Marks where output was cut |
| `cutoff_date` | `null` | ISO date given to each proxy, for proxies that limit data by date |
| `deploy_mode` | `false` | Given to each proxy; LLLM itself does not read it |
| `prompt_template` | `null` | Replaces the instructions added to the system prompt |

A proxy from a [package](packages.md) can also go in `tools:` by its full
ref (`pkg.proxies:name`). It joins `activate_proxies`, with the default
settings if the agent has no `proxy` block. A bare name in `tools:` is read
as a tool, not a proxy:

```yaml
agent_configs:
  - name: analyst
    system_prompt: You analyze markets.
    tools: [search, market_pkg.proxies:fmp]
```

The directory in the system prompt lists every endpoint of every proxy,
which is long for a large API. To leave it out, set `prompt_template` to
instructions without `{api_directory}`; the agent then reads the docs it
needs through `query_api_doc`. A template can use `{api_directory}`,
`{max_output_chars}`, and `{truncation_indicator}`, and must double any
other braces.

## The interpreter

`run_python` runs code in an `AgentInterpreter`, one per agent, built fresh
for every tactic call:

```python title="interpreter.py"
from lllm import ProxyManager
from lllm.runtimes.native.proxies import AgentInterpreter

python = AgentInterpreter(ProxyManager(), max_output_chars=40)
python.run("squares = [n * n for n in range(100)]")
print(python.run("print(len(squares))"))  # (1)!
"""
100
"""
print(python.run("print(squares)"))  # (2)!
"""
[0, 1, 4, 9, 16, 25,... (truncated), 9409, 9604, 9801]
"""
```

1. Variables persist between calls, and only what the code prints comes
   back; the last expression is not echoed. Agents can run code at the
   same time, as under `bcall`: each call captures what its code prints,
   including the endpoint methods `CALL_API` runs and the threads the code
   starts. A thread kept from an earlier call, such as a pool worker in a
   variable, prints into the agent's newest running call; outside calls,
   threads print to your program.
2. Longer output loses its middle: the first and the last 20 characters
   (half of `max_output_chars` each) are kept, so the end of a traceback,
   where the exception is, survives. The instructions in the system prompt
   tell the agent to print only what it needs.

An exception comes back as its traceback, so the agent can fix the code and
try again with its variables intact. The code's own redirects
(`redirect_stdout`, assigning `sys.stdout`, doctest) work, but `sys.stdout`
is one per process: while a redirect lasts, other running calls and your
program print into it too. When the last call ends, your program's
`sys.stdout` is put back.

A call that runs past `timeout` returns
`Error: Code execution timed out after 60s...` to the model, but the code is
not stopped: it keeps running in a background thread, what it changes stays
in the namespace, and what it and its threads print is dropped. Once that
code ends, its threads print to your program, and `sys.stdout` is put back
unless your program has set its own since the timeout. `reset()` clears
every variable except `CALL_API`.

!!! warning "Not a security boundary"
    The interpreter runs `exec()` inside your process, with your files,
    network, and environment variables. It suits trusted agents calling
    your APIs. For code you do not trust, run it in a container or on a
    machine you control; see [Sandbox](sandbox.md#safety).

## Built-in proxies

LLLM ships proxies for research data. Each reads its API key from the
environment. Call `load_builtin_proxies()` (from
`lllm.runtimes.native.proxies`) before you build a tactic that activates
them: a name in `activate_proxies` that nothing registered is skipped,
with a warning in the log. It returns `(loaded, errors)`, and a module
that cannot be imported, such as `exa` without `exa_py`, lands in `errors`
instead of raising.

| Key | API | Key variable |
| --- | --- | --- |
| `fmp` | Financial Modeling Prep: prices, statements, filings | `FMP_API_KEY` |
| `fred` | Federal Reserve Economic Data | `FRED_API_KEY` |
| `exa` | Exa web search (needs `pip install exa_py`) | `EXA_API_KEY` |
| `gt` | Google Trends, through SearchApi | `SEARCH_API_KEY` |
| `msd` | Main Street Data | `MSD_API_KEY` |
| `wa` | Wolfram Alpha LLM API | `WA_API_DEV` |

`fmp`, `fred`, `gt`, and `exa` honor `cutoff_date`, keeping requests
before that date for backtests. Their source
(`lllm/runtimes/native/proxies/builtin/`) shows query and header keys,
cutoff handling, and `_call_api` overrides.

## Proxies in packages

In a project, put proxy classes in a folder that `lllm.toml` lists:

```toml
[proxies]
paths = ["proxies"]
```

LLLM imports every file there and registers each `BaseProxy` subclass as
`<package>.proxies:<path>`, using the path `@ProxyRegistrator` set, a
`_proxy_path` class attribute, or else `<file>/<ClassName>`.
`lllm create my-api --template proxy` starts
a project with a sample proxy, a config that attaches it to an agent, and
`python main.py --catalog` to print its docs. See [Packages](packages.md).

An endpoint can also run a whole tactic:
`BaseProxy.tactic_endpoint("my_pkg.tactics:summarize", config=...)` takes
its parameters from the tactic's input and returns the tactic's result.
When the tactic has one required parameter, its value alone works in
place of the params dict, as in `CALL_API("tools/summarize", text)`.

!!! note "`ProxyTactic` is something else"
    `lllm.proxy_tactic()` wraps a whole tactic at its typed boundary with
    `before`, `after`, and `on_error` hooks and records each call. It has
    nothing to do with API proxies; see
    [Observability](observability.md#records-at-the-tactic-boundary).
