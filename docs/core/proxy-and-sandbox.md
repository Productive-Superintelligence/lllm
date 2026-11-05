# Proxies & Sandbox

LLLM treats API access and code execution as first-class capabilities. Proxies
standardize how agents reach external APIs; execution environments let agents
run the Python they write.

A proxy is not just another way to define a function tool. It is the
**programming-based** tool surface: the agent reads endpoint docs with
`query_api_doc`, writes small Python programs with `run_python`, and calls
registered endpoints through `CALL_API`. Use proxies when an agent needs many
related endpoints, stateful analysis, or a small sandbox; use `@tool` functions
when the model only needs a few direct schemas.

The native runtime ships two execution environments:

- **Interpreter**: an in-process `exec()` with a persistent namespace. Zero
  start-up cost and safe to run many agents in parallel. Best for fast data
  retrieval and computation.
- **Jupyter**: a real kernel driven through `JupyterSession`, producing `.ipynb`
  artifacts and supporting plots. Install `lllm-core[sandbox]`.

Both are optional, and any object with a "run this code, return output" method
can take Jupyter's place; see [Custom execution environments](#custom-execution-environments).

```python
from lllm import BaseProxy, ProxyManager, ProxyRegistrator
from lllm.runtimes.native.proxies import AgentInterpreter, load_builtin_proxies
```

---

## Overview: execution modes

When an agent has a `proxy:` block or a proxy ref in its config `tools`, it gets
API awareness automatically. `exec_env` decides how it uses the APIs:

| `exec_env` | Tools injected | What the agent does | Who runs code |
| --- | --- | --- | --- |
| `"interpreter"` (default) | `query_api_doc` + `run_python` | Calls `run_python` with Python | LLLM's `AgentInterpreter`, in process |
| `"jupyter"` | `query_api_doc` | Writes `<python_cell>` tags | Your tactic, through `JupyterSession` |
| `null` (or any other string) | `query_api_doc` | Uses the API directory for awareness | Nobody (or your own runner) |

Every mode appends an API directory block to the system prompt and adds
`query_api_doc` to its tools. Like all of an agent's tools, they are offered on
every turn.

---

## Proxy config reference

Set under `proxy:` in an agent entry, or under `global:` for all agents.
Per-agent values deep-merge over global ones, so an agent can change one field
(`exec_env`, say) without repeating the rest.

```yaml
proxy:
  activate_proxies: [fmp, fred]        # which proxies to load; empty = every registered proxy
  deploy_mode: false                   # passed to proxy instances
  cutoff_date: "2024-01-01"            # ISO date passed to proxies (data cutoff)
  exec_env: interpreter                # "interpreter" | "jupyter" | null
  max_output_chars: 5000               # truncate run_python output (0 = no limit)
  truncation_indicator: "... (truncated)"
  timeout: 60.0                        # seconds per run_python call
  prompt_template: null                # replace the injected system-prompt block
```

For one packaged proxy with default settings, a `tools` ref is enough:

```yaml
global:
  model_name: gpt-4o
  tools:
    - market_pkg.proxies:fmp
```

Proxy refs in `tools` must be explicit (`pkg.proxies:name`). They are added to
`activate_proxies` (merged with any `proxy:` block). Use `proxy:` when you need
to tune `exec_env`, timeouts, truncation, or the cutoff date.

```yaml
global:
  model_name: gpt-4o
  proxy:
    activate_proxies: [fmp, fred]
    exec_env: interpreter

agent_configs:
  - name: researcher
    system_prompt_path: research/system
    # inherits the global proxy block

  - name: notebook_analyst
    system_prompt_path: analyst/notebook
    proxy:
      exec_env: jupyter       # this agent writes notebook cells instead

  - name: macro_analyst
    system_prompt_path: analyst/macro
    proxy:
      activate_proxies: [fred]
      timeout: 30.0
```

A custom `prompt_template` is formatted with `{api_directory}`,
`{max_output_chars}`, and `{truncation_indicator}`; double any other braces.
The defaults are `DEFAULT_PROXY_PROMPT_TEMPLATE` (interpreter) and
`DEFAULT_PROXY_PROMPT_NO_PYTHON_TEMPLATE` (other modes) in
`lllm.runtimes.native.proxies`.

Agents are rebuilt for every tactic call, so each call gets a fresh
`ProxyManager` and a fresh interpreter namespace.

---

## Interpreter mode (default)

The agent calls `run_python` with code. LLLM executes it in a persistent
namespace and returns what the code printed. Variables survive from one call to
the next within the agent.

### Why not Jupyter?

For most API work the interpreter is the better default:

- **No overhead**: no subprocess, no kernel start, no files.
- **Parallel-safe**: each agent has its own namespace, so `bcall` over many
  tasks works.
- **Instant**: no kernel boot per agent.

Use Jupyter when you need notebook files, plots, or a cell-by-cell record of the work.

### What gets injected

When the agent is built, LLLM appends a block to the system prompt explaining
the interpreter, `CALL_API`, printing, and truncation, followed by the API
directory, and adds two tools:

- **`query_api_doc(proxy_name)`**: full endpoint docs for one proxy.
- **`run_python(code)`**: execute Python in the persistent interpreter.

The agent writes code like:

```python
obs = CALL_API("fred/series/observations", {"series_id": "GDP", "limit": 4})
print(obs)
```

`CALL_API("<proxy>/<endpoint path>", params)` takes the proxy key followed by an
endpoint path exactly as the API directory lists it (see
[How a call is made](#how-a-call-is-made)). Params can also be passed as keyword
arguments: `CALL_API("fred/series/observations", series_id="GDP")`.

Only stdout comes back, so code must `print()` what matters. Exceptions come
back as tracebacks. Output longer than `max_output_chars` is cut and the
indicator appended. A call that runs past `timeout` returns
`Error: Code execution timed out after ...`; the namespace keeps whatever the
code changed before the timeout.

The interpreter uses `exec()` in your process: it is meant for trusted
agent code that calls APIs and processes data; it does not isolate that code from
your process.

### Using the interpreter directly

```python
from lllm import BaseProxy, ProxyManager, ProxyRegistrator
from lllm.runtimes.native.proxies import AgentInterpreter


@ProxyRegistrator(path="calc", name="Calculator", description="Local arithmetic.")
class CalcProxy(BaseProxy):          # no base_url: endpoints return their own result
    @BaseProxy.endpoint(
        category="math",
        endpoint="add",
        description="Add two numbers.",
        params={"a*": (float, 1.0), "b*": (float, 2.0)},
        response={"sum": 3.0},
    )
    def add(self, params: dict) -> dict:
        return {"sum": params["a"] + params["b"]}


interpreter = AgentInterpreter(
    ProxyManager(activate_proxies=["calc"]),
    max_output_chars=2000,
    timeout=10.0,
)

print(interpreter.run('r = CALL_API("calc/add", {"a": 2, "b": 3})\nprint(r)'))
print(interpreter.run("print(r['sum'] * 10)"))   # state persists
interpreter.reset()                               # clear variables, keep CALL_API
```

### Full example

```python
from lllm import Prompt

analyst_system = Prompt(
    path="analyst/system",
    prompt="You are a financial analyst. Answer the user's question with data.",
)
```

```yaml
global:
  model_name: gpt-4o
  model_args:
    temperature: 0.1

agent_configs:
  - name: analyst
    system_prompt_path: analyst/system
    proxy:
      activate_proxies: [fmp]
      max_output_chars: 8000
```

The analyst agent now has `run_python` and `query_api_doc` on its system prompt,
and they are offered on every turn, whichever prompt drives it (unless a prompt
sets `agent_tools=False`); no tactic-level wiring is needed. See
[Agent](agent.md#where-tools-and-parsers-come-from).

---

## Jupyter mode

In Jupyter mode the agent writes `<python_cell>` and `<markdown_cell>` tags in
its reply. Your tactic extracts the cells, runs them in a `JupyterSession`, and
feeds the output back.

Use it when you want notebooks as artifacts, matplotlib/plotly figures,
cell-level error recovery, or a reproducible record of the analysis.

```bash
pip install "lllm-core[native,sandbox]"
```

### Config

```yaml
proxy:
  activate_proxies: [fred]
  exec_env: jupyter    # no run_python; the agent writes cell tags
```

Only `query_api_doc` is injected. The API directory block is still appended (a
shorter version without interpreter instructions), and your system prompt
explains the notebook interface.

### System prompt and parser

```python
from lllm import Prompt
from lllm.runtimes.native import ParseError
from lllm.runtimes.native.utils import find_all_xml_tags_sorted

NOTEBOOK_SYSTEM = """
You are an expert analyst working in a Jupyter notebook.

Every response must contain one or more cells:
1. Python cells: wrap code in <python_cell>...</python_cell>.
2. Markdown cells: wrap notes in <markdown_cell>...</markdown_cell>.

Rules:
- Cells run in order; variables persist across turns.
- If a cell fails you will see the error; fix that cell only.
- When the analysis is complete, reply with <TERMINATE_NOTEBOOK> alone.

CALL_API(api_path, params) is available in Python cells.
Call query_api_doc before using a new endpoint.
"""


def notebook_parser(message: str, **kwargs):
    matches = find_all_xml_tags_sorted(message)
    terminate = "<TERMINATE_NOTEBOOK>" in message
    cells = [(m["tag"], m["content"]) for m in matches if m["tag"] in ("python_cell", "markdown_cell")]
    if not cells and not terminate:
        raise ParseError(
            "No cells found. Wrap code in <python_cell>...</python_cell> "
            "or notes in <markdown_cell>...</markdown_cell>."
        )
    return {"cells": cells, "terminate": terminate}


notebook_turn = Prompt(path="analyst/notebook_turn", prompt="{content}", parser=notebook_parser)
```

The system prompt (`NOTEBOOK_SYSTEM`, registered as `analyst/notebook`) sets
the interface; `notebook_turn` carries the parser, so every turn the tactic
sends with it is validated and repaired by the agent loop.

### Tactic wiring with `JupyterSession`

The tactic drives the notebook loop. `JupyterSandbox(config)` manages session
folders (under `~/.lllm/sandbox/<name>/sessions/` by default) and
`new_session()` creates a notebook. Cells are appended with
`append_code_cell()` / `append_markdown_cell()`, which return the cell index,
and executed with `run_cell(index)`, which starts the kernel on first use and
returns `True` on success. Outputs are stored on the notebook cell.

```python
from lllm import Tactic
from lllm.runtimes.native.sandbox import JupyterSandbox

SETUP_CELL = """\
from lllm.runtimes.native.proxies import ProxyManager, load_builtin_proxies
load_builtin_proxies()                      # or import / load your own proxy package
CALL_API = ProxyManager(activate_proxies=["fred"]).__call__
"""


def cell_text(session, index: int) -> str:
    parts = []
    for out in session.cells[index].outputs:
        if out.output_type == "stream":
            parts.append(out.text)
        elif out.output_type in ("execute_result", "display_data"):
            parts.append(out.data.get("text/plain", ""))
        elif out.output_type == "error":
            parts.append(f"{out.ename}: {out.evalue}")
    return "".join(parts)


class NotebookAnalysis(Tactic):
    name = "notebook_analysis"
    agent_group = ["analyst"]

    def call(self, task: str, max_turns: int = 10) -> str:
        sandbox = JupyterSandbox({"name": "analysis", "project_root": ".", "activate_proxies": []})
        session = sandbox.new_session()
        session.run_cell(session.append_code_cell(SETUP_CELL))

        agent = self.agents["analyst"]
        turn = agent.runtime.get_prompt("analyst/notebook_turn")
        agent.open("main")
        content = task
        try:
            for _ in range(max_turns):
                agent.receive_prompt(turn, {"content": content})
                parsed = agent.respond().parsed
                if parsed["terminate"]:
                    break
                outputs = []
                for tag, code in parsed["cells"]:
                    if tag == "markdown_cell":
                        session.append_markdown_cell(code)
                        continue
                    index = session.append_code_cell(code)
                    ok = session.run_cell(index)
                    status = "ok" if ok else "error"
                    outputs.append(f"[cell {index}: {status}]\n{cell_text(session, index)}")
                content = "\n\n".join(outputs) or "No code cells were run."
        finally:
            session.shutdown()
        return session.notebook_file
```

`JupyterSession` reserves cell 0 for an init cell it writes itself: it builds
`CALL_API` from a `ProxyManager` with the session's `activate_proxies`. A fresh
kernel has no proxy classes registered, so that manager starts empty; register
your proxies in the kernel (as the setup cell does) and build `CALL_API` after
they are registered. Other useful members: `cells`, `n_cells`,
`get_cells(indices)`, `run_all_cells(stop_on_error=True)`,
`delete_cells(indices)`, `launch_server()` (open the notebook in a browser), and
`shutdown()`.

---

## Custom execution environments

Nothing forces you to use the built-in sandboxes. The tactic-drives-execution
pattern works with any code runner, because the runner is just an object in
your tactic's `call()`.

### The pattern

The agent side (prompt + parser) stays the same wherever code runs:

1. The agent writes `<python_cell>` / `<markdown_cell>` tags.
2. The tactic reads `parsed["cells"]`.
3. The tactic runs each code cell and feeds the output back.
4. Repeat until `parsed["terminate"]` is `True`.

Only step 3 changes when you swap sandboxes.

### Example: remote execution service

```python
import httpx

from lllm import Tactic


class RemoteSandbox:
    """Execute cells on a remote code runner."""

    def __init__(self, base_url: str, session_id: str):
        self.base_url = base_url
        self.session_id = session_id
        self.last_cell_failed = False

    def run_cell(self, code: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/execute",
            json={"session_id": self.session_id, "code": code},
            timeout=60,
        )
        result = resp.json()
        self.last_cell_failed = result.get("error", False)
        return result.get("output", "")


class RemoteAnalysis(Tactic):
    name = "remote_analysis"
    agent_group = ["analyst"]

    def call(self, task: str, **kwargs) -> str:
        agent = self.agents["analyst"]
        turn = agent.runtime.get_prompt("analyst/notebook_turn")
        sandbox = RemoteSandbox("https://runner.example.com", session_id="abc123")

        agent.open("main")
        content = task
        for _ in range(10):
            agent.receive_prompt(turn, {"content": content})
            parsed = agent.respond().parsed
            if parsed["terminate"]:
                break
            outputs = []
            for tag, code in parsed["cells"]:
                if tag == "python_cell":
                    output = sandbox.run_cell(code)
                    label = "failed" if sandbox.last_cell_failed else "output"
                    outputs.append(f"[{label}]\n{output}")
            content = "\n\n".join(outputs) or "No code cells were run."

        return agent.current_dialog.tail.content
```

Any executor that takes code and returns output works: Docker containers,
WebAssembly runtimes, database query runners, or a fake sandbox in tests. The
prompt and parser stay identical.

### When to use each option

| | Interpreter | Jupyter (built-in) | Custom sandbox |
| --- | --- | --- | --- |
| Setup | None | Kernel start (seconds) | You provide it |
| Proxy integration | Automatic (`CALL_API` injected) | A setup cell defines `CALL_API` | Inject it yourself |
| Parallel agents | Safe | Heavy | Depends on your implementation |
| Visualizations | No | Yes (matplotlib/plotly) | Depends |
| Output artifact | None | `.ipynb` notebook | Whatever you return |
| Best for | Fast API orchestration | Exploratory analysis, charts | Isolation, testing, non-Python runtimes |

---

## Writing a proxy

A proxy is a `BaseProxy` subclass. Each endpoint is a method decorated with
`@BaseProxy.endpoint`, and the class says where requests go:

```python
import os

from lllm import BaseProxy, ProxyRegistrator


@ProxyRegistrator(path="fmp", name="Financial Modeling Prep", description="Market data API.")
class FMPProxy(BaseProxy):
    base_url = "https://financialmodelingprep.com/stable"
    api_key_name = "apikey"                 # sent as ?apikey=<api_key>

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("FMP_API_KEY")

    @BaseProxy.endpoint(
        category="search",
        endpoint="search-symbol",
        description="Search tickers by company name or symbol.",
        params={"query*": (str, "AAPL"), "limit": (int, 10)},
        response=[{"symbol": "AAPL", "name": "Apple Inc."}],
    )
    def search_symbol(self, params: dict) -> dict:
        params.setdefault("limit", 10)      # validate, add defaults, rename keys
        return params

    @BaseProxy.endpoint(
        category="company",
        endpoint="profile",
        description="Company profile for a ticker.",
        params={"symbol*": (str, "AAPL")},
        response=[{"symbol": "AAPL", "companyName": "Apple Inc."}],
    )
    def profile(self, params: dict) -> dict:
        return params
```

`CALL_API("fmp/search-symbol", {"query": "AAPL"})` sends
`GET https://financialmodelingprep.com/stable/search-symbol?query=AAPL&limit=10&apikey=...`
and returns the decoded JSON.

`@BaseProxy.endpoint(category, endpoint, description, params, response,
name=None, sub_category=None, remove_keys=None, dt_cutoff=None, method="GET")`
records the endpoint:

- `endpoint` is the path under `base_url`, and the name the API directory and
  `CALL_API` use. It may contain `/`, hyphens, and `{name}` placeholders
  (`series/{series_id}/observations`); any placeholder name works, `{series-id}`
  included.
- `params` maps each parameter to `(type, example)`; a trailing `*` marks it
  required. `response` is an example response.
- `method` is `"GET"` or `"POST"`; `remove_keys` lists keys to drop from the
  response.

`ProxyRegistrator(path, name, description, runtime=None, *, registry=None)`
registers the class (in the default runtime unless you pass one) and sets the
key `path` that `activate_proxies` and `CALL_API` use.

`lllm create my-proxy --template proxy` starts a project whose
`proxies/sample_proxy.py` follows these conventions without the decorator: it
sets `_proxy_path = "sample"`, its endpoint methods take the params dict, and
`account_id*` marks a required parameter. `python main.py --catalog` prints its
API docs.

### How a call is made

`CALL_API(endpoint, params)` is `ProxyManager.__call__`, which hands the call to
the proxy's `call(path, params)`:

1. **Pick the proxy.** The manager matches the longest loaded proxy key that
   starts the endpoint string (`fmp` in `fmp/search-symbol`, `acme/notes` in
   `acme/notes/notes`) and passes on the rest as the path.
2. **Resolve the endpoint.** The path is matched against the endpoints' `endpoint`
   values, exactly as listed in the API directory, then against `{name}`
   templates, whose values are taken from the path. As a fallback the Python
   method name works too (`"fmp.search_symbol"`). Only decorated endpoints are
   reachable; other attributes of the proxy never are.
3. **Shape the params.** The endpoint method receives the params (path values
   included) and returns the params to send.
4. **Build the request.** With a `base_url`, the URL is `<base_url>/<endpoint>`
   with templates filled in, and the instance's `api_key` is added: as a query
   parameter named `api_key_name`, or as a header when the name starts with `*`
   (`api_key_name = "*x-api-key"` sends header `x-api-key`). Each `{name}` takes
   its value out of the params, URL-encoded as exactly one path segment
   (`{"ticker": "BRK/A"}` fills `companies/{ticker}` as `companies/BRK%2FA`),
   so pass raw values, not encoded ones. A missing or empty value, `.`, or `..`
   raises `ValueError` before anything is sent.
5. **Send it.** `_call_api(url, params, endpoint_info, headers)` performs the
   request. The default sends GET or POST by the endpoint's `method`, raises on
   HTTP errors and on JSON responses carrying an `error` key, drops
   `remove_keys`, and caches responses on disk while `use_cache` is true (the
   default; under `~/.lllm/.cache/`, or `$TMP_DIR/.cache/`). Override
   `_call_api` for APIs that answer in plain text or need custom authentication.

A proxy with neither a `base_url` nor its own `_call_api` returns whatever the
endpoint method returns, which suits local computation (the `CalcProxy` above).
One that overrides `_call_api` without a `base_url`, such as an SDK client (the
built-in `exa`), gets the filled-in endpoint path instead of a URL, with the
same placeholder rules. Tactic-backed endpoints always return the tactic's
result.

You can call a proxy without a manager: `FMPProxy().call("search-symbol",
{"query": "AAPL"})`, and `resolve_endpoint(path)` returns the method, its
metadata, and the path values. `endpoint_directory()`, `api_directory()`, and
`auto_test()` (a metadata check) read the decorators.

### A plain-text API

Wolfram Alpha's LLM API answers in text, so this proxy overrides `_call_api`:

```python
import os

import requests

from lllm import BaseProxy, ProxyRegistrator


@ProxyRegistrator(
    path="wolfram",
    name="Wolfram Alpha",
    description="Computations and factual answers from Wolfram Alpha.",
)
class WolframProxy(BaseProxy):
    base_url = "https://www.wolframalpha.com/api/v1"
    api_key_name = "appid"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("WA_API_DEV")

    def _call_api(self, url, params, endpoint_info, headers):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return {"response": resp.text}

    @BaseProxy.endpoint(
        category="query",
        endpoint="llm-api",
        name="Wolfram Alpha LLM API",
        description="Natural-language query; returns a text answer.",
        params={"input*": (str, "10 densest elemental metals"), "assumption": (str, None)},
        response={"response": "Input interpretation: ..."},
    )
    def query(self, params: dict) -> dict:
        return params
```

Key points:

- `__init__` receives `cutoff_date`, `activate_proxies`, `deploy_mode`, and the
  runtime through `**kwargs`; pass them on to `super().__init__`, which also
  accepts `use_cache`.
- Read API keys from the environment, and fail with a clear message when one is
  required and missing.
- The endpoint method is the place for per-endpoint logic; `_call_api` is the
  place for transport.

---

## Three ways proxies enter the registry

### 1. `@ProxyRegistrator` at import time

```python
@ProxyRegistrator(path="wa", name="Wolfram Alpha API", description="...")
class WAProxy(BaseProxy):
    ...
# registered under the key "wa" in the default runtime
```

### 2. Package discovery

```toml
[package]
name = "my_system"

[proxies]
paths = ["proxies"]
```

Discovery imports every `.py` file in the folder and registers each `BaseProxy`
subclass as `my_system.proxies:<path>`: the class's `_proxy_path`, which
`@ProxyRegistrator` sets and a class may also declare itself, else
`<file>/<ClassName>`.

### 3. `load_builtin_proxies()`

LLLM bundles proxies for research data APIs. Importing them registers them in
the default runtime:

```python
from lllm.runtimes.native.proxies import load_builtin_proxies

loaded, errors = load_builtin_proxies()
print("Loaded:", loaded)
print("Failed (missing deps):", errors)       # e.g. exa_proxy needs exa_py

load_builtin_proxies(["lllm.runtimes.native.proxies.builtin.wa_proxy"])   # selective
```

| Key | API | API key variable |
| --- | --- | --- |
| `fmp` | Financial Modeling Prep | `FMP_API_KEY` |
| `fred` | Federal Reserve Economic Data | `FRED_API_KEY` |
| `exa` | Exa search (needs `exa_py`) | `EXA_API_KEY` |
| `gt` | Google Trends through SearchApi | `SEARCH_API_KEY` |
| `msd` | Main Street Data | `MSD_API_KEY` |
| `wa` | Wolfram Alpha LLM API | `WA_API_DEV` |

`BUILTIN_PROXY_MODULES` lists the module paths. Use them as ready proxies, or
as worked examples of `base_url`, header and query-parameter keys, cutoff-date
handling, and `_call_api` overrides.

---

## `ProxyManager`

`ProxyManager` instantiates registered proxies and dispatches calls:

```python
from lllm import ProxyManager

pm = ProxyManager(activate_proxies=["fmp", "calc"], cutoff_date=None, deploy_mode=False)

pm.available()                   # ['calc', 'fmp']
pm.retrieve_api_docs()           # human-readable docs for every loaded proxy
pm.retrieve_api_docs("calc")     # one proxy (what query_api_doc returns)
pm.api_catalog()                 # {name: api_directory()} structured metadata
pm.get_api_directory("calc")

pm("calc/add", {"a": 1, "b": 2})       # same as CALL_API
pm("calc.add", {"a": 1, "b": 2})       # method-name form
pm("calc/add", a=1, b=2)               # keyword params
pm.register("custom/my_api", MyAPIProxy)   # add or replace at runtime
```

`ProxyManager(activate_proxies=None, cutoff_date=None, deploy_mode=False,
runtime=None, registry=None)` reads proxies from the given registry (default
runtime) when it is created. A proxy is loaded if any entry of
`activate_proxies` equals its qualified key (`my_system.proxies:fmp`), its bare
key (`fmp`), or its `@ProxyRegistrator` path. An empty list loads every
registered proxy. Loaded proxies are addressed by their bare key.

### Tactic-backed endpoints

A proxy endpoint can run a tactic:

```python
class SharedProxy(BaseProxy):
    code_review = BaseProxy.tactic_endpoint(
        "shared_pkg.tactics:code_review",
        config="shared_pkg:default",
    )


SharedProxy.register_tactic("shared_pkg.tactics:review", endpoint="review", config="shared_pkg:default")
```

The endpoint's params and response schema come from the tactic's tool schema, so
it shows up in `query_api_doc` like any endpoint and runs through `CALL_API`.
See [Tactics](tactic.md#tactics-as-tools).

---

## Related utilities

- **Computer use.** `lllm.runtimes.native.tools.cua` is an experimental
  browser-automation agent built on the OpenAI computer-use API and Playwright
  (`pip install "lllm-core[tools]"`). For provider-side computer use inside a
  normal agent turn, see `addon_args` in [Prompts](prompts.md#provider-capabilities).
- **Protocol wrappers.** `lllm.proxy_tactic` (call recording) and
  `lllm.sandbox_tactic` (timeouts and payload limits) wrap protocol tactics when
  you serve them. They are unrelated to native API proxies and code sandboxes.
