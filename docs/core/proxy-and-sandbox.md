# Proxies & Sandbox

LLLM treats external API access as a first-class capability. Proxies standardize how agents reach external APIs. The sandbox system enables agents to execute Python code.

LLLM ships two built-in execution environments:

- **Interpreter** — lightweight in-process `exec()`, zero overhead, parallel-safe. Best for fast data retrieval and computation.
- **Jupyter** — full notebook kernel via `JupyterSession`, supports visualisations and produces `.ipynb` artefacts. Optimised for the proxy system but usable for any code-execution workflow a tactic needs to drive.

Both are optional. You can also bring your own execution environment — any object with a `run_cell(code) -> str` method drops straight into the tactic pattern Jupyter uses. See [Custom Execution Environments](#custom-execution-environments) below.

---

## Overview: Execution Modes

When an agent is configured with `proxy:`, it gets automatic API awareness. The `exec_env` field controls *how* the agent uses those APIs:

| `exec_env` | Tools injected | What the agent does | Who runs code |
|-----------------|----------------|---------------------|---------------|
| `"interpreter"` (default) | `run_python` + `query_api_doc` | Calls `run_python` with Python | LLLM `AgentInterpreter` (in-process, parallel-safe) |
| `"jupyter"` | `query_api_doc` only | Writes `<python_cell>` tags | Your tactic via `JupyterSession` |
| `null` | `query_api_doc` only | Uses API directory for awareness only | — |
| *(future)* | configurable | extensible | custom sandbox |

All modes inject `query_api_doc` as a tool and append an API directory block to the system prompt.

---

## Proxy Config Reference

Set under `proxy:` in agent config (or under `global:` to apply to all agents).
Per-agent values deep-merge on top of global, so each agent can override
individual fields — including swapping `exec_env` — without repeating
everything.

```yaml
proxy:
  activate_proxies: [fmp, fred, exa]   # which proxies to load; empty = all
  deploy_mode: false                    # passed through to proxy instances
  cutoff_date: "2024-01-01"            # ISO date; restricts data range
  exec_env: interpreter          # "interpreter" | "jupyter" | null
  max_output_chars: 5000               # truncate run_python output (interpreter only)
  truncation_indicator: "... (truncated)"
  timeout: 60.0                        # seconds before TimeoutError (interpreter only)
  prompt_template: null                # override auto-selected system prompt block
```

```yaml
global:
  model_name: gpt-4o
  proxy:
    activate_proxies: [fmp, fred]
    deploy_mode: false
    exec_env: interpreter

agent_configs:
  - name: researcher
    # inherits global proxy config entirely

  - name: notebook_analyst
    proxy:
      exec_env: jupyter    # override: this agent uses JupyterSession
      # activate_proxies, deploy_mode etc. inherited from global

  - name: crypto_analyst
    proxy:
      activate_proxies: [fmp]    # override: only fmp
      timeout: 30.0              # override: shorter timeout
```

---

## Interpreter Mode (default)

The agent calls a `run_python` tool with Python code. LLLM executes it in a persistent in-process namespace and returns captured stdout. State accumulates across calls — variables from one call are available in the next.

### Why not Jupyter?

For most use cases, interpreter mode is preferred over a full Jupyter session:
- **Zero overhead** — no subprocess, no kernel startup, no file I/O
- **Parallel-safe** — each agent has its own namespace dict; hundreds of agents via `bcall` work fine
- **Instant** — no 2-3s kernel boot time per agent

Interpreter mode is **not** a drop-in replacement when you need: persistent notebook files, visualizations written to `.ipynb`, or reproducible cell-by-cell audit trails. For those, use Jupyter mode.

### Config

```yaml
proxy:
  activate_proxies: [fmp, fred]
  exec_env: interpreter     # default — can be omitted
  max_output_chars: 5000
  timeout: 60.0
```

### What gets injected

At agent build time, LLLM appends a block to the system prompt explaining the interpreter, then adds two tools to the prompt's `function_list`:

- **`query_api_doc(proxy_name)`** — retrieve full endpoint docs before first use
- **`run_python(code)`** — execute Python in the persistent interpreter

The agent writes code like:

```python
# Fetched in one call, processed in the next — variables persist
data = CALL_API("fmp/stock/historical-price-full", {"symbol": "AAPL", "from": "2024-01-01"})
print(data["historical"][:3])
```

Output is captured from stdout. Exceptions are returned as tracebacks. Output longer than `max_output_chars` is truncated with the indicator appended.

### Full example

```python
from lllm.core.prompt import Prompt

analyst_system = Prompt(
    path="analyst/system",
    prompt="You are a financial analyst. Answer the user's question using data.",
)
```

```yaml
# config.yaml
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

The analyst agent now has `run_python` and `query_api_doc` available. No tactic-level wiring needed.

---

## Jupyter Mode

In Jupyter mode, the agent writes `<python_cell>` and `<markdown_cell>` XML tags in its response text. Your tactic extracts these cells and runs them in a `JupyterSession`, feeding execution results back to the agent.

Use this when you need: notebook files as output artifacts, matplotlib/plotly visualizations, cell-level error recovery, or a reproducible audit trail.

### Config

```yaml
proxy:
  activate_proxies: [fmp, fred]
  exec_env: jupyter    # no run_python tool — agent writes cell tags instead
```

With `exec_env: jupyter`, only `query_api_doc` is injected as a tool. The API directory block is still appended (minimal version without interpreter instructions). Your system prompt explains the notebook interface.

### System prompt structure

The system prompt for a Jupyter agent has two parts:
1. **Your prompt** — explains the notebook interface (`<python_cell>`, `<markdown_cell>`, `<TERMINATE_NOTEBOOK>`)
2. **Auto-appended proxy block** — injects the API directory and `query_api_doc` hint

```python
from lllm.core.prompt import Prompt
from lllm.core.const import ParseError
from lllm.utils import find_all_xml_tags_sorted

NOTEBOOK_SYSTEM = """
You are an expert analyst. You work in a Jupyter Notebook environment.

## Jupyter Notebook Interface

Every response must contain one or more cells:

1. **Python cells**: wrap code in `<python_cell>...</python_cell>` tags.
   Use these for data retrieval, computation, and visualisation.
2. **Markdown cells**: wrap prose in `<markdown_cell>...</markdown_cell>` tags.
   Use these for notes, intermediate summaries, and structured analysis.

Rules:
- Cells are executed sequentially in the order you provide them.
- Variables persist across turns — you can reference data from previous cells.
- If a cell fails, you will receive the error and must fix that cell only.
- You cannot delete or modify previously submitted cells.
- When your analysis is complete, respond with `<TERMINATE_NOTEBOOK>` alone
  (no Python cells in the same response).

## API Library

`CALL_API(api_path, params)` is available in all Python cells.
Always call `query_api_doc` before using a new endpoint.
"""

def notebook_parser(message: str, **kwargs):
    matches = find_all_xml_tags_sorted(message)
    terminate = "<TERMINATE_NOTEBOOK>" in message
    cells = [
        (m["tag"], m["content"])
        for m in matches
        if m["tag"] in ("python_cell", "markdown_cell")
    ]
    if not cells and not terminate:
        raise ParseError(
            "No cells found. Wrap code in <python_cell>...</python_cell> "
            "or markdown in <markdown_cell>...</markdown_cell>."
        )
    return {"raw": message, "cells": cells, "terminate": terminate}

analyst_system = Prompt(
    path="analyst/system",
    prompt=NOTEBOOK_SYSTEM,
    parser=notebook_parser,
)
```

```yaml
# config.yaml
agent_configs:
  - name: analyst
    system_prompt_path: analyst/system
    proxy:
      activate_proxies: [fmp]
      allow_python: false
```

### Tactic wiring with JupyterSession

The tactic drives the notebook execution loop:

```python
from lllm.core.tactic import Tactic
from lllm.sandbox.jupyter import JupyterSession

class NotebookAnalysisTactic(Tactic):
    name = "notebook_analysis"
    agent_group = ["analyst"]

    def call(self, task: str, **kwargs):
        agent = self.agents["analyst"]
        session = JupyterSession(
            name="analysis",
            dir="/tmp/notebooks",
            metadata={
                "proxy": {
                    "activate_proxies": self.config.get("activate_proxies", []),
                    "deploy_mode": self.config.get("deploy_mode", False),
                }
            },
        )
        session.init_session()   # injects CALL_API into the kernel namespace

        agent.open("main")
        agent.receive(task)

        while True:
            msg = agent.respond()
            parsed = msg.parsed

            if parsed.get("terminate"):
                break

            # Execute cells and collect output
            cell_outputs = []
            for tag, code in parsed["cells"]:
                if tag == "python_cell":
                    output = session.run_cell(code)
                    cell_outputs.append(f"[Output]\n{output}")
                else:
                    session.add_markdown_cell(code)

            # Feed results back to agent
            if cell_outputs:
                agent.receive("\n\n".join(cell_outputs))

        return session.notebook_path
```

### JupyterSession init_code

`JupyterSession.init_session()` injects a setup cell that wires `CALL_API`:

```python
# Injected automatically — do not remove
import sys
sys.path.append('/path/to/project')
from lllm.proxies import ProxyManager
proxy = ProxyManager(
    activate_proxies=["fmp", "fred"],
    deploy_mode=False,
)
CALL_API = proxy.__call__
```

After this, any `CALL_API(...)` call in a notebook cell is dispatched through the `ProxyManager`.

---

## Custom Execution Environments

`JupyterSession` is the built-in sandbox, but nothing forces you to use it. The tactic-drives-execution pattern works with any code runner because the execution boundary is just a Python object in your tactic's `call()` method.

### The pattern

The agent side (prompt + parser) is the same regardless of where code runs:

1. Agent writes `<python_cell>` / `<markdown_cell>` tags.
2. Tactic extracts cells from `msg.parsed["cells"]`.
3. Tactic calls `sandbox.run_cell(code)` and feeds stdout back.
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

    def add_markdown_cell(self, text: str) -> None:
        pass   # or post to notebook service


class RemoteAnalysisTactic(Tactic):
    name = "remote_analysis"
    agent_group = ["analyst"]

    def call(self, task: str, **kwargs) -> str:
        agent = self.agents["analyst"]
        sandbox = RemoteSandbox("https://runner.example.com", session_id="abc123")

        agent.open("main")
        agent.receive(task)

        for _ in range(10):
            msg = agent.respond()
            parsed = msg.parsed
            if parsed["terminate"]:
                break
            outputs = []
            for tag, code in parsed["cells"]:
                if tag == "python_cell":
                    output = sandbox.run_cell(code)
                    if sandbox.last_cell_failed:
                        agent.receive(
                            f"Cell failed:\n\n{output}\n\n"
                            "Fix the error. Provide one <python_cell> only."
                        )
                        fix = agent.respond()
                        _, fixed_code = fix.parsed["cells"][0]
                        output = sandbox.run_cell(fixed_code)
                    outputs.append(f"[output]\n{output}")
                else:
                    sandbox.add_markdown_cell(code)
            if outputs:
                agent.receive("\n\n".join(outputs))

        return agent.current_dialog.tail.content
```

Any executor that accepts a code string and returns output string works: Docker containers, WebAssembly runtimes, database query runners, or even a mock sandbox for tests. The agent prompt and parser remain identical — only `sandbox` changes.

### When to use each option

| | Interpreter | Jupyter (built-in) | Custom sandbox |
|--|-------------|-------------------|----------------|
| Setup | Zero | ~2–3 s kernel boot | You provide it |
| Proxy integration | Automatic (`CALL_API` injected) | Automatic | Manual (inject yourself) |
| Parallel agents | Safe | Heavy | Depends on your impl |
| Visualisations | No | Yes (matplotlib/plotly) | Depends |
| Output artefact | None | `.ipynb` notebook | Whatever you return |
| Best for | Fast API orchestration | Exploratory analysis, charts | Isolated envs, testing, non-Python runtimes |

---

## BaseProxy & Endpoint Registration

`lllm/proxies/base.py` defines `BaseProxy`, a reflection-based registry for HTTP endpoints. Endpoints are declared via decorators:

```python
from lllm.proxies import BaseProxy, ProxyRegistrator

@ProxyRegistrator(path="finance/fmp", name="Financial Modeling Prep", description="Market data API")
class FMPProxy(BaseProxy):
    base_url = "https://financialmodelingprep.com/api/v3"
    api_key_name = "apikey"

    @BaseProxy.endpoint(
        category="search",
        endpoint="search-symbol",
        description="Search tickers",
        params={"query*": (str, "AAPL"), "limit": (int, 10)},
        response=[{"symbol": "AAPL", "name": "Apple"}],
    )
    def search_symbol(self, params):
        return params
```

`@ProxyRegistrator` registers the class into the runtime at decoration time (import time). The `path` argument becomes the resource key used by `activate_proxies` for filtering and by `ProxyManager` for dispatch.

`@BaseProxy.endpoint` is metadata-only — it records category, params, response schema, etc. for documentation and auto-testing. `endpoint_directory()`, `api_directory()`, and `auto_test()` consume this metadata.


## Three Ways Proxies Enter the Registry

### 1. `@ProxyRegistrator` at Import Time

The primary mechanism. When a module containing a decorated class is imported, the class is registered immediately:

```python
@ProxyRegistrator(path="wa", name="Wolfram Alpha API", description="...")
class WAProxy(BaseProxy):
    ...
# WAProxy is now registered under key "wa" in the default runtime
```

### 2. Discovery via `lllm.toml`

List proxy folders in the `[proxies]` section:

```toml
[package]
name = "my_system"

[proxies]
paths = ["proxies"]
```

Discovery recursively walks the folder, imports every `.py` file (triggering any `@ProxyRegistrator` decorators), and also scans for `BaseProxy` subclasses directly.

### 3. `load_builtin_proxies()` for Manual Use

LLLM ships with built-in proxies (financial data, search, Wolfram Alpha, etc.). In notebooks or scripts without an `lllm.toml`, import them manually:

```python
from lllm.proxies import load_builtin_proxies

loaded, errors = load_builtin_proxies()
print(f"Loaded: {loaded}")
print(f"Failed (missing deps): {errors}")
```

For selective loading:

```python
load_builtin_proxies(["lllm.proxies.builtin.wa_proxy", "lllm.proxies.builtin.fmp_proxy"])
```


## ProxyManager Runtime

`ProxyManager` composes multiple `BaseProxy` subclasses into a single callable dispatcher:

```python
from lllm.proxies import ProxyManager

pm = ProxyManager(activate_proxies=["fmp", "wa"], cutoff_date="2024-01-01")
result = pm("fmp/search_symbol", {"query": "AAPL"})
result = pm("wa/query", {"input": "10 densest metals"})
```

### How Activation Matching Works

A proxy matches if **any** of these equals an entry in `activate_proxies`:
- The qualified key (`"my_system.proxies:fmp"`)
- The bare key (`"fmp"`)
- The `_proxy_path` set by `@ProxyRegistrator` (`"fmp"`)

When `activate_proxies` is empty, **all** registered proxies are loaded.

### Features

```python
pm = ProxyManager(activate_proxies=["fmp"])
print(pm.available())           # sorted list of loaded proxy keys
print(pm.retrieve_api_docs())   # human-readable docs for all proxies
print(pm.retrieve_api_docs("fmp"))  # docs for one proxy
print(pm.api_catalog())         # structured directory as dict
```

### Dynamic Registration

```python
pm.register("custom/my_api", MyAPIProxy)
```


## Writing a Proxy

A complete example (Wolfram Alpha, simplified):

```python
import os
from lllm.proxies import BaseProxy, ProxyRegistrator

@ProxyRegistrator(
    path="wa",
    name="Wolfram Alpha API",
    description="Query Wolfram Alpha for computations and factual answers.",
)
class WAProxy(BaseProxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("WA_API_DEV")
        self.base_url = "https://www.wolframalpha.com/api/v1"

    @BaseProxy.endpoint(
        category="Query",
        endpoint="llm-api",
        name="Wolfram Alpha LLM API",
        description="Natural language query to Wolfram Alpha.",
        params={
            "input*": (str, "10 densest elemental metals"),
            "assumption": (str, None),
        },
        response={"response": "Input interpretation: ..."},
    )
    def query(self, params: dict) -> dict:
        return params   # BaseProxy handles the actual HTTP call
```

Key points:
- `__init__` receives `cutoff_date`, `activate_proxies`, `deploy_mode` via `**kwargs`
- `@BaseProxy.endpoint` is metadata — it describes the endpoint for docs and auto-testing
- The `path` in `@ProxyRegistrator` is what users pass to `activate_proxies` and dispatch
