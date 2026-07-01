# Proxies, Skills, And Sandbox

The v1 proxy, skills, sandbox, and computer-use surroundings are preserved
inside native. They are useful for native agents but remain optional and do not
reshape the public tactic protocol.

## Proxies

A proxy class describes endpoint metadata with `BaseProxy.endpoint`, and
`ProxyManager` loads registered proxy resources.

```python
from lllm.runtimes.native.proxies import BaseProxy, ProxyRegistrator


@ProxyRegistrator(
    path="demo",
    name="Demo API",
    description="Small local proxy used in examples.",
)
class DemoProxy(BaseProxy):
    @BaseProxy.endpoint(
        category="demo",
        endpoint="lookup",
        name="lookup",
        description="Look up a local value.",
        params={"key": (str, "alpha")},
        response=[{"value": "str"}],
    )
    def lookup(self, params=None):
        params = params or {}
        return {"value": params.get("key", "")}
```

When an agent has proxy config, native can inject `query_api_doc(proxy_name)`,
`run_python(code)`, and a rendered API-directory block into the system prompt.

## AgentInterpreter

`AgentInterpreter` uses `exec()` in a persistent namespace, injects `CALL_API`,
captures stdout, truncates long output, and returns tracebacks as tool output.
Use it for trusted agent workflows and swap the backend when stronger
isolation is required.

```yaml
proxy:
  activate_proxies: [demo]
  deploy_mode: false
  cutoff_date: "2024-01-01"
  exec_env: interpreter
  max_output_chars: 5000
  truncation_indicator: "... (truncated)"
  timeout: 60.0
```

`exec_env: jupyter` keeps the API docs but leaves code execution to a tactic
using the Jupyter sandbox. `exec_env: null` injects API awareness without a
runtime execution tool.

## Skills

Native agent config preserves progressive skill disclosure:

```yaml
skills:
  - pdf
  - commit-review
```

Local skills are discovered from standard project and user directories such as
`.agents/skills/<name>/SKILL.md` and `.claude/skills/<name>/SKILL.md`. A compact
catalog is appended to the system prompt, and an `activate_skill(name)` tool
loads full instructions on demand.

`skills: "*"` exposes all discovered local skills. URL entries are fetched as
remote `SKILL.md` files. Provider-hosted skill ids that start with `skill_` are
passed through as model args for compatible Anthropic models.

## Sandbox And Computer Use

| Area | Module | Notes |
| --- | --- | --- |
| Jupyter sessions | `lllm.runtimes.native.sandbox.jupyter` | Notebook file management, kernel execution, markdown/code cells, and optional proxy-aware init code. |
| In-process interpreter | `lllm.runtimes.native.proxies.interpreter` | Persistent Python namespace for trusted proxy workflows. |
| Computer use | `lllm.runtimes.native.tools.cua` | Browser automation helpers using OpenAI/Azure clients and Playwright when installed. |
| Response API tools | `LiteLLMInvoker` | `Prompt.addon_args` can request web-search or computer-use tools for response API calls. |

Install only the extras you need. Offline native tests do not require live
providers, Jupyter, Playwright, or proxy API credentials.
