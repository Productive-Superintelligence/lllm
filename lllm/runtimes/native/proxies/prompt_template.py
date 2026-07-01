"""
System-prompt blocks injected when an agent has proxy config.

Two built-in templates, selected automatically based on ``exec_env``:

``DEFAULT_PROXY_PROMPT_TEMPLATE``
    Used when ``exec_env == "interpreter"`` (default).  Explains the
    persistent Python interpreter, ``CALL_API``, stdout capture, truncation,
    and ``query_api_doc``.

``DEFAULT_PROXY_PROMPT_NO_PYTHON_TEMPLATE``
    Used for all other modes (``"jupyter"``, ``None``, future sandboxes).
    A minimal block that injects the API directory and reminds the agent to
    call ``query_api_doc`` before using a new endpoint.  The system prompt the
    user writes handles the rest (e.g. the Jupyter ``<python_cell>`` interface).

Both blocks are rendered once at agent build time via ``render_proxy_prompt``
and appended to the agent's system prompt text.  The rendered result is
re-escaped (``{`` → ``{{}}``) so it is safe to embed inside a Prompt template
whose renderer calls ``str.format``.

Template variables filled at render time (none left for the Prompt renderer):
    {api_directory}        — formatted string from ProxyManager.retrieve_api_docs()
    {max_output_chars}     — int, matches AgentInterpreter.max_output_chars
    {truncation_indicator} — str, matches AgentInterpreter.truncation_indicator

Literal braces in code examples use {{ and }} so they survive str.format()
as { and } in the intermediate output, then get re-escaped before appending.
"""
from __future__ import annotations

from typing import Optional


DEFAULT_PROXY_PROMPT_TEMPLATE = """\

---

## API Library Usage

You have access to a set of external APIs callable from `run_python`. \
This tool runs your code in a **persistent Python interpreter** — variables \
assigned in one call are available in the next, so you can build up results \
progressively across multiple tool calls.

### `CALL_API(api_path: str, api_params: dict)`

Call any registered API endpoint. Authentication is handled automatically \
— never construct raw HTTP requests or hardcode credentials.

```python
# Example
response = CALL_API("fmp/stock/quote", {{"symbol": "AAPL"}})
print(response)
```

### How the interpreter works

- **State persists**: variables survive across `run_python` calls in this \
session. Fetch data in one call, process it in the next.
- **Print your results**: output is captured from stdout. Use `print()` to \
surface anything you need. The last expression is **not** auto-returned like \
a REPL — you must print it explicitly.
- **Output truncation**: output longer than {max_output_chars} characters is \
cut off and `"{truncation_indicator}"` is appended. Print the most critical \
information first, or break large results across multiple calls.
- **Full Python available**: the standard library is at your disposal. Use it \
for data wrangling, aggregation, formatting, numerical computation, string \
processing — anything a Python script can do. `CALL_API` + Python together \
give you a highly flexible tool for complex, multi-step data tasks.
- **Error reporting**: exceptions are caught and returned as formatted \
tracebacks, so you can diagnose and fix problems without losing session state.

### Workflow guidelines

1. **Consult docs first** — before using an endpoint for the first time this \
session, call `query_api_doc` with the proxy name to get full parameter specs \
and examples. Skip this if you already retrieved the docs earlier.
2. **Use `CALL_API`** — always use `CALL_API` for API calls; never make raw \
HTTP requests.
3. **Mind truncation** — if a result might be large, select only the fields \
you need before printing, or summarise with Python first.

---

## Available APIs

{api_directory}

---
"""


DEFAULT_PROXY_PROMPT_NO_PYTHON_TEMPLATE = """\

---

## API Library Usage

You have access to a set of external APIs callable via ``CALL_API``. \
Authentication is handled automatically — never hardcode credentials.

```python
response = CALL_API("fmp/stock/quote", {{"symbol": "AAPL"}})
```

Use `query_api_doc` to retrieve full endpoint documentation before using \
any endpoint for the first time this session.

---

## Available APIs

{api_directory}

---
"""


def render_proxy_prompt(
    api_directory: str,
    max_output_chars: int,
    truncation_indicator: str,
    exec_env: Optional[str] = "interpreter",
    custom_template: Optional[str] = None,
) -> str:
    """
    Render the proxy system-prompt block and re-escape it for safe embedding
    inside a :class:`~lllm.core.prompt.Prompt` template.

    The Prompt renderer calls ``str.format(**prompt_args)`` on the full
    template string, so any literal ``{`` / ``}`` characters in the rendered
    block must be doubled.  This function handles that automatically.

    Parameters
    ----------
    api_directory:
        Formatted API directory string from
        :meth:`~lllm.proxies.base.ProxyManager.retrieve_api_docs`.
    max_output_chars:
        Truncation threshold; embedded in the interpreter-mode template.
        Passed through for all modes (unused by the no-python template).
    truncation_indicator:
        Truncation suffix; embedded in the interpreter-mode template.
        Passed through for all modes (unused by the no-python template).
    exec_env:
        Controls which default template is selected when ``custom_template``
        is not provided:

        - ``"interpreter"`` (default) — :data:`DEFAULT_PROXY_PROMPT_TEMPLATE`,
          which explains the persistent interpreter and ``run_python`` tool.
        - anything else (``"jupyter"``, ``None``, future modes) —
          :data:`DEFAULT_PROXY_PROMPT_NO_PYTHON_TEMPLATE`, which injects only
          the API directory and the ``query_api_doc`` reminder.
    custom_template:
        If provided, used instead of the automatic selection above.
        Must use ``{api_directory}``, ``{max_output_chars}``, and
        ``{truncation_indicator}`` placeholders and double any literal braces
        in code examples (``{{`` / ``}}``) as in the default templates.

    Returns
    -------
    str
        Rendered block, safe to concatenate with a Prompt template string.
    """
    if custom_template is not None:
        template = custom_template
    elif exec_env == "interpreter":
        template = DEFAULT_PROXY_PROMPT_TEMPLATE
    else:
        template = DEFAULT_PROXY_PROMPT_NO_PYTHON_TEMPLATE

    rendered = template.format(
        api_directory=api_directory,
        max_output_chars=max_output_chars,
        truncation_indicator=truncation_indicator,
    )
    # Re-escape all remaining braces so the block is inert when the Prompt
    # renderer calls str.format(**prompt_args) on the full system prompt.
    return rendered.replace("{", "{{").replace("}", "}}")
