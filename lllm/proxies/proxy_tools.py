"""
Factory functions that build the two proxy tool Functions injected into an
agent's system prompt when proxy config is present.

    ``make_query_api_doc_tool``  — retrieve full endpoint docs for one proxy
    ``make_run_python_tool``     — execute Python in the persistent interpreter

These are auto-injected at agent build time; users can override them by
modifying ``function_list`` on the prompt directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from lllm.core.prompt import Function

if TYPE_CHECKING:
    from lllm.proxies.base import ProxyManager
    from lllm.proxies.interpreter import AgentInterpreter


def make_query_api_doc_tool(proxy_manager: "ProxyManager") -> Function:
    """
    Return a :class:`~lllm.core.prompt.Function` that retrieves full endpoint
    documentation for a named proxy.

    The agent should call this before using any endpoint it hasn't seen this
    session to get accurate parameter specs and examples.
    """

    def query_api_doc(proxy_name: str) -> str:
        """Retrieve detailed API documentation for a proxy by name."""
        try:
            return proxy_manager.retrieve_api_docs(proxy_name)
        except KeyError:
            available = proxy_manager.available()
            return (
                f"Proxy '{proxy_name}' not found. "
                f"Available proxies: {available}"
            )

    return Function.from_callable(
        query_api_doc,
        description=(
            "Retrieve full API documentation for a proxy, including all "
            "endpoints, parameter names, types, and examples. "
            "Call this before using any endpoint you haven't used in this "
            "session to avoid parameter mistakes. "
            "Available proxy names are listed in the API Directory."
        ),
        prop_desc={
            "proxy_name": (
                "Identifier of the proxy to look up "
                "(e.g. 'fmp', 'fred', 'exa'). "
                "Find available names in the API Directory."
            )
        },
    )


def make_run_python_tool(interpreter: "AgentInterpreter") -> Function:
    """
    Return a :class:`~lllm.core.prompt.Function` that executes Python code
    inside the agent's persistent :class:`~lllm.proxies.interpreter.AgentInterpreter`.
    """

    def run_python(code: str) -> str:
        """Execute Python code in a stateful interpreter with CALL_API available."""
        return interpreter.run(code)

    return Function.from_callable(
        run_python,
        description=(
            "Execute Python code in a persistent interpreter. "
            "CALL_API is pre-injected and ready to use. "
            "Variables persist across calls within the same session — "
            "you can fetch data in one call and process it in the next. "
            "Always use print() to surface results: output is captured from "
            "stdout only. "
            "Long outputs are truncated; print the most important data first. "
            "Exceptions are returned as tracebacks so you can diagnose and fix them."
        ),
        prop_desc={
            "code": (
                "Python source code to execute. "
                "Use print() to produce output visible to the model. "
                "Call CALL_API(endpoint_path, params_dict) to invoke APIs."
            )
        },
    )
