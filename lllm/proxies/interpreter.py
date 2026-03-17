"""
AgentInterpreter — lightweight, stateful Python interpreter for agent tool use.

Each instance holds a persistent namespace so variables survive across
successive ``run()`` calls within the same agent session.  Execution is
wrapped in a thread with a configurable timeout to prevent runaway code.

This design is intentionally simple and suitable for trusted LLM-generated
code whose primary job is to call HTTP APIs and process their responses.
For stronger isolation, swap the ``exec`` backend with RestrictedPython.
"""
from __future__ import annotations

import io
import contextlib
import threading
import traceback
from typing import Any, Dict


class AgentInterpreter:
    """
    A stateful Python interpreter backed by ``exec()`` and a persistent namespace.

    - ``CALL_API`` is pre-injected, ready to use on the first call.
    - Variables assigned in one ``run()`` call are available in subsequent calls.
    - Output is captured from stdout; the caller receives a plain string.
    - Long output is truncated to ``max_output_chars`` with an indicator appended.
    - Execution is run in a daemon thread; if it does not complete within
      ``timeout`` seconds a ``TimeoutError`` is raised.  The namespace is NOT
      rolled back on timeout — partial side-effects may have occurred.

    Parameters
    ----------
    proxy_manager:
        A :class:`~lllm.proxies.base.ProxyManager` instance.  Its ``__call__``
        is injected as ``CALL_API`` into the execution namespace.
    max_output_chars:
        Captured stdout is truncated to this many characters.
        0 means no truncation.
    truncation_indicator:
        String appended when output is truncated.
    timeout:
        Maximum wall-clock seconds allowed per ``run()`` call.
    """

    def __init__(
        self,
        proxy_manager,
        *,
        max_output_chars: int = 5000,
        truncation_indicator: str = "... (truncated)",
        timeout: float = 60.0,
    ) -> None:
        self.max_output_chars = max_output_chars
        self.truncation_indicator = truncation_indicator
        self.timeout = timeout

        # Persistent namespace — survives across run() calls within a session.
        self.namespace: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "CALL_API": proxy_manager.__call__,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, code: str) -> str:
        """
        Execute *code* in the persistent namespace and return captured output.

        Output is everything written to stdout during execution.  Exceptions
        are caught, formatted as tracebacks, and returned as the output string
        so the agent can read and fix them.

        If execution exceeds ``self.timeout`` seconds, a ``TimeoutError`` is
        raised (not caught here — the tool framework will surface it as an
        error result to the agent).

        Parameters
        ----------
        code:
            Python source to execute.

        Returns
        -------
        str
            Captured stdout, or a formatted traceback on exception, possibly
            truncated to ``max_output_chars``.
        """
        captured = io.StringIO()
        exc_holder: list = []

        def _run() -> None:
            try:
                with contextlib.redirect_stdout(captured):
                    exec(code, self.namespace)  # noqa: S102
            except Exception as exc:  # noqa: BLE001
                exc_holder.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(self.timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Code execution timed out after {self.timeout:.0f}s. "
                "Check for infinite loops or overly long-running operations."
            )

        if exc_holder:
            exc = exc_holder[0]
            output = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        else:
            output = captured.getvalue()

        return self._maybe_truncate(output)

    def reset(self) -> None:
        """
        Clear all user-defined variables from the namespace.

        ``CALL_API`` and ``__builtins__`` are re-injected automatically.
        Useful between independent agent sessions sharing the same interpreter.
        """
        call_api = self.namespace.get("CALL_API")
        self.namespace.clear()
        self.namespace["__builtins__"] = __builtins__
        if call_api is not None:
            self.namespace["CALL_API"] = call_api

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_truncate(self, text: str) -> str:
        if self.max_output_chars and len(text) > self.max_output_chars:
            return text[: self.max_output_chars] + self.truncation_indicator
        return text
