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

import builtins
import contextlib
import io
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional

# The output buffer of the ``run()`` call executing on this thread, if any.
_capture = threading.local()
_stdout_lock = threading.Lock()
_active_captures = 0


class _ThreadStdout:
    """Stands in for ``sys.stdout`` while ``run()`` calls are in flight.

    A thread executing an agent's code writes to its own call's buffer;
    every other thread writes to the stream this object replaced, so the
    program's own output is neither captured nor lost.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def _target(self) -> Any:
        buffer = getattr(_capture, "buffer", None)
        return self._stream if buffer is None else buffer

    def write(self, text: str) -> int:
        target = self._target()
        if target is None:  # sys.stdout was None (pythonw): drop the text
            return len(text)
        return target.write(text)

    def flush(self) -> None:
        target = self._target()
        if target is not None:
            target.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)


@contextlib.contextmanager
def _capture_stdout(buffer: io.StringIO):
    """Send what this thread writes to ``sys.stdout`` into *buffer*."""

    global _active_captures
    with _stdout_lock:
        if not isinstance(sys.stdout, _ThreadStdout):
            sys.stdout = _ThreadStdout(sys.stdout)
        _active_captures += 1
    previous = getattr(_capture, "buffer", None)
    _capture.buffer = buffer
    try:
        yield
    finally:
        _capture.buffer = previous
        with _stdout_lock:
            _active_captures -= 1
            if not _active_captures and isinstance(sys.stdout, _ThreadStdout):
                sys.stdout = sys.stdout._stream


class AgentInterpreter:
    """
    A stateful Python interpreter backed by ``exec()`` and a persistent namespace.

    - ``CALL_API`` is pre-injected, ready to use on the first call.
    - Variables assigned in one ``run()`` call are available in subsequent calls.
    - Output is captured from stdout; the caller receives a plain string.
      Each call captures only its own output, so calls may run at the same
      time (for example under ``bcall``) and the program's own prints are
      never captured.
    - Long output keeps its first and last ``max_output_chars / 2``
      characters, with the truncation indicator between them, so the end of
      a traceback survives.
    - Execution is run in a daemon thread; if it does not complete within
      ``timeout`` seconds a ``TimeoutError`` is raised.  The namespace is NOT
      rolled back on timeout — partial side-effects may have occurred.

    Parameters
    ----------
    proxy_manager:
        A :class:`~lllm.proxies.base.ProxyManager` instance.  Its ``__call__``
        is injected as ``CALL_API`` into the execution namespace.
    max_output_chars:
        Captured output longer than this keeps this many characters, half
        from the start and half from the end.  0 means no truncation.
    truncation_indicator:
        String put where output was cut.
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
        # Output buffers of this interpreter's run() calls in flight.
        self._outputs: List[io.StringIO] = []

        # Persistent namespace — survives across run() calls within a session.
        self.namespace: Dict[str, Any] = {
            "__builtins__": self._builtins(),
            "CALL_API": proxy_manager.__call__,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, code: str) -> str:
        """
        Execute *code* in the persistent namespace and return captured output.

        Output is everything the code prints or writes to stdout during
        execution.  Exceptions are caught, formatted as tracebacks, and
        returned as the output string so the agent can read and fix them.

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
            self._outputs.append(captured)
            try:
                with _capture_stdout(captured):
                    exec(code, self.namespace)  # noqa: S102
            except Exception as exc:  # noqa: BLE001
                exc_holder.append(exc)
            finally:
                self._outputs.remove(captured)

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
        self.namespace["__builtins__"] = self._builtins()
        if call_api is not None:
            self.namespace["CALL_API"] = call_api

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _builtins(self) -> Dict[str, Any]:
        # The agent's code gets a print() that writes to its own call's output.
        return {**vars(builtins), "print": self._print}

    def _print(
        self,
        *args: Any,
        sep: Optional[str] = " ",
        end: Optional[str] = "\n",
        file: Any = None,
        flush: bool = False,
    ) -> None:
        """print() for the agent's code: writes to its run() call's output."""
        if file is None:
            file = self._output()
        print(*args, sep=sep, end=end, file=file, flush=flush)

    def _output(self) -> Any:
        """The run() call on this thread, else this interpreter's latest call
        (for threads the agent's code starts), else ``sys.stdout``."""
        buffer = getattr(_capture, "buffer", None)
        if buffer is not None:
            return buffer
        try:
            return self._outputs[-1]
        except IndexError:
            return sys.stdout

    def _maybe_truncate(self, text: str) -> str:
        limit = self.max_output_chars
        if not limit or len(text) <= limit:
            return text
        head = limit // 2
        tail = limit - head
        return text[:head] + self.truncation_indicator + text[len(text) - tail:]
