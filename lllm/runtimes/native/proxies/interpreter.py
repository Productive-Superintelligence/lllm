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
from typing import Any, Dict, Optional

# The output of the ``run()`` call executing on this thread, if any. A thread
# that a call's code starts carries that call's output as ``_lllm_output``.
_capture = threading.local()
_lock = threading.Lock()
_active_captures = 0
_thread_start = threading.Thread.start


class _CallOutput(io.StringIO):
    """One ``run()`` call's output.

    Once the call has timed out, what is written to it is dropped, so the
    abandoned code and its threads print nowhere.
    """

    returned = False  # the call returned its output
    abandoned = False  # the call timed out

    def write(self, text: str) -> int:
        if self.abandoned:
            return len(text)
        return super().write(text)


def _current_output() -> Optional[_CallOutput]:
    """The output of the call this thread belongs to: the call it runs, or
    the call whose code started it (or started the thread that started it).
    A started thread writes to the program's stdout again once its call has
    returned."""
    output = getattr(_capture, "output", None)
    if output is None:
        output = getattr(threading.current_thread(), "_lllm_output", None)
        if output is not None and output.returned:
            return None
    return output


def _start_in_call(self: threading.Thread) -> None:
    """``Thread.start`` while calls are in flight: a thread started by a
    call's code belongs to that call."""
    output = _current_output()
    if output is not None:
        self._lllm_output = output
    _thread_start(self)


class _ThreadStdout:
    """Stands in for ``sys.stdout`` while ``run()`` calls are in flight.

    A thread that belongs to a call writes to the call's output; every other
    thread writes to the stream this object replaced, so the program's own
    output is neither captured nor lost.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def _target(self) -> Any:
        output = _current_output()
        return self._stream if output is None else output

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
        if name.startswith("__") or name == "_stream":  # e.g. while copying
            raise AttributeError(name)
        return getattr(self._target(), name)


@contextlib.contextmanager
def _capture_stdout(output: _CallOutput):
    """Send what this thread, and the threads it starts, write to
    ``sys.stdout`` into *output*. Yields the ``sys.stdout`` stand-in."""

    global _active_captures, _thread_start
    with _lock:
        if not isinstance(getattr(sys, "stdout", None), _ThreadStdout):
            sys.stdout = _ThreadStdout(getattr(sys, "stdout", None))
        stand_in = sys.stdout
        if not _active_captures and threading.Thread.start is not _start_in_call:
            _thread_start = threading.Thread.start
            threading.Thread.start = _start_in_call
        _active_captures += 1
    previous = getattr(_capture, "output", None)
    _capture.output = output
    try:
        yield stand_in
    finally:
        _capture.output = previous
        with _lock:
            _active_captures -= 1
            if not _active_captures:
                while isinstance(getattr(sys, "stdout", None), _ThreadStdout):
                    sys.stdout = sys.stdout._stream
                if threading.Thread.start is _start_in_call:
                    threading.Thread.start = _thread_start


class AgentInterpreter:
    """
    A stateful Python interpreter backed by ``exec()`` and a persistent namespace.

    - ``CALL_API`` is pre-injected, ready to use on the first call.
    - Variables assigned in one ``run()`` call are available in subsequent calls.
    - Output is captured from stdout; the caller receives a plain string.
      A call captures what its code writes to stdout, also from threads it
      starts and from code ``CALL_API`` runs (endpoint methods, ``_call_api``
      overrides, tactic endpoints), and never output from the program's
      other threads, so calls may run at the same time (for example under
      ``bcall``). A thread the code started writes to the program's stdout
      again once its call has returned.
    - The code's own redirections of ``sys.stdout`` (``redirect_stdout``,
      doctest) are honored, and a ``sys.stdout`` it leaves replaced is put
      back when the call ends.
    - Long output keeps its first and last ``max_output_chars / 2``
      characters, with the truncation indicator between them, so the end of
      a traceback survives.
    - Execution is run in a daemon thread; if it does not complete within
      ``timeout`` seconds a ``TimeoutError`` is raised.  The namespace is NOT
      rolled back on timeout — partial side-effects may have occurred — and
      what the code and its threads print from then on is dropped.

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

        # Persistent namespace — survives across run() calls within a session.
        self.namespace: Dict[str, Any] = {
            "__builtins__": vars(builtins),
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
        captured = _CallOutput()
        exc_holder: list = []

        def _run() -> None:
            try:
                with _capture_stdout(captured) as stand_in:
                    try:
                        exec(code, self.namespace)  # noqa: S102
                    finally:
                        # Like redirect_stdout, put back a sys.stdout the code
                        # replaced; after a timeout the program has moved on.
                        current = getattr(sys, "stdout", None)
                        replaced = not isinstance(current, _ThreadStdout)
                        if replaced and not captured.abandoned:
                            sys.stdout = stand_in
            except Exception as exc:  # noqa: BLE001
                exc_holder.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(self.timeout)

        if thread.is_alive():
            captured.abandoned = True
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
        captured.returned = True

        return self._maybe_truncate(output)

    def reset(self) -> None:
        """
        Clear all user-defined variables from the namespace.

        ``CALL_API`` and ``__builtins__`` are re-injected automatically.
        Useful between independent agent sessions sharing the same interpreter.
        """
        call_api = self.namespace.get("CALL_API")
        self.namespace.clear()
        self.namespace["__builtins__"] = vars(builtins)
        if call_api is not None:
            self.namespace["CALL_API"] = call_api

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_truncate(self, text: str) -> str:
        limit = self.max_output_chars
        if not limit or len(text) <= limit:
            return text
        head = limit // 2
        tail = limit - head
        return text[:head] + self.truncation_indicator + text[len(text) - tail:]
