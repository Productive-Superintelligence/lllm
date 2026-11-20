"""
AgentInterpreter — lightweight, stateful Python interpreter for agent tool use.

Each instance holds a persistent namespace so variables survive across
successive ``run()`` calls within the same agent session.  Execution is
wrapped in a thread with a configurable timeout to prevent runaway code.

This design is intentionally simple and suitable for trusted LLM-generated
code whose primary job is to call HTTP APIs and process their responses.
For stronger isolation, swap the ``exec`` backend with RestrictedPython.

How output is captured: while any call's code is running, ``sys.stdout`` is
one process-wide router (:class:`_ThreadStdout`) that sends each write by
thread. The exec thread of a call writes into that call. A thread started
while a call ran belongs to it (a ``Thread.start`` hook records the call):
it writes into the call while the call runs, nowhere while the call is timed
out but its code still runs, and, once the call has returned, into the
interpreter's newest running call (a reused pool worker) or else to the
program. Every other thread writes to the program's own stream, which is put
back in ``sys.stdout`` when the last call's code ends. Only calls that have
not timed out own ``sys.stdout``: the code's redirects are left alone while
such a call runs, and a stream set while just timed-out code runs counts as
the program's.
"""
from __future__ import annotations

import builtins
import io
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional

_lock = threading.Lock()
_flights: set = set()  # calls whose code is running, timed-out ones included
_original: Any = None  # the program's sys.stdout, saved when the first call starts
_router: Optional["_ThreadStdout"] = None  # the one stand-in, made on first use
_original_start: Any = None  # Thread.start before the hook, saved once
_capture = threading.local()  # .call: the call whose code runs on this thread


class _CallOutput:
    """What one ``run()`` call captures, and its state.

    Threads started by the call's code keep a reference to this record, so
    it holds only flags and the text buffer, which is released once
    ``run()`` has read it or the call has timed out.
    """

    __slots__ = ("running", "buffer", "returned", "abandoned", "finished", "stdout_at_timeout")

    def __init__(self, running: List["_CallOutput"]) -> None:
        self.running = running  # the interpreter's running calls, newest last
        self.buffer: Optional[io.StringIO] = io.StringIO()
        self.returned = False  # run() has returned its output
        self.abandoned = False  # the call timed out
        self.finished = False  # its code has ended
        self.stdout_at_timeout: Any = None


class _Null(io.TextIOBase):
    """Where a timed-out call's output goes: nowhere."""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        return len(text)


_NULL = _Null()


def _owner() -> Optional[_CallOutput]:
    """The call the current thread writes for, or None for the program."""

    call = getattr(_capture, "call", None)
    if call is not None:  # the exec thread of a call
        return call
    call = getattr(threading.current_thread(), "_lllm_call", None)
    if call is None:
        return None
    if call.abandoned:  # timed out: its threads write nowhere, then to the program
        return None if call.finished else call
    if call.returned:  # a reused worker: the interpreter's newest running call
        try:
            return call.running[-1]
        except IndexError:
            return None
    return call


def _route() -> Any:
    """Where the current thread's writes to ``sys.stdout`` go."""

    call = _owner()
    if call is None:
        return _NULL if _original is None else _original
    buffer = call.buffer
    return _NULL if buffer is None else buffer


def _start_hook(self: threading.Thread) -> None:
    """``Thread.start``, installed once: a thread started while a call runs
    belongs to the call its starter writes for."""

    if _flights:
        call = _owner()
        if call is not None:
            self._lllm_call = call
    _original_start(self)


class _ThreadStdout:
    """``sys.stdout`` while calls' code runs: one router per process.

    Each write goes where :func:`_route` sends the writing thread: into the
    call it belongs to, nowhere for a timed-out call's threads, or to the
    program's own stream. The code's own redirects replace this router in
    ``sys.stdout`` and are process-wide while they last.
    """

    def write(self, text: str) -> int:
        return _route().write(text)

    def flush(self) -> None:
        _route().flush()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):  # e.g. while copying
            raise AttributeError(name)
        return getattr(_route(), name)


def _owned() -> bool:
    """Whether a call that has not timed out is running: only such calls
    own ``sys.stdout``; once the rest have timed out, it is the program's."""

    return any(not call.abandoned for call in _flights)


def _enter(call: _CallOutput) -> None:
    """A call's code starts: install the router unless another running
    call owns ``sys.stdout`` (whatever is there then is left alone)."""

    global _router, _original, _original_start
    with _lock:
        if _router is None:
            _router = _ThreadStdout()
        if _original_start is None:
            _original_start = threading.Thread.start
        if threading.Thread.start is _original_start:  # first use, or put back
            threading.Thread.start = _start_hook
        current = getattr(sys, "stdout", None)
        if not _owned() and current is not _router:
            # The first call, or the program set its own stream while only
            # timed-out code ran: save it. A stream timed-out code set before
            # its timeout is not the program's, so the saved one stays.
            if not any(current is c.stdout_at_timeout for c in _flights):
                _original = current
            sys.stdout = _router
        _flights.add(call)
        call.running.append(call)
    _capture.call = call


def _exit(call: _CallOutput) -> None:
    """A call's code ends: put back what it replaced, and the program's
    stream after the last call."""

    _capture.call = None
    with _lock:
        call.finished = True
        _flights.discard(call)
        current = getattr(sys, "stdout", None)
        at_timeout, call.stdout_at_timeout = call.stdout_at_timeout, None
        if call.abandoned and current is not at_timeout:
            return  # set after the timeout, when the program had moved on: keep it
        if current is not _router and not _owned():
            sys.stdout = current = _router  # the code left it replaced
        if not _flights and current is _router:
            sys.stdout = _original


def _stop(call: _CallOutput, *, timed_out: bool) -> None:
    """``run()`` is done with a call: release its text."""

    with _lock:
        if call in call.running:
            call.running.remove(call)
        call.buffer = None
        if timed_out:
            call.abandoned = True
            if not call.finished:  # its code still runs; see _exit
                call.stdout_at_timeout = getattr(sys, "stdout", None)
        else:
            call.returned = True


class AgentInterpreter:
    """
    A stateful Python interpreter backed by ``exec()`` and a persistent namespace.

    - ``CALL_API`` is pre-injected, ready to use on the first call.
    - Variables assigned in one ``run()`` call are available in subsequent calls.
    - Output is captured from stdout; the caller receives a plain string.
      A call captures what its code prints, including code ``CALL_API`` runs
      and threads the code starts; calls may run at the same time (for
      example under ``bcall``). A thread started in an earlier call, such as
      a pool worker kept in a variable, prints into the interpreter's newest
      running call; outside calls, threads print to the program.
    - The code's own redirects (``redirect_stdout``, assigning
      ``sys.stdout``, doctest) work, but are process-wide while they last:
      other running calls' output, and the program's, goes into them too.
      When the last call's code ends, the program's ``sys.stdout`` is
      restored.
    - Long output keeps its first and last ``max_output_chars / 2``
      characters, with the truncation indicator between them, so the end of
      a traceback survives.
    - Execution is run in a daemon thread; if it does not complete within
      ``timeout`` seconds a ``TimeoutError`` is raised.  The namespace is NOT
      rolled back on timeout — partial side-effects may have occurred.  While
      the code keeps running, what it and its threads print is dropped; once
      it ends, its threads print to the program, and ``sys.stdout`` is
      restored then unless the program set its own stream after the timeout.

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
        # This interpreter's running calls, newest last (see _owner).
        self._running: List[_CallOutput] = []

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
        call = _CallOutput(self._running)
        exc_holder: list = []

        def _run() -> None:
            try:
                _enter(call)
                try:
                    exec(code, self.namespace)  # noqa: S102
                finally:
                    _exit(call)
            except Exception as exc:  # noqa: BLE001
                exc_holder.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(self.timeout)

        if thread.is_alive():
            _stop(call, timed_out=True)
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
            output = call.buffer.getvalue()
        _stop(call, timed_out=False)

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
