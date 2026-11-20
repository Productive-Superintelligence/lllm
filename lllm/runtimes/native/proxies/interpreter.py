"""
AgentInterpreter — lightweight, stateful Python interpreter for agent tool use.

Each instance holds a persistent namespace so variables survive across
successive ``run()`` calls within the same agent session.  Execution is
wrapped in a thread with a configurable timeout to prevent runaway code.

This design is intentionally simple and suitable for trusted LLM-generated
code whose primary job is to call HTTP APIs and process their responses.
For stronger isolation, swap the ``exec`` backend with RestrictedPython.

How output is captured: while any call's code runs, ``sys.stdout`` is one
process-wide router (:class:`_ThreadStdout`) that sends each write by thread.
The exec thread of a call writes into that call. Only threads started by a
call's code belong to it; a ``Thread.start`` hook, installed by the first call
and kept for the life of the process, records them. Such a thread writes into
its call while the call runs; nowhere while the call is timed out, until its
code ends; and once the call has returned or its timed-out code has ended,
into the interpreter's newest running call (a reused pool worker), or else to
the program. Every other thread writes to the program's stream, saved when
the first call starts and put back in ``sys.stdout`` when the last call's code
ends.

Only calls that have not timed out own ``sys.stdout``. While one runs, a
redirect, the code's own or the program's, is left alone and captures what
every thread prints; when the last call ends, the program's saved stream
replaces it. A stream set after a timeout, while only timed-out code runs, is
kept, whether the program or the timed-out code set it: until another call
starts, the timed-out code and its threads print into it, and the next call
saves it as the program's.
"""
from __future__ import annotations

import builtins
import io
import sys
import threading
import traceback
from typing import Any, Callable, Dict, List, Optional

_UNSET: Any = object()  # "nothing saved", unlike a saved None stream

_lock = threading.Lock()
_flights: set = set()  # calls whose code is running, timed-out ones included
_original: Any = None  # the program's stream, saved when it gets the router
_earlier: Any = _UNSET  # the program's stream saved before _original, if any
_wrappers: List[Any] = []  # program streams seen calling back into the router
_kept: Any = None  # a stream set after a timeout and kept when that code ended
_kept_original: Any = None  # _original when _kept was kept
_router: Optional["_ThreadStdout"] = None  # the one stand-in, made on first use
_start_raw: Any = None  # threading.Thread.__dict__["start"] before the hook
_capture = threading.local()  # .call: the call whose code runs on this thread
_nesting = threading.local()  # .stack: what router operations on this thread forward to


class _CallOutput:
    """What one ``run()`` call captures, and its state.

    Threads started by the call's code keep a reference to this record. It
    holds the interpreter's list of running calls (``running``), the text
    buffer, which is released once ``run()`` has read it or the call has
    timed out, a few flags, and ``sys.stdout`` as it was at the timeout
    (``stdout_at_timeout``) until the timed-out code ends.
    """

    __slots__ = ("running", "buffer", "returned", "abandoned", "finished", "stdout_at_timeout")

    def __init__(self, running: List["_CallOutput"]) -> None:
        self.running = running  # the interpreter's running calls, newest last
        self.buffer: Optional[io.StringIO] = io.StringIO()
        self.returned = False  # run() has returned its output
        self.abandoned = False  # the call timed out (or run() was interrupted)
        self.finished = False  # its code has ended
        self.stdout_at_timeout: Any = None


class _Null(io.TextIOBase):
    """Where a timed-out call's output goes, until its code ends: nowhere."""

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
    if call.abandoned and not call.finished:
        return call  # timed out, code still running: dropped
    if call.abandoned or call.returned:  # e.g. a reused pool worker
        try:
            return call.running[-1]  # the interpreter's newest running call
        except IndexError:
            return None
    return call


def _known(stream: Any) -> bool:
    """Whether ``stream`` has been seen calling back into the router, i.e.
    it wraps the router (a tee, rich's file proxy, a codecs writer...).

    A known false positive: rich's Console looks up ``sys.stdout`` when it
    prints, and during a call that is the router, so a ``FileProxy`` that
    was in ``sys.stdout`` before any call (rich's Progress or Live) is marked
    once the program prints through it during a call. Afterwards, writes
    through held router references skip it (see :func:`_program_stream`).
    """

    return stream is not None and any(stream is wrapper for wrapper in _wrappers)


def _program_stream(reentry: bool) -> Any:
    """The program's stream for an operation on the router.

    A first operation goes to ``_original``, the program's stream (nowhere
    when the program set ``sys.stdout`` to None), unless ``_original`` is a
    known wrapper of the router and is ``sys.stdout``: then it is that
    wrapper's own inner operation, handled like a re-entry. A re-entry (a
    program stream calling back into the router) goes to the stream saved
    before ``_original``, unless that is itself a known wrapper, and
    otherwise to ``sys.__stdout__``; a saved None goes nowhere.
    """

    if not reentry and not (_known(_original) and getattr(sys, "stdout", None) is _original):
        return _NULL if _original is None else _original
    if _earlier is not _UNSET and _earlier is not _router and not _known(_earlier):
        return _NULL if _earlier is None else _earlier
    return _NULL if sys.__stdout__ is None else sys.__stdout__


def _route(reentry: bool = False) -> Any:
    """Where the current thread's write to ``sys.stdout`` goes."""

    call = _owner()
    if call is None:
        return _program_stream(reentry)
    buffer = call.buffer
    return _NULL if buffer is None else buffer


def _make_start_hook(original: Callable[[threading.Thread], None]) -> Callable:
    """``Thread.start`` with the hook: a thread started by a call's code
    belongs to that call. The hook keeps its own reference to the original,
    so it survives a reload of this module."""

    def start(self: threading.Thread) -> None:
        if _flights:
            call = _owner()
            if call is not None:
                self._lllm_call = call
        original(self)

    start.__name__ = "start"
    start.__qualname__ = "Thread.start"
    start.__doc__ = getattr(original, "__doc__", None)
    start._lllm_start_hook = True  # type: ignore[attr-defined]
    return start


def _install_start_hook() -> None:
    """Install the hook once per process (callers hold ``_lock``)."""

    global _start_raw
    raw = vars(threading.Thread).get("start")
    if getattr(raw, "_lllm_start_hook", False) is True:
        return  # in place, maybe from an earlier load of this module
    if _start_raw is None:
        _start_raw = raw
    if raw is _start_raw:  # the first use, or someone put the original back
        threading.Thread.start = _make_start_hook(threading.Thread.start)


def _stack() -> List[Any]:
    """What the router operations in progress on this thread forward to."""

    try:
        return _nesting.stack
    except AttributeError:
        _nesting.stack = []
        return _nesting.stack


def _entered() -> Any:
    """The target of an operation entering the router now. A program stream
    that is in the middle of a router operation on this thread and calls
    back into the router is marked as a wrapper of it (the evidence), and
    the operation is a re-entry. Inside any other operation (one going
    nowhere, or into a call), it is a first operation."""

    stack = _stack()
    outer = stack[-1] if stack else None  # targets are never None (_NULL is)
    reentry = outer is not None and (outer is _original or outer is _earlier)
    if reentry and not _known(outer):
        _wrappers.append(outer)
    return _route(reentry)


def _within(target: Any, function: Callable[[], Any]) -> Any:
    """Run ``function`` as a router operation forwarding to ``target``."""

    stack = _stack()
    stack.append(target)
    try:
        return function()
    finally:
        stack.pop()


class _ThreadStdout:
    """``sys.stdout`` while calls' code runs: one router per process.

    Each operation (a write, a flush, or a method looked up on it) goes
    where :func:`_route` sends the thread: into the call it belongs to,
    nowhere for a timed-out call's threads until its code ends, or to the
    program's stream. A program stream that calls back into the router while
    serving one of its operations (a tee, rich's file proxy, a codecs
    writer, saved as the program's stream) is a known wrapper of it, and the
    call-back is a re-entry: it goes to the stream saved before ``_original``
    unless that is itself a known wrapper, and otherwise to
    ``sys.__stdout__``; a saved None goes nowhere (see
    :func:`_program_stream`). The code's own redirects replace this router
    in ``sys.stdout`` and are process-wide while they last.
    """

    def write(self, text: str) -> int:
        target = _entered()
        return _within(target, lambda: target.write(text))

    def flush(self) -> None:
        target = _entered()
        _within(target, lambda: target.flush())

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):  # e.g. while copying
            raise AttributeError(name)
        target = _entered()
        value = _within(target, lambda: getattr(target, name))
        if not callable(value):
            return value

        def method(*args: Any, **kwargs: Any) -> Any:
            return _within(target, lambda: value(*args, **kwargs))

        return method


def _owned() -> bool:
    """Whether a call that has not timed out is running: only such calls
    own ``sys.stdout``; once the rest have timed out, it is the program's."""

    return any(not call.abandoned for call in _flights)


def _enter(call: _CallOutput) -> None:
    """A call's code starts: install the router unless another running
    call owns ``sys.stdout`` (whatever is there then is left alone)."""

    global _router, _original, _earlier, _wrappers, _kept, _kept_original
    with _lock:
        if _router is None:
            _router = _ThreadStdout()
        _install_start_hook()
        current = getattr(sys, "stdout", None)
        if not _owned() and current is not _router:
            # The first call, or the program set its own stream while only
            # timed-out code ran: save it. A stream timed-out code set before
            # its timeout is not the program's, so the saved one stays.
            if not any(current is c.stdout_at_timeout for c in _flights):
                if current is _original:
                    pass  # put back: what is saved beneath it stays
                elif _flights:  # set after a timeout, while that code runs
                    _earlier = _original
                elif _kept is not None and current is _kept:  # ... and kept since
                    _earlier = _kept_original
                elif not _known(current):
                    _earlier, _wrappers = _UNSET, []  # a new stream: forget the old
                _original = current
                _kept = _kept_original = None
                _wrappers = [w for w in _wrappers if w is _original or w is _earlier]
            sys.stdout = _router
        _flights.add(call)
        if not call.abandoned:  # the timeout can fire before the code starts
            call.running.append(call)
    _capture.call = call


def _exit(call: _CallOutput) -> None:
    """A call's code ends: put back what it replaced, and the program's
    stream after the last call."""

    global _kept, _kept_original
    _capture.call = None
    with _lock:
        call.finished = True
        _flights.discard(call)
        current = getattr(sys, "stdout", None)
        at_timeout, call.stdout_at_timeout = call.stdout_at_timeout, None
        if call.abandoned and current is not at_timeout and current is not _router:
            # Set after the timeout, when the program had moved on: keep it,
            # and what it may wrap, for the next call (see _enter).
            _kept, _kept_original = current, _original
            return
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
      example under ``bcall``). Only threads started by a call's code belong
      to it: once that call has returned, such a thread (a pool worker kept
      in a variable, say) prints into the interpreter's newest running call,
      or to the program when none runs. Other threads print to the program.
      The first call installs a ``threading.Thread.start`` hook that stays
      for the life of the process.
    - The code's own redirects (``redirect_stdout``, assigning
      ``sys.stdout``, doctest) work, but are process-wide while they last:
      other running calls' output, and the program's, goes into them too.
      Likewise a redirect the program makes while a call runs captures the
      call's output. When the last call's code ends, the program's stream
      saved when calls began is put back in ``sys.stdout``.
    - Long output keeps its first and last ``max_output_chars / 2``
      characters, with the truncation indicator between them, so the end of
      a traceback survives.
    - Execution is run in a daemon thread; if it does not complete within
      ``timeout`` seconds a ``TimeoutError`` is raised.  The namespace is NOT
      rolled back on timeout — partial side-effects may have occurred.  Until
      the code ends, what it and its threads print is dropped; then its
      threads print like a returned call's.  A stream set after the timeout
      while no other call runs, by the program or by the timed-out code, is
      kept: the timed-out code and its threads print into it until another
      call starts, and the next call saves it as the program's.

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
        error result to the agent).  If ``run()`` itself is interrupted (for
        example by Ctrl-C), the call is given up the same way.

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
        # Whatever ends run() early, a timeout or an interrupt such as
        # KeyboardInterrupt (not an Exception), gives the call up.
        given_up = True
        try:
            thread.start()
            thread.join(self.timeout)
            if thread.is_alive():
                raise TimeoutError(
                    f"Code execution timed out after {self.timeout:.0f}s. "
                    "Check for infinite loops or overly long-running operations."
                )
            given_up = False
            if exc_holder:
                exc = exc_holder[0]
                output = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            else:
                output = call.buffer.getvalue()
        finally:
            _stop(call, timed_out=given_up)

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
