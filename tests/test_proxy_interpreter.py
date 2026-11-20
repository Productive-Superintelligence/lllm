"""run_python: each call captures its own output, and long output keeps both ends."""

import codecs
import contextlib
import copy
import io
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from lllm.runtimes.native import Runtime
from lllm.runtimes.native.proxies import AgentInterpreter, ProxyManager
from lllm.runtimes.native.proxies import interpreter as interpreter_module

ROOT = Path(__file__).resolve().parents[1]


def interpreter(**kwargs):
    return AgentInterpreter(ProxyManager(runtime=Runtime()), **kwargs)


def run_in_thread(interp, code, results, key):
    thread = threading.Thread(target=lambda: results.__setitem__(key, interp.run(code)))
    thread.start()
    return thread


def wait_for_calls_to_end(timeout=10):
    """Wait until no call's code is running, timed-out code included."""

    deadline = time.monotonic() + timeout
    while True:
        with interpreter_module._lock:
            if not interpreter_module._flights:
                return
        assert time.monotonic() < deadline, "a call's code is still running"
        time.sleep(0.01)


class Stuck:
    """A call that timed out; its code runs on until ``finish()``."""

    def __init__(self, interp, release):
        self.interp = interp
        self._release = release

    def finish(self):
        self._release.set()
        wait_for_calls_to_end()


@contextlib.contextmanager
def stuck(code, interp=None):
    """Run ``code`` until it times out and keep it running for the block.

    The code sets ``ready`` once it is where it must be before the timeout,
    and waits on ``release``. The block's end lets the code finish, waits
    for it, and puts back the test's ``sys.stdout``.
    """

    stdout = sys.stdout
    ready, release = threading.Event(), threading.Event()
    interp = interp or interpreter(timeout=0.5)
    interp.namespace.update(ready=ready, release=release)
    call = Stuck(interp, release)
    try:
        with pytest.raises(TimeoutError):
            interp.run(code)
        assert ready.is_set(), "the code must set `ready` before its timeout"
        yield call
    finally:
        call.finish()
        sys.stdout = stdout


BLOCK = "ready.set()\nrelease.wait(10)\n"


# -- One call at a time, and calls at the same time --------------------------


def test_concurrent_runs_capture_only_their_own_output():
    stdout = sys.stdout
    barrier = threading.Barrier(2, timeout=5)
    second_may_finish = threading.Event()
    first, second = interpreter(), interpreter()
    for interp in (first, second):
        interp.namespace.update(barrier=barrier, second_may_finish=second_may_finish)
    loop = "for i in range(3):\n    barrier.wait()\n    print({tag!r}, i)\n"
    results = {}

    # The barrier interleaves the two programs' prints; the first call
    # returns before the second, so a swapped sys.stdout would be left behind.
    one = run_in_thread(first, loop.format(tag="first"), results, "first")
    two = run_in_thread(
        second, loop.format(tag="second") + "second_may_finish.wait(5)\n", results, "second"
    )
    one.join(10)
    second_may_finish.set()
    two.join(10)

    assert results == {
        "first": "first 0\nfirst 1\nfirst 2\n",
        "second": "second 0\nsecond 1\nsecond 2\n",
    }
    assert sys.stdout is stdout


def test_the_programs_own_prints_are_not_captured_while_a_run_is_in_flight(capsys):
    started, finish = threading.Event(), threading.Event()
    interp = interpreter()
    interp.namespace.update(started=started, finish=finish)
    results = {}

    thread = run_in_thread(
        interp, "print('agent')\nstarted.set()\nfinish.wait(5)\nprint('done')\n", results, "out"
    )
    assert started.wait(5)
    print("program")  # the main thread, while the agent's code is running
    finish.set()
    thread.join(10)

    assert results["out"] == "agent\ndone\n"
    assert capsys.readouterr().out == "program\n"


def test_output_written_to_sys_stdout_or_from_worker_threads_is_captured():
    interp = interpreter()

    code = (
        "import sys, pprint\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "sys.stdout.write('raw\\n')\n"
        "pprint.pprint({'a': 1})\n"
        "with ThreadPoolExecutor(2) as pool:\n"
        "    list(pool.map(print, ['w1', 'w2']))\n"
        "    list(pool.map(lambda w: sys.stdout.write(w + '\\n'), ['w3', 'w4']))\n"
    )
    lines = interp.run(code).splitlines()

    assert lines[:2] == ["raw", "{'a': 1}"]
    assert sorted(lines[2:]) == ["w1", "w2", "w3", "w4"]


def test_threads_the_code_starts_write_into_its_call():
    code = (
        "import pprint, sys, threading\n"
        "def work():\n"
        "    sys.stdout.write('raw\\n')\n"
        "    pprint.pprint({'a': 1})\n"
        "    print('printed')\n"
        "    inner = threading.Thread(target=print, args=('grandchild',))\n"
        "    inner.start()\n"
        "    inner.join()\n"
        "worker = threading.Thread(target=work)\n"
        "worker.start()\n"
        "worker.join()\n"
    )

    assert interpreter().run(code) == "raw\n{'a': 1}\nprinted\ngrandchild\n"


def test_a_thread_left_running_writes_to_the_program_after_its_call(capsys):
    proceed, done = threading.Event(), threading.Event()
    interp = interpreter()
    interp.namespace.update(proceed=proceed, done=done)
    code = (
        "import threading\n"
        "def later():\n"
        "    proceed.wait(5)\n"
        "    print('after the call')\n"
        "    done.set()\n"
        "threading.Thread(target=later).start()\n"
    )

    assert interp.run(code) == ""
    started, finish = threading.Event(), threading.Event()
    other = interpreter()
    other.namespace.update(started=started, finish=finish)
    results = {}
    thread = run_in_thread(other, "started.set()\nfinish.wait(5)\n", results, "out")
    assert started.wait(5)
    proceed.set()  # the thread prints while another interpreter's call runs
    assert done.wait(5)
    finish.set()
    thread.join(10)

    assert results["out"] == ""
    assert capsys.readouterr().out == "after the call\n"


def test_a_pool_kept_from_an_earlier_call_prints_into_the_running_call():
    interp = interpreter()
    interp.run(
        "from concurrent.futures import ThreadPoolExecutor\n"
        "pool = ThreadPoolExecutor(max_workers=1)\n"
        "pool.submit(int).result()\n"  # the worker starts during this call
    )
    code = (
        "import sys\n"
        "pool.submit(print, 'printed').result()\n"
        "pool.submit(sys.stdout.write, 'written\\n').result()\n"
        "pool.shutdown()\n"
    )

    assert interp.run(code) == "printed\nwritten\n"


def test_threads_kept_after_a_call_do_not_keep_its_output():
    interp = interpreter(timeout=0.5)
    code = (
        "import threading\n"
        "worker = threading.Thread(target=print, args=('x' * 100_000,))\n"
        "worker.start()\n"
        "worker.join()\n"
    )
    assert len(interp.run(code)) == 5000 + len("... (truncated)")
    record = interp.namespace["worker"]._lllm_call
    assert record.returned and record.buffer is None

    timed_out = "worker = threading.Thread(target=release.wait)\nworker.start()\n" + BLOCK
    with stuck(timed_out, interp):
        record = interp.namespace["worker"]._lllm_call
        assert record.abandoned and record.buffer is None


# -- The code's own redirects -------------------------------------------------


def test_the_codes_own_stdout_redirects_are_honored():
    code = (
        "import contextlib, io, sys\n"
        "with contextlib.redirect_stdout(io.StringIO()) as inner:\n"
        "    print('redirected')\n"
        "saved, sys.stdout = sys.stdout, io.StringIO()\n"
        "print('assigned')\n"
        "assigned, sys.stdout = sys.stdout.getvalue(), saved\n"
        "print(repr(inner.getvalue()), repr(assigned))\n"
    )

    assert interpreter().run(code) == "'redirected\\n' 'assigned\\n'\n"


DOCTESTED = '''
import doctest

def greet(name):
    """
    >>> greet("Ada")
    Hello, Ada!
    """
    print(f"Hello, {name}!")

runner = doctest.DocTestRunner(verbose=False)
for test in doctest.DocTestFinder().find(greet, "greet", globs={"greet": greet}):
    runner.run(test)
print(runner.summarize(verbose=False))
'''


def test_doctest_runs_on_functions_the_code_defines():
    assert interpreter().run(DOCTESTED) == "TestResults(failed=0, attempted=1)\n"


def test_a_run_that_leaves_sys_stdout_replaced_puts_it_back():
    stdout = sys.stdout
    interp = interpreter()

    assert interp.run("import io, sys\nsys.stdout = io.StringIO()\nprint('lost')\n") == ""
    assert sys.stdout is stdout
    assert interp.run("import sys\ndel sys.stdout\n") == ""
    assert sys.stdout is stdout

    assert interp.run("print('next')") == "next\n"


def test_another_calls_end_does_not_undo_a_running_calls_redirect():
    redirected, other_done = threading.Event(), threading.Event()
    first, second = interpreter(), interpreter()
    first.namespace.update(redirected=redirected, other_done=other_done)
    code = (
        "import contextlib, io\n"
        "with contextlib.redirect_stdout(io.StringIO()) as inner:\n"
        "    redirected.set()\n"
        "    other_done.wait(5)\n"
        "    print('inside my redirect')\n"
        "print(repr(inner.getvalue()))\n"
    )
    results = {}

    one = run_in_thread(first, code, results, "first")
    assert redirected.wait(5)
    assert second.run("pass") == ""  # starts and ends inside first's redirect
    other_done.set()
    one.join(10)

    assert results["first"] == "'inside my redirect\\n'\n"


REDIRECTS = {
    "assigned": "import io, sys\nsys.stdout = io.StringIO()\nredirected.set()\nfirst_may_end.wait(5)\n",
    "with": (
        "import contextlib, io\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    redirected.set()\n"
        "    first_may_end.wait(5)\n"
    ),
}


@pytest.mark.parametrize("redirect", sorted(REDIRECTS))
@pytest.mark.parametrize("second_first", [True, False])
@pytest.mark.parametrize("redirecting_call_ends_first", [True, False])
def test_overlapping_calls_where_one_redirects_leave_the_program_its_stdout(
    redirect, second_first, redirecting_call_ends_first
):
    stdout = sys.stdout
    events = {name: threading.Event() for name in ("second_started", "redirected", "first_may_end", "second_may_end")}
    first, second = interpreter(), interpreter()
    for interp in (first, second):
        interp.namespace.update(events)
    results = {}

    def start_first():
        thread = run_in_thread(first, REDIRECTS[redirect], results, "first")
        assert events["redirected"].wait(5)
        return thread

    def start_second():
        thread = run_in_thread(second, "second_started.set()\nsecond_may_end.wait(5)\n", results, "second")
        assert events["second_started"].wait(5)
        return thread

    if second_first:  # the second call is running before the first redirects
        two = start_second()
        one = start_first()
    else:
        one = start_first()
        two = start_second()
    endings = [(events["first_may_end"], one), (events["second_may_end"], two)]
    if not redirecting_call_ends_first:
        endings.reverse()
    for may_end, thread in endings:
        may_end.set()
        thread.join(10)

    assert set(results) == {"first", "second"}
    assert sys.stdout is stdout


# -- Timeouts -----------------------------------------------------------------


def test_a_timed_out_run_does_not_swallow_later_prints(capsys):
    with stuck(BLOCK + "print('late')\n") as call:
        print("after the timeout")  # the code is still running
        assert call.interp.run("print('next')") == "next\n"
        call.finish()

    assert capsys.readouterr().out == "after the timeout\n"


def test_a_timed_out_calls_threads_do_not_print_into_the_next_call(capsys):
    next_running, late_done = threading.Event(), threading.Event()
    interp = interpreter(timeout=0.5)
    interp.namespace.update(next_running=next_running, late_done=late_done)
    code = (
        "import sys, threading\n"
        "def late():\n"
        "    next_running.wait(5)\n"
        "    print('late print')\n"
        "    sys.stdout.write('late write\\n')\n"
        "    late_done.set()\n"
        "threading.Thread(target=late).start()\n" + BLOCK
    )

    with stuck(code, interp):
        # The timed-out call's thread prints while the next call runs.
        retry = "next_running.set()\nassert late_done.wait(5)\nprint('next')\n"
        assert interp.run(retry) == "next\n"

    assert "late" not in capsys.readouterr().out


def test_a_timed_out_calls_threads_print_to_the_program_once_its_code_ends(capsys):
    may_print, printed = threading.Event(), threading.Event()
    interp = interpreter(timeout=0.5)
    interp.namespace.update(may_print=may_print, printed=printed)
    code = (
        "import threading\n"
        "def late():\n"
        "    may_print.wait(5)\n"
        "    print('from the late thread')\n"
        "    printed.set()\n"
        "threading.Thread(target=late).start()\n" + BLOCK
    )

    with stuck(code, interp) as call:
        call.finish()
        started, finish = threading.Event(), threading.Event()
        other = interpreter()
        other.namespace.update(started=started, finish=finish)
        results = {}
        thread = run_in_thread(other, "started.set()\nfinish.wait(5)\n", results, "out")
        assert started.wait(5)
        may_print.set()  # while another interpreter's call runs
        assert printed.wait(5)
        finish.set()
        thread.join(10)

    assert results["out"] == ""
    assert capsys.readouterr().out == "from the late thread\n"


def test_a_pool_from_a_timed_out_call_prints_into_the_next_call():
    interp = interpreter(timeout=0.5)
    code = (
        "from concurrent.futures import ThreadPoolExecutor\n"
        "pool = ThreadPoolExecutor(max_workers=1)\n"
        "pool.submit(int).result()\n" + BLOCK  # the worker starts in this call
    )
    retry = (
        "import sys\n"
        "pool.submit(print, 'printed').result()\n"
        "pool.submit(sys.stdout.write, 'written\\n').result()\n"
        "pool.shutdown()\n"
    )

    with stuck(code, interp) as call:
        call.finish()  # the agent retries on the same pool
        assert interp.run(retry) == "printed\nwritten\n"


def test_a_timed_out_call_that_replaced_sys_stdout_puts_it_back_when_its_code_ends(capsys):
    stdout = sys.stdout

    with stuck("import io, sys\nsys.stdout = io.StringIO()\n" + BLOCK) as call:
        call.finish()
        assert sys.stdout is stdout
        print("visible")

    assert capsys.readouterr().out == "visible\n"


def test_a_redirect_block_closing_after_its_timeout_gives_the_program_its_stdout():
    stdout = sys.stdout
    code = (
        "import contextlib, io\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    ready.set()\n"
        "    release.wait(10)\n"
    )

    with stuck(code) as call:
        call.finish()  # the block puts the router back after the timeout
        assert sys.stdout is stdout


def test_a_retry_while_timed_out_code_holds_its_own_stdout():
    stdout = sys.stdout
    program = io.StringIO()
    sys.stdout = program
    try:
        with stuck("import io, sys\nsys.stdout = io.StringIO()\n" + BLOCK) as call:
            # The retry puts the router in place of the timed-out code's
            # stream, which is not saved as the program's.
            assert interpreter().run("print('retry')") == "retry\n"
            call.finish()
            assert sys.stdout is program
            print("later")
        assert program.getvalue() == "later\n"
    finally:
        sys.stdout = stdout


def test_a_stream_the_program_sets_after_a_timeout_is_kept():
    with stuck(BLOCK) as call:
        program_stream = io.StringIO()
        sys.stdout = program_stream  # the program has moved on and sets its own
        call.finish()
        assert sys.stdout is program_stream


def test_calls_capture_while_timed_out_code_runs_after_the_program_sets_its_stdout():
    with stuck(BLOCK) as call:
        program_stream = io.StringIO()
        sys.stdout = program_stream  # e.g. a test runner between two tests
        assert interpreter().run("print('captured')") == "captured\n"
        print("from the program")
        assert program_stream.getvalue() == "from the program\n"
        call.finish()
        assert sys.stdout is program_stream


class ForwardingTee:
    """A tee whose isatty and fileno ask its first stream."""

    def __init__(self, stream, copy):
        self.stream, self.copy = stream, copy

    def write(self, text):
        self.stream.write(text)
        self.copy.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.copy.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()


def make_tee(stream):
    copy = io.StringIO()
    return ForwardingTee(stream, copy), copy, str


def make_rich(stream):  # what rich's Progress and Live install
    pytest.importorskip("rich")
    from rich.console import Console
    from rich.file_proxy import FileProxy

    return FileProxy(Console(), stream), None, str


def make_codecs(stream):  # a text-to-text codecs writer
    return codecs.getwriter("rot_13")(stream), None, lambda text: codecs.encode(text, "rot_13")


WRAPPERS = {"tee": make_tee, "rich": make_rich, "codecs": make_codecs}


def operate(operation, label, program):
    """Do ``operation`` on sys.stdout from the program; check what it reaches."""

    if operation == "print":
        print(label)
    elif operation == "flush":
        print(label, end="")
        sys.stdout.flush()
    elif operation == "isatty":
        assert sys.stdout.isatty() is False
    else:
        assert sys.stdout.fileno() == program.fileno()  # P's, not sys.__stdout__'s


@pytest.mark.parametrize("operation", ["print", "flush", "isatty", "fileno"])
@pytest.mark.parametrize("order", ["code running", "code ended"])
@pytest.mark.parametrize("wrapper", sorted(WRAPPERS))
def test_a_wrapper_of_sys_stdout_made_while_timed_out_code_runs(tmp_path, wrapper, order, operation):
    stdout = sys.stdout
    labels = ("during the call", "after the calls")
    started, finish = threading.Event(), threading.Event()
    results = {}
    with open(tmp_path / "program.txt", "w+", encoding="utf-8") as program:
        sys.stdout = program  # P, the program's real stream
        try:
            with stuck(BLOCK) as call:
                wrapped, copy, encode = WRAPPERS[wrapper](sys.stdout)  # wraps the router
                sys.stdout = wrapped
                if order == "code ended":
                    call.finish()  # the wrapper is kept as the program's stream
                following = interpreter()
                following.namespace.update(started=started, finish=finish)
                code = "print('captured')\nstarted.set()\nfinish.wait(5)\n"
                thread = run_in_thread(following, code, results, "out")
                assert started.wait(5)
                operate(operation, labels[0], program)
                finish.set()
                thread.join(10)
                call.finish()
                assert sys.stdout is wrapped
                operate(operation, labels[1], program)
                sys.stdout.flush()
        finally:
            sys.stdout = stdout
        program.seek(0)
        written = program.read()

    assert results["out"] == "captured\n"
    if operation in ("print", "flush"):
        for label in labels:
            assert written.count(encode(label)) == 1, written
            if copy is not None:
                assert copy.getvalue().count(label) == 1, copy.getvalue()
    else:
        assert written == "" and (copy is None or copy.getvalue() == "")


def test_a_handler_holding_sys_stdout_from_a_call_writes_to_the_program_after_it():
    stdout = sys.stdout
    program, own = io.StringIO(), io.StringIO()
    sys.stdout = program
    try:
        with stuck(BLOCK) as call:
            sys.stdout = own  # the program's own stream, set after the timeout
            started, finish = threading.Event(), threading.Event()
            following = interpreter()
            following.namespace.update(started=started, finish=finish)
            results = {}
            thread = run_in_thread(following, "started.set()\nfinish.wait(5)\n", results, "out")
            assert started.wait(5)
            handler = logging.StreamHandler(sys.stdout)  # holds the router
            finish.set()
            thread.join(10)
            call.finish()
            assert sys.stdout is own
            handler.emit(logging.makeLogRecord({"msg": "held"}))
    finally:
        sys.stdout = stdout

    assert own.getvalue() == "held\n"
    assert program.getvalue() == ""


def test_a_program_that_silenced_its_stdout_stays_silent_while_a_call_runs(monkeypatch):
    real_stdout = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", real_stdout)
    monkeypatch.setattr(sys, "stdout", None)
    started, finish = threading.Event(), threading.Event()
    interp = interpreter()
    interp.namespace.update(started=started, finish=finish)
    results = {}

    thread = run_in_thread(interp, "print('captured')\nstarted.set()\nfinish.wait(5)\n", results, "out")
    assert started.wait(5)
    print("silenced")  # through the router, to the program's None stream
    finish.set()
    thread.join(10)

    assert results["out"] == "captured\n"
    assert real_stdout.getvalue() == ""
    assert sys.stdout is None


def test_an_interrupted_run_gives_up_its_call(monkeypatch):
    blocked = threading.Event()
    interp = interpreter(timeout=30)
    interp.namespace["blocked"] = blocked

    def interrupted_join(self, timeout=None):
        assert blocked.wait(5)
        raise KeyboardInterrupt  # what Ctrl-C in run() looks like

    code = "import threading\nworker = threading.Thread(target=release.wait)\nworker.start()\nblocked.set()\nrelease.wait(10)\n"
    stdout, release = sys.stdout, threading.Event()
    interp.namespace["release"] = release
    try:
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "join", interrupted_join)
            with pytest.raises(KeyboardInterrupt):
                interp.run(code)
        record = interp.namespace["worker"]._lllm_call
        assert record.abandoned and record.buffer is None
        program_stream = io.StringIO()
        sys.stdout = program_stream  # a later call captures even so
        assert interpreter().run("print('captured')") == "captured\n"
    finally:
        release.set()
        wait_for_calls_to_end()
        sys.stdout = stdout


def test_an_interrupt_landing_as_run_returns_releases_the_call():
    interp = interpreter(timeout=30)
    code = (
        "import _thread, threading, time\n"
        "worker = threading.Thread(target=int)\n"
        "worker.start()\n"
        "_thread.interrupt_main()\n"
        "time.sleep(0.05)\n"
    )

    # interrupt_main needs Python's SIGINT handler, which a shell running
    # the tests in the background (with &) replaces with SIG_IGN.
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            interp.run(code)
    finally:
        signal.signal(signal.SIGINT, previous)
    wait_for_calls_to_end()

    record = interp.namespace["worker"]._lllm_call
    assert record.buffer is None
    assert interp._running == []


def test_a_call_that_times_out_before_its_code_starts_is_not_running(monkeypatch):
    go, entered = threading.Event(), []
    real_enter = interpreter_module._enter

    def late_enter(call):
        entered.append(threading.current_thread())
        go.wait(5)
        real_enter(call)

    monkeypatch.setattr(interpreter_module, "_enter", late_enter)
    interp = interpreter(timeout=0.05)
    try:
        with pytest.raises(TimeoutError):
            interp.run("pass")
    finally:
        go.set()
        for thread in entered:
            thread.join(5)

    assert interp._running == []


# -- The Thread.start hook ----------------------------------------------------


def test_a_thread_start_wrapper_installed_during_a_call_does_not_recurse():
    interp = interpreter()
    interp.run("")  # the Thread.start hook is in place from the first call on
    saved = threading.Thread.start
    traced = []
    interp.namespace["traced"] = traced
    named = ("after the call", "in the next call")
    try:
        interp.run(  # what Sentry or OpenTelemetry do
            "import threading\n"
            "inner = threading.Thread.start\n"
            "def traced_start(self):\n"
            "    traced.append(self.name)\n"
            "    return inner(self)\n"
            "threading.Thread.start = traced_start\n"
        )
        after = threading.Thread(target=lambda: None, name=named[0])
        after.start()
        after.join()
        code = (
            "import threading\n"
            f"worker = threading.Thread(target=print, args=('captured',), name={named[1]!r})\n"
            "worker.start()\n"
            "worker.join()\n"
        )
        assert interp.run(code) == "captured\n"
        assert [name for name in traced if name in named] == list(named)
    finally:
        threading.Thread.start = saved


def test_threads_are_captured_again_after_the_original_start_is_put_back():
    interp = interpreter()
    interp.run("")
    hook = threading.Thread.start
    threading.Thread.start = interpreter_module._start_raw  # e.g. a test's cleanup
    try:
        code = "import threading\nt = threading.Thread(target=print, args=('w',))\nt.start()\nt.join()\n"
        assert interp.run(code) == "w\n"
    finally:
        threading.Thread.start = hook


def test_thread_start_behaves_after_all_calls_end(capsys):
    interpreter().run("import threading\nthreading.Thread(target=int).start()\n")
    wait_for_calls_to_end()

    worker = threading.Thread(target=print, args=("from the program",))
    worker.start()
    worker.join()

    assert getattr(worker, "_lllm_call", None) is None
    assert capsys.readouterr().out == "from the program\n"


FRESH_PROCESS = r'''
import importlib, sys, threading

how = sys.argv[1]
if how == "wrapt":  # how OpenTelemetry instruments threading
    import wrapt

    wrapt.wrap_function_wrapper(
        threading.Thread, "start", lambda wrapped, instance, args, kwargs: wrapped(*args, **kwargs)
    )
elif how == "descriptor":  # every read of Thread.start is a new object
    class Bound:
        def __init__(self, function, instance):
            self.function, self.instance = function, instance

        def __call__(self, *args):
            if self.instance is None:  # read from the class: the thread comes first
                return self.function(*args)
            return self.function(self.instance, *args)

    class Descriptor:
        def __init__(self, function):
            self.function = function

        def __get__(self, instance, owner):
            return Bound(self.function, instance)

    threading.Thread.start = Descriptor(threading.Thread.start)
if how != "plain":
    assert threading.Thread.start is not threading.Thread.start

from lllm.runtimes.native import Runtime
from lllm.runtimes.native.proxies import ProxyManager
from lllm.runtimes.native.proxies import interpreter as module

CODE = "import threading\nt = threading.Thread(target=print, args=('from a thread',))\nt.start()\nt.join()\n"
print(repr(module.AgentInterpreter(ProxyManager(runtime=Runtime())).run(CODE)))
importlib.reload(module)
worker = threading.Thread(target=int)
worker.start()  # the hook installed before the reload still works
worker.join()
print(repr(module.AgentInterpreter(ProxyManager(runtime=Runtime())).run(CODE)))
'''


@pytest.mark.parametrize("how", ["plain", "descriptor", "wrapt"])
def test_the_thread_start_hook_under_wrappers_and_a_reload(how):
    if how == "wrapt":
        pytest.importorskip("wrapt")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}

    done = subprocess.run(
        [sys.executable, "-c", FRESH_PROCESS, how],
        capture_output=True, text=True, timeout=120, env=env, cwd=ROOT,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["'from a thread\\n'", "'from a thread\\n'"]


# -- Long output ----------------------------------------------------------------


def test_print_keeps_its_arguments_and_reset_keeps_capture():
    interp = interpreter()

    assert interp.run("print('a', 'b', sep='-', end='!')") == "a-b!"
    interp.run("x = 1")
    interp.reset()
    assert interp.run("print('x' in globals())") == "False\n"
    threaded = (
        "import threading\n"
        "worker = threading.Thread(target=print, args=('from a thread',))\n"
        "worker.start()\n"
        "worker.join()\n"
    )
    assert interp.run(threaded) == "from a thread\n"


def test_copies_of_sys_stdout_keep_routing_by_thread(capsys):
    started, finish = threading.Event(), threading.Event()
    interp = interpreter()
    interp.namespace.update(started=started, finish=finish)
    results = {}

    code = (
        "import copy, sys\n"
        "copy.copy(sys.stdout).write('copied in the call\\n')\n"
        "started.set()\n"
        "finish.wait(5)\n"
    )
    thread = run_in_thread(interp, code, results, "out")
    assert started.wait(5)
    copy.copy(sys.stdout).write("copied in the program\n")  # while the call runs
    finish.set()
    thread.join(10)

    assert results["out"] == "copied in the call\n"
    assert capsys.readouterr().out == "copied in the program\n"


def test_long_output_keeps_the_end_of_a_traceback():
    interp = interpreter(max_output_chars=300)

    out = interp.run("def f(n):\n    return f(n - 1) if n else 1 / 0\n\nf(40)\n")

    assert out.startswith("Traceback (most recent call last):")
    assert out.rstrip().endswith("ZeroDivisionError: division by zero")
    assert "... (truncated)" in out
    assert len(out) == 300 + len("... (truncated)")


def test_truncation_splits_the_budget_between_head_and_tail():
    interp = interpreter(max_output_chars=10, truncation_indicator="[...]")

    assert interp.run("print('0123456789' * 3, end='')") == "01234[...]56789"
    assert interp.run("print('short', end='')") == "short"
    assert interpreter(max_output_chars=0).run("print('x' * 50, end='')") == "x" * 50
