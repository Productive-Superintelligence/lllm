"""run_python: each call captures its own output, and long output keeps both ends."""

import copy
import io
import sys
import threading
import time

import pytest

from lllm.runtimes.native import Runtime
from lllm.runtimes.native.proxies import AgentInterpreter, ProxyManager
from lllm.runtimes.native.proxies import interpreter as interpreter_module


def interpreter(**kwargs):
    return AgentInterpreter(ProxyManager(runtime=Runtime()), **kwargs)


def run_in_thread(interp, code, results, key):
    thread = threading.Thread(target=lambda: results.__setitem__(key, interp.run(code)))
    thread.start()
    return thread


def wait_for_calls_to_end(timeout=10):
    """Wait until no call's code is running, timed-out code included."""

    deadline = time.monotonic() + timeout
    while getattr(interpreter_module, "_flights", None) or getattr(
        interpreter_module, "_active_captures", 0
    ):
        assert time.monotonic() < deadline, "a call's code is still running"
        time.sleep(0.01)


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
    started.wait(5)
    print("program")  # the main thread, while the agent's code is running
    finish.set()
    thread.join(10)

    assert results["out"] == "agent\ndone\n"
    assert capsys.readouterr().out == "program\n"


def test_a_timed_out_run_does_not_swallow_later_prints(capsys):
    finish, finished = threading.Event(), threading.Event()
    interp = interpreter(timeout=0.2)
    interp.namespace.update(finish=finish, finished=finished)

    try:
        interp.run("finish.wait(5)\nprint('late')\nfinished.set()\n")
    except TimeoutError:
        pass
    else:  # pragma: no cover - the run must time out
        raise AssertionError("expected a timeout")
    print("after the timeout")  # the code is still running in the background
    finish.set()
    finished.wait(5)

    assert capsys.readouterr().out == "after the timeout\n"
    assert interp.run("print('next')") == "next\n"
    wait_for_calls_to_end()


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


def test_a_timed_out_calls_threads_do_not_print_into_the_next_call(capsys):
    next_running, late_done, release = (threading.Event() for _ in range(3))
    interp = interpreter(timeout=0.3)
    interp.namespace.update(next_running=next_running, late_done=late_done, release=release)
    first = (
        "import sys, threading\n"
        "def late():\n"
        "    next_running.wait(5)\n"
        "    print('late print')\n"
        "    sys.stdout.write('late write\\n')\n"
        "    late_done.set()\n"
        "threading.Thread(target=late).start()\n"
        "release.wait(5)\n"
    )

    with pytest.raises(TimeoutError):
        interp.run(first)
    # The timed-out call's thread prints while the next call is in flight.
    second = "next_running.set()\nlate_done.wait(5)\nprint('next')\n"
    assert interp.run(second) == "next\n"
    release.set()
    wait_for_calls_to_end()

    assert "late" not in capsys.readouterr().out


def test_a_timed_out_calls_threads_print_to_the_program_once_its_code_ends(capsys):
    release, may_print, printed = (threading.Event() for _ in range(3))
    interp = interpreter(timeout=0.2)
    interp.namespace.update(release=release, may_print=may_print, printed=printed)
    code = (
        "import threading\n"
        "def late():\n"
        "    may_print.wait(5)\n"
        "    print('from the late thread')\n"
        "    printed.set()\n"
        "threading.Thread(target=late).start()\n"
        "release.wait(5)\n"
    )

    with pytest.raises(TimeoutError):
        interp.run(code)
    release.set()
    wait_for_calls_to_end()
    started, finish = threading.Event(), threading.Event()
    other = interpreter()
    other.namespace.update(started=started, finish=finish)
    results = {}
    thread = run_in_thread(other, "started.set()\nfinish.wait(5)\n", results, "out")
    started.wait(5)
    may_print.set()  # while another call is in flight
    printed.wait(5)
    finish.set()
    thread.join(10)

    assert results["out"] == ""
    assert capsys.readouterr().out == "from the late thread\n"


def test_a_thread_left_running_writes_to_the_program_after_its_call(capsys):
    proceed, done = threading.Event(), threading.Event()
    interp = interpreter()
    interp.namespace.update(proceed=proceed, done=done)
    code = (
        "import sys, threading\n"
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
    started.wait(5)
    proceed.set()  # the thread prints while another call is in flight
    done.wait(5)
    finish.set()
    thread.join(10)

    assert results["out"] == ""
    assert capsys.readouterr().out == "after the call\n"


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

runner = doctest.DocTestRunner()
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
    started.wait(5)
    copy.copy(sys.stdout).write("copied in the program\n")  # while the call runs
    finish.set()
    thread.join(10)

    assert results["out"] == "copied in the call\n"
    assert capsys.readouterr().out == "copied in the program\n"


def test_a_timed_out_call_that_replaced_sys_stdout_puts_it_back_when_its_code_ends(capsys):
    stdout = sys.stdout
    release = threading.Event()
    interp = interpreter(timeout=0.2)
    interp.namespace["release"] = release

    with pytest.raises(TimeoutError):
        interp.run("import io, sys\nsys.stdout = io.StringIO()\nrelease.wait(5)\n")
    release.set()
    wait_for_calls_to_end()

    assert sys.stdout is stdout
    print("visible")
    assert capsys.readouterr().out == "visible\n"


def test_a_stream_the_program_sets_after_a_timeout_is_kept():
    stdout = sys.stdout
    release = threading.Event()
    interp = interpreter(timeout=0.2)
    interp.namespace["release"] = release

    with pytest.raises(TimeoutError):
        interp.run("release.wait(5)\n")
    program_stream = io.StringIO()
    sys.stdout = program_stream  # the program has moved on and sets its own
    try:
        release.set()
        wait_for_calls_to_end()
        assert sys.stdout is program_stream
    finally:
        sys.stdout = stdout


def test_calls_capture_while_timed_out_code_runs_after_the_program_sets_its_stdout():
    stdout = sys.stdout
    release = threading.Event()
    stuck = interpreter(timeout=0.2)
    stuck.namespace["release"] = release

    with pytest.raises(TimeoutError):
        stuck.run("release.wait(10)\n")
    program_stream = io.StringIO()
    sys.stdout = program_stream  # e.g. a test runner between two tests
    try:
        assert interpreter().run("print('captured')") == "captured\n"
        print("from the program")
        assert program_stream.getvalue() == "from the program\n"
        release.set()
        wait_for_calls_to_end()
        assert sys.stdout is program_stream
    finally:
        sys.stdout = stdout


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
@pytest.mark.parametrize("redirecting_call_ends_first", [True, False])
def test_overlapping_calls_where_one_redirects_leave_the_program_its_stdout(
    redirect, redirecting_call_ends_first
):
    stdout = sys.stdout
    redirected, first_may_end, second_may_end = (threading.Event() for _ in range(3))
    first, second = interpreter(), interpreter()
    for interp in (first, second):
        interp.namespace.update(
            redirected=redirected, first_may_end=first_may_end, second_may_end=second_may_end
        )
    results = {}

    one = run_in_thread(first, REDIRECTS[redirect], results, "first")
    two = run_in_thread(second, "redirected.wait(5)\nsecond_may_end.wait(5)\n", results, "second")
    redirected.wait(5)
    endings = [(first_may_end, one), (second_may_end, two)]
    if not redirecting_call_ends_first:
        endings.reverse()
    for may_end, thread in endings:
        may_end.set()
        thread.join(10)

    assert set(results) == {"first", "second"}
    assert sys.stdout is stdout


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
    threading.Thread.start = interpreter_module._original_start  # e.g. a test's cleanup
    try:
        code = "import threading\nt = threading.Thread(target=print, args=('w',))\nt.start()\nt.join()\n"
        assert interp.run(code) == "w\n"
    finally:
        threading.Thread.start = hook


def test_threads_kept_after_a_call_do_not_keep_its_output():
    interp = interpreter(timeout=0.3)
    code = (
        "import threading\n"
        "worker = threading.Thread(target=print, args=('x' * 100_000,))\n"
        "worker.start()\n"
        "worker.join()\n"
    )
    assert len(interp.run(code)) == 5000 + len("... (truncated)")
    record = interp.namespace["worker"]._lllm_call
    assert record.returned and record.buffer is None

    release = threading.Event()
    interp.namespace["release"] = release
    with pytest.raises(TimeoutError):
        interp.run("worker = threading.Thread(target=release.wait)\nworker.start()\nrelease.wait(5)\n")
    record = interp.namespace["worker"]._lllm_call
    assert record.abandoned and record.buffer is None
    release.set()
    wait_for_calls_to_end()


def test_thread_start_behaves_after_all_calls_end(capsys):
    interpreter().run("import threading\nthreading.Thread(target=int).start()\n")
    wait_for_calls_to_end()

    worker = threading.Thread(target=print, args=("from the program",))
    worker.start()
    worker.join()

    assert getattr(worker, "_lllm_call", None) is None
    assert capsys.readouterr().out == "from the program\n"


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
