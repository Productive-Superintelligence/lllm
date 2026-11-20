"""run_python: each call captures its own output, and long output keeps both ends."""

import copy
import sys
import threading

import pytest

from lllm.runtimes.native import Runtime
from lllm.runtimes.native.proxies import AgentInterpreter, ProxyManager


def interpreter(**kwargs):
    return AgentInterpreter(ProxyManager(runtime=Runtime()), **kwargs)


def run_in_thread(interp, code, results, key):
    thread = threading.Thread(target=lambda: results.__setitem__(key, interp.run(code)))
    thread.start()
    return thread


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

    assert "late" not in capsys.readouterr().out


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
