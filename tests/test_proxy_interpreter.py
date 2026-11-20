"""run_python: each call captures its own output, and long output keeps both ends."""

import sys
import threading

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
    )
    lines = interp.run(code).splitlines()

    assert lines[:2] == ["raw", "{'a': 1}"]
    assert sorted(lines[2:]) == ["w1", "w2"]


def test_print_keeps_its_arguments_and_reset_keeps_capture():
    interp = interpreter()

    assert interp.run("print('a', 'b', sep='-', end='!')") == "a-b!"
    interp.run("x = 1")
    interp.reset()
    assert interp.run("print('x' in globals())") == "False\n"


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
