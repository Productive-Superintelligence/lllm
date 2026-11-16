"""Every Python example in the docs runs, and prints what the page says it prints.

Examples call real model names, as users would; here every model call is
answered by ``docs_model.answer`` (canned replies keyed by what the example
asks), so the outputs shown after ``#>`` are checked on every test run.

Mark an example that cannot run here (it needs a server, files, or a real
provider) with ``{test="skip"}`` after the language in its fence.
Update the ``#>`` outputs after changing an example with:

    pytest tests/test_docs_examples.py --update-examples
"""

from __future__ import annotations

import builtins
import inspect
from pathlib import Path

import pytest

pytest_examples = pytest.importorskip("pytest_examples")
pytest.importorskip("litellm", reason="docs examples use the native runtime")

from pytest_examples import CodeExample, EvalExample, find_examples  # noqa: E402
from pytest_examples import run_code  # noqa: E402

from docs_model import answer  # noqa: E402
from lllm.testing import FunctionModel  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"
_real_print = builtins.print


class _ExamplePrint(run_code.MockPrintFunction):
    """Record the example's own prints; pass prints from code it execs through.

    Code an example runs with exec, such as an agent's ``run_python``, has the
    filename ``<string>``, which pytest-examples' check (``Path.samefile``)
    cannot compare, so its prints go to the real ``print``, where the
    interpreter captures them. ``__call__`` is reimplemented rather than
    wrapped, because pytest-examples finds the caller at a fixed stack depth.
    """

    def __call__(self, *args, sep=" ", **kwargs):
        frame = inspect.stack()[run_code.parent_frame_id]
        try:
            ours = self._include_file(frame, args)
        except OSError:
            ours = False
        if not ours:
            _real_print(*args, sep=sep, **kwargs)
            return
        lineno = self._find_line_number(frame)
        self.statements.append(run_code.PrintStatement(lineno, sep, [run_code.Arg(a) for a in args]))


@pytest.mark.parametrize("example", list(find_examples(str(DOCS))), ids=str)
def test_docs_example(example: CodeExample, eval_example: EvalExample, tmp_path, monkeypatch):
    settings = example.prefix_settings()
    if settings.get("test") == "skip":
        pytest.skip(settings.get("reason", "marked test=skip"))
    monkeypatch.chdir(tmp_path)  # examples that write files write them here
    monkeypatch.setattr(run_code, "MockPrintFunction", _ExamplePrint)
    eval_example.set_config(line_length=76)  # the code column's width
    with FunctionModel(answer):
        if eval_example.update_examples:
            eval_example.run_print_update(example)
        else:
            eval_example.run_print_check(example)
