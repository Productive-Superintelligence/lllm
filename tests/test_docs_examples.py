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

from pathlib import Path

import pytest

pytest_examples = pytest.importorskip("pytest_examples")
pytest.importorskip("litellm", reason="docs examples use the native runtime")

from pytest_examples import CodeExample, EvalExample, find_examples  # noqa: E402

from docs_model import answer  # noqa: E402
from lllm.testing import FunctionModel  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.mark.parametrize("example", find_examples(str(DOCS)), ids=str)
def test_docs_example(example: CodeExample, eval_example: EvalExample, tmp_path, monkeypatch):
    settings = example.prefix_settings()
    if settings.get("test") == "skip":
        pytest.skip(settings.get("reason", "marked test=skip"))
    monkeypatch.chdir(tmp_path)  # examples that write files write them here
    eval_example.set_config(line_length=88)
    with FunctionModel(answer):
        if eval_example.update_examples:
            eval_example.run_print_update(example)
        else:
            eval_example.run_print_check(example)
