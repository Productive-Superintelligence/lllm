"""MkDocs hooks for the LLLM docs (see ``hooks:`` in mkdocs.yml)."""

from __future__ import annotations

import re

# tests/test_docs_examples.py reads pytest-examples settings from a code
# fence's info string, as in ```python test="skip" reason="needs a server".
# Material treats them as unknown fence options and then drops the block's
# title, so they are removed before the page is rendered.
_TEST_SETTINGS = ("test", "reason", "lint", "upgrade", "py", "requires", "duration")
_FENCE = re.compile(r"^(?P<open>[ \t]*(?:`{3,}|~{3,}))(?P<info>[^`\n]*)$", re.M)
_SETTING = re.compile(r'\s+(?:%s)="[^"]*"' % "|".join(_TEST_SETTINGS))


def strip_test_settings(markdown: str) -> str:
    return _FENCE.sub(lambda m: m.group("open") + _SETTING.sub("", m.group("info")), markdown)


def on_page_markdown(markdown: str, **kwargs) -> str:
    return strip_test_settings(markdown)
