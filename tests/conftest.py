import importlib.abc
import os
import sys

import pytest

# Use the model price table bundled with LiteLLM instead of the copy it would
# download at import time, so costs in tests (and in the docs examples) are
# the same on every run.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


class _Uninstalled(importlib.abc.MetaPathFinder):
    """Import hook that makes packages look uninstalled."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.names:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@pytest.fixture
def uninstalled(monkeypatch):
    """Make packages look uninstalled for one test: ``uninstalled("sssn")``.

    Works whether or not the package is installed, so tests of the "missing
    package" paths also pass where sssn and aaax are present.
    """

    def hide(*names: str) -> None:
        for name in list(sys.modules):
            if name.split(".", 1)[0] in names:
                monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "meta_path", [_Uninstalled(names), *sys.meta_path])

    return hide
