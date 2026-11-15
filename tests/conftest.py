import importlib.abc
import sys

import pytest


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
