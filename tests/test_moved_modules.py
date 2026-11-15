"""lllm.network and lllm.shell moved to the sssn and aaax packages.

The old import paths are deprecated aliases: they forward to the new packages,
or say what to install when the package is missing.
"""

import importlib
import re
import sys
import types

import pytest

import lllm

MOVED = [
    pytest.param(
        "lllm.network",
        "sssn",
        ("core", "stores", "server", "client", "resolver", "processing", "resources", "cli"),
        id="network",
    ),
    pytest.param(
        "lllm.shell",
        "aaax",
        ("cli", "loader", "models", "packages", "server", "strategy"),
        id="shell",
    ),
]


@pytest.fixture(autouse=True)
def forget_moved_modules():
    """Import lllm.network, lllm.shell, sssn, and aaax afresh in each test."""

    roots = ("lllm.network", "lllm.shell", "sssn", "aaax")

    def tracked(name):
        return any(name == root or name.startswith(f"{root}.") for root in roots)

    saved = {name: module for name, module in sys.modules.items() if tracked(name)}
    saved_attrs = {name: vars(lllm)[name] for name in ("network", "shell") if name in vars(lllm)}
    for name in saved:
        del sys.modules[name]
    yield
    for name in [name for name in sys.modules if tracked(name)]:
        del sys.modules[name]
    sys.modules.update(saved)
    for name in ("network", "shell"):
        vars(lllm).pop(name, None)
    vars(lllm).update(saved_attrs)


def _fake_package(package, submodules):
    """Put a fake package with submodules and one nested module in sys.modules."""

    root = types.ModuleType(package)
    root.__path__ = []
    root.Thing = type("Thing", (), {})
    root.helper = lambda: "helped"
    root.extra = "not in __all__"
    root.__all__ = ["Thing", "helper"]
    sys.modules[package] = root
    for name in submodules:
        module = types.ModuleType(f"{package}.{name}")
        module.__path__ = []
        module.marker = name
        sys.modules[module.__name__] = module
        setattr(root, name, module)
    nested = types.ModuleType(f"{package}.{submodules[0]}.nested")
    nested.marker = "nested"
    sys.modules[nested.__name__] = nested
    setattr(sys.modules[f"{package}.{submodules[0]}"], "nested", nested)
    return root


@pytest.mark.parametrize(("old", "package", "submodules"), MOVED)
def test_a_moved_module_says_what_to_install(uninstalled, old, package, submodules):
    uninstalled(package)

    with pytest.raises(ImportError, match=re.escape(
        f"{old} moved to the {package} package: pip install {package}"
    )) as raised:
        importlib.import_module(old)

    assert raised.value.name == package
    assert old not in sys.modules


@pytest.mark.parametrize(("old", "package", "submodules"), MOVED)
def test_a_moved_module_forwards_to_its_package(old, package, submodules):
    fake = _fake_package(package, submodules)

    with pytest.warns(DeprecationWarning) as record:
        __import__(old)
    module = sys.modules[old]

    assert [str(warning.message) for warning in record] == [
        f"{old} moved to the {package} package; use `import {package}` instead."
    ]
    assert record[0].filename == __file__
    assert module.__all__ == ["Thing", "helper"]
    assert module.Thing is fake.Thing
    assert module.helper() == "helped"
    assert module.extra == "not in __all__"
    with pytest.raises(AttributeError, match=f"module '{old}' has no attribute 'missing'"):
        module.missing


@pytest.mark.parametrize(("old", "package", "submodules"), MOVED)
def test_old_submodule_imports_are_the_new_modules(old, package, submodules):
    _fake_package(package, submodules)

    with pytest.warns(DeprecationWarning):
        module = importlib.import_module(old)

    for name in submodules:
        new = sys.modules[f"{package}.{name}"]
        assert sys.modules[f"{old}.{name}"] is new
        assert getattr(module, name) is new
        assert importlib.import_module(f"{old}.{name}").marker == name
    first = submodules[0]
    nested = importlib.import_module(f"{old}.{first}.nested")
    assert nested is sys.modules[f"{package}.{first}.nested"]
    namespace = {}
    exec(f"from {old}.{first}.nested import marker", namespace)
    assert namespace["marker"] == "nested"


@pytest.mark.parametrize(("old", "package", "submodules"), MOVED)
def test_a_package_missing_its_own_dependency_raises_that_error(
    tmp_path, monkeypatch, old, package, submodules
):
    (tmp_path / package).mkdir()
    (tmp_path / package / "__init__.py").write_text(
        "import fake_dependency_that_is_not_installed\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as raised:
        importlib.import_module(old)

    assert raised.value.name == "fake_dependency_that_is_not_installed"


def test_lllm_network_is_sssn_when_installed():
    sssn = pytest.importorskip("sssn", reason="needs the sssn package")

    with pytest.warns(DeprecationWarning, match="import sssn"):
        import lllm.network

    from lllm.network.server.fastapi import create_app

    assert lllm.network.LocalStore is sssn.LocalStore
    assert lllm.network.server is sssn.server
    assert create_app is sssn.server.fastapi.create_app


def test_lllm_shell_is_aaax_when_installed():
    aaax = pytest.importorskip("aaax", reason="needs the aaax package")

    with pytest.warns(DeprecationWarning, match="import aaax"):
        import lllm.shell

    assert lllm.shell.Strategy is aaax.Strategy
    assert lllm.shell.strategy is aaax.strategy
