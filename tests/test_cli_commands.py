"""The single `lllm` command reaches every module CLI and plugin command."""

import json
import re
import sys
import types
from importlib.metadata import EntryPoint

import pytest

import lllm.cli
from lllm.cli import main


def _entry_points(monkeypatch, **commands):
    """Replace the installed `lllm.commands` entry points: name -> module."""

    points = [
        EntryPoint(name=name, value=value, group="lllm.commands")
        for name, value in commands.items()
    ]

    def entry_points(*, group):
        return [point for point in points if point.group == group]

    monkeypatch.setattr(lllm.cli, "entry_points", entry_points)


def _module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_init_writes_a_package_manifest(tmp_path, capsys):
    package = tmp_path / "briefs"

    assert main(["init", str(package), "--org", "demo"]) == 0

    manifest = (package / "lllm.toml").read_text(encoding="utf-8")
    assert 'org = "demo"' in manifest
    assert 'name = "briefs"' in manifest
    assert capsys.readouterr().out.strip().endswith("lllm.toml")


def test_hub_commands_dispatch_to_the_hub_cli(tmp_path, capsys):
    package = tmp_path / "echo"
    main(["init", str(package), "--org", "demo"])
    hub_root = tmp_path / "hub"
    capsys.readouterr()

    assert main(["hub", "--hub", str(hub_root), "publish", str(package), "--local"]) == 0
    assert main(["hub", "--hub", str(hub_root), "list"]) == 0

    assert "demo/echo" in capsys.readouterr().out


def test_plugin_commands_dispatch_through_lllm_commands_entry_points(monkeypatch, capsys):
    calls = []

    def plugin_main(argv):
        calls.append(argv)
        print("hello from the plugin")
        return 0

    _module(monkeypatch, "fake_lllm_plugin", main=plugin_main)
    _entry_points(monkeypatch, greet="fake_lllm_plugin")

    assert main(["greet", "--loud", "world"]) == 0

    assert calls == [["--loud", "world"]]
    assert capsys.readouterr().out == "hello from the plugin\n"


def test_an_installed_package_adds_its_command_through_its_entry_point(monkeypatch):
    calls = []
    _module(monkeypatch, "fake_sssn_cli", main=lambda argv: calls.append(argv) or 3)
    _entry_points(monkeypatch, network="fake_sssn_cli")

    assert main(["network", "--store", "store", "channels"]) == 3

    assert calls == [["--store", "store", "channels"]]


def test_plugins_cannot_replace_core_commands(tmp_path, monkeypatch, capsys):
    _module(monkeypatch, "fake_lllm_plugin", main=lambda argv: pytest.fail("plugin ran"))
    _entry_points(monkeypatch, init="fake_lllm_plugin", hub="fake_lllm_plugin")

    assert main(["init", str(tmp_path / "briefs"), "--org", "demo"]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    assert (tmp_path / "briefs" / "lllm.toml").is_file()
    assert "Added by fake_lllm_plugin" not in capsys.readouterr().out


@pytest.mark.parametrize(("command", "package"), [("network", "sssn"), ("shell", "aaax")])
def test_known_plugins_fall_back_to_the_package_cli(monkeypatch, command, package):
    calls = []
    _module(monkeypatch, package)
    _module(monkeypatch, f"{package}.cli", main=lambda argv: calls.append(argv) or 0)
    _entry_points(monkeypatch)

    assert main([command, "--help"]) == 0

    assert calls == [["--help"]]


@pytest.mark.parametrize(("command", "package"), [("network", "sssn"), ("shell", "aaax")])
def test_a_missing_plugin_package_prints_what_to_install(
    monkeypatch, uninstalled, capsys, command, package
):
    uninstalled(package)
    _entry_points(monkeypatch)

    assert main([command, "anything"]) == 2

    captured = capsys.readouterr()
    assert captured.err == f"lllm {command} needs the {package} package: pip install {package}\n"
    assert captured.out == ""


def test_a_plugin_missing_its_own_dependency_raises(tmp_path, monkeypatch):
    (tmp_path / "fake_broken_lllm_plugin.py").write_text(
        "import fake_dependency_that_is_not_installed\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    _entry_points(monkeypatch, broken="fake_broken_lllm_plugin")

    with pytest.raises(ModuleNotFoundError) as exc:
        main(["broken"])

    assert exc.value.name == "fake_dependency_that_is_not_installed"


def test_top_level_help_lists_core_and_installed_plugin_commands(
    monkeypatch, uninstalled, capsys
):
    uninstalled("aaax")
    _module(monkeypatch, "fake_lllm_plugin", main=lambda argv: 0)
    _entry_points(monkeypatch, greet="fake_lllm_plugin")
    _module(monkeypatch, "sssn")  # installed without an entry point

    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    listed = re.findall(r"^    (\S+)\s", out, re.M)
    assert listed == ["create", "init", "setup", "inspect", "serve", "hub", "greet", "network"]
    assert re.search(r"^    greet\s+Added by fake_lllm_plugin$", out, re.M)
    assert re.search(r"^    network\s+SSSN channel stores$", out, re.M)
    assert "More commands: lllm shell (pip install aaax)." in out


def test_network_commands_dispatch_to_the_sssn_cli(tmp_path, capsys):
    pytest.importorskip("sssn", reason="lllm network needs the sssn package")
    store = tmp_path / "store"

    assert main(["network", "--store", str(store), "create-channel", "events"]) == 0
    assert main(["network", "--store", str(store), "channels"]) == 0

    assert "events" in capsys.readouterr().out


def test_shell_commands_dispatch_to_the_aaax_cli(tmp_path, capsys):
    pytest.importorskip("aaax", reason="lllm shell needs the aaax package")
    strategy = tmp_path / "strategy.py"
    strategy.write_text(
        "from aaax import Strategy\n"
        "strategy = Strategy('demo', description='Demo strategy.')\n",
        encoding="utf-8",
    )

    assert main(["shell", "inspect", str(strategy), "--json"]) == 0

    assert json.loads(capsys.readouterr().out)["name"] == "demo"
