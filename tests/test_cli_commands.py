"""The single `lllm` command reaches every module CLI."""

import json

import pytest

from lllm.cli import main


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


def test_network_commands_dispatch_to_the_network_cli(tmp_path, capsys):
    store = tmp_path / "store"

    assert main(["network", "--store", str(store), "create-channel", "events"]) == 0
    assert main(["network", "--store", str(store), "channels"]) == 0

    assert "events" in capsys.readouterr().out


def test_shell_commands_dispatch_to_the_shell_cli(tmp_path, capsys):
    strategy = tmp_path / "strategy.py"
    strategy.write_text(
        "from lllm.shell import Strategy\n"
        "strategy = Strategy('demo', description='Demo strategy.')\n",
        encoding="utf-8",
    )

    assert main(["shell", "inspect", str(strategy), "--json"]) == 0

    assert json.loads(capsys.readouterr().out)["name"] == "demo"


def test_top_level_help_lists_module_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("create", "init", "setup", "inspect", "serve", "hub", "network", "shell"):
        assert command in out
