"""lllm.Resolver resolves tactic refs on its own; channel refs need sssn."""

from pathlib import Path

import pytest

from lllm import Resolver, as_tactic

TACTIC_REF = "lllm://demo/briefs/tactics/shout"
CHANNEL_REF = "lllm://demo/briefs/channels/events"
SSSN_HINT = "Channel and snapshot refs need the sssn package: pip install sssn"


def _write_config(root: Path) -> Path:
    (root / ".lllm").mkdir()
    (root / ".lllm" / "config.toml").write_text(
        f"""
[refs."{TACTIC_REF}"]
url = "http://127.0.0.1:8000"

[refs."{CHANNEL_REF}"]
store = ".lllm/network"

[settings]
region = "us"
""",
        encoding="utf-8",
    )
    return root


def test_a_resolver_runs_tactic_refs_without_sssn(uninstalled):
    uninstalled("sssn")
    resolver = Resolver()
    resolver.register(TACTIC_REF, as_tactic(lambda text: text.upper(), name="shout"))

    assert resolver.run(TACTIC_REF, "hi") == "HI"
    assert resolver.refs() == (TACTIC_REF,)
    with pytest.raises(ImportError, match=SSSN_HINT):
        resolver.store(CHANNEL_REF)


def test_a_config_with_channel_refs_loads_without_sssn(tmp_path, uninstalled):
    uninstalled("sssn")

    resolver = Resolver.from_config(_write_config(tmp_path))

    assert resolver.setting("region") == "us"
    assert resolver.refs() == (TACTIC_REF,)
    assert resolver.tactic(TACTIC_REF).info().name == "shout"
    with pytest.raises(ImportError, match=SSSN_HINT):
        resolver.resolve(CHANNEL_REF)
    with pytest.raises(ImportError, match=SSSN_HINT):
        resolver.channels


def test_channel_refs_resolve_to_sssn_stores(tmp_path):
    sssn = pytest.importorskip("sssn", reason="channel refs need the sssn package")

    resolver = Resolver.from_config(_write_config(tmp_path))
    store = resolver.resolve(CHANNEL_REF)

    assert isinstance(resolver.channels, sssn.ChannelResolver)
    assert isinstance(store, sssn.LocalStore)
    assert resolver.refs() == (CHANNEL_REF, TACTIC_REF)
