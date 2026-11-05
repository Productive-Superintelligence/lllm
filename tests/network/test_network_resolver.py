from pathlib import Path
from types import MappingProxyType

import pytest

from lllm.network import (
    AsyncRemoteStore,
    LocalStore,
    RemoteStore,
    ChannelRef,
    ChannelRefError,
    ChannelResolver,
)


CHANNEL_REF = "lllm://demo/combo/channels/events"
SNAPSHOT_REF = "lllm://demo/combo/snapshots/latest"
TACTIC_REF = "lllm://demo/combo/tactics/analyze"
SERVICE_REF = "lllm://demo/combo/services/api"


def test_resolver_loads_channel_refs_from_shared_local_config(tmp_path):
    resolver = ChannelResolver.from_text(
        f"""
[refs."{CHANNEL_REF}"]
store = ".lllm/network"

[refs."{SNAPSHOT_REF}"]
store = ".lllm/network"
retention = "latest"

[refs."{TACTIC_REF}"]
url = "http://lllm/tactics/analyze"

[refs."{SERVICE_REF}"]
url = "http://sssn"

[stores.default]
path = ".lllm/network"
""".lstrip(),
        root=tmp_path,
    )

    assert resolver.refs() == (CHANNEL_REF, SNAPSHOT_REF)
    assert resolver.stores() == {"default": {"path": ".lllm/network"}}
    assert resolver.store("default") == {"path": ".lllm/network"}

    channel = resolver.resolve(CHANNEL_REF)
    snapshot = resolver.resolve(SNAPSHOT_REF)

    assert channel.resource_kind == "channels"
    assert channel.name == "events"
    assert channel.store == ".lllm/network"
    assert channel.metadata == {}
    assert snapshot.resource_kind == "snapshots"
    assert snapshot.name == "latest"
    assert snapshot.metadata == {"retention": "latest"}
    assert resolver.store_path(CHANNEL_REF) == (tmp_path / ".lllm/network").resolve()

    store = resolver.local_store(CHANNEL_REF)

    assert isinstance(store, LocalStore)
    assert store.root == (tmp_path / ".lllm/network").resolve()


def test_resolver_uses_config_file_parent_as_relative_root(tmp_path):
    config = tmp_path / "workspace" / ".lllm" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f"""
[refs."{CHANNEL_REF}"]
store = "data/sssn"
""".lstrip(),
        encoding="utf-8",
    )

    resolver = ChannelResolver.from_config(config)

    assert resolver.store_path(CHANNEL_REF) == (
        tmp_path / "workspace" / "data" / "sssn"
    ).resolve()


def test_resolver_creates_clients_for_http_channel_refs():
    resolver = ChannelResolver()
    transport = object()
    resolver.bind(CHANNEL_REF, url="http://testserver/api")

    client = resolver.client(CHANNEL_REF, timeout=3.0, transport=transport)
    async_client = resolver.async_client(CHANNEL_REF, timeout=5.0, transport=transport)

    assert isinstance(client, RemoteStore)
    assert client.base_url == "http://testserver/api"
    assert client.timeout == 3.0
    assert client.transport is transport
    assert isinstance(async_client, AsyncRemoteStore)
    assert async_client.base_url == "http://testserver/api"
    assert async_client.timeout == 5.0
    assert async_client.transport is transport

    with pytest.raises(ChannelRefError, match="local store"):
        resolver.store_path(CHANNEL_REF)


def test_resolver_rejects_clients_for_store_only_refs(tmp_path):
    resolver = ChannelResolver(root=tmp_path)
    resolver.bind(CHANNEL_REF, store=".lllm/network")

    with pytest.raises(ChannelRefError, match="HTTP URL"):
        resolver.client(CHANNEL_REF)

    with pytest.raises(ChannelRefError, match="HTTP URL"):
        resolver.async_client(CHANNEL_REF)


def test_resolver_rejects_invalid_refs_and_ignores_other_layers(tmp_path):
    resolver = ChannelResolver.from_text(
        f"""
[refs."{TACTIC_REF}"]
url = "http://lllm/tactics/analyze"

[refs."{SERVICE_REF}"]
url = "http://sssn"
""".lstrip(),
        root=tmp_path,
    )

    assert resolver.refs() == ()

    for ref in (
        None,
        123,
        b"lllm://demo/combo/channels/events",
        "",
        "   ",
    ):
        with pytest.raises(ChannelRefError, match="non-empty string"):
            ChannelRef.parse(ref)  # type: ignore[arg-type]
        with pytest.raises(ChannelRefError, match="non-empty string"):
            ChannelResolver().bind(ref, store=".lllm/network")  # type: ignore[arg-type]

    for ref in (
        "not-a-ref",
        "lllm://demo/combo/widgets/events",
        "lllm://demo/combo/channels/events?env=dev",
        "lllm://demo/combo/channels//events",
        "lllm://demo/../channels/events",
        "lllm://demo/combo/channels/..",
        "lllm://demo org/combo/channels/events",
        "lllm://demo/combo pkg/channels/events",
        "lllm://demo/combo/chan nels/events",
        "lllm://demo/combo/channels/event name",
        "lllm://demo%2Forg/combo/channels/events",
        "lllm://demo/combo%2Fpkg/channels/events",
        "lllm://demo/combo/channels/event%2Fname",
        "lllm://demo/combo/channels/event%5Cname",
        "lllm://demo/combo/channels/%2E%2E",
        "lllm://demo/combo/channels/event%20name",
        "lllm://demo/combo/channels/event%3Aname",
    ):
        with pytest.raises(ChannelRefError):
            ChannelResolver().bind(ref, store=".lllm/network")
        with pytest.raises(ChannelRefError):
            ChannelResolver().resolve(ref)


@pytest.mark.parametrize(
    "ref",
    (
        "http://demo/combo/tactics/analyze",
        "lllm://demo/combo/tactics/analyze?env=dev",
        "lllm://demo/combo/tactics/analyze#latest",
        "lllm://demo/combo/tactics/bad name",
        "lllm://demo/combo/tactics/analyze%2Fhidden",
    ),
)
def test_resolver_validates_ignored_non_channel_config_refs(tmp_path, ref):
    with pytest.raises(ChannelRefError):
        ChannelResolver.from_text(
            f"""
[refs."{ref}"]
url = "http://lllm/tactics/analyze"
""".lstrip(),
            root=tmp_path,
        )


def test_resolver_requires_exactly_one_concrete_target(tmp_path):
    with pytest.raises(ChannelRefError, match="one concrete target"):
        ChannelResolver.from_text(
            f"""
[refs."{CHANNEL_REF}"]
""".lstrip(),
            root=tmp_path / "missing-target",
        )

    with pytest.raises(ChannelRefError, match="only one concrete target"):
        ChannelResolver.from_text(
            f"""
[refs."{CHANNEL_REF}"]
url = "http://sssn"
store = ".lllm/network"
""".lstrip(),
            root=tmp_path / "many-targets",
        )

    for index, target_line in enumerate(
        (
            "url = 123",
            'url = ""',
            'url = "   "',
            "store = false",
            'store = ""',
            'store = "   "',
            'store = "bad store"',
            'path = ["x"]',
            'path = ""',
            'path = "   "',
            'path = "bad path"',
        ),
        start=1,
    ):
        with pytest.raises(ChannelRefError, match="non-empty string"):
            ChannelResolver.from_text(
                f"""
[refs."{CHANNEL_REF}"]
{target_line}
""".lstrip(),
                root=tmp_path / f"bad-target-{index}",
            )


def test_resolver_rejects_whitespace_bearing_store_targets(tmp_path):
    resolver = ChannelResolver()

    with pytest.raises(ChannelRefError, match="without whitespace"):
        resolver.bind(CHANNEL_REF, store="bad store")
    with pytest.raises(ChannelRefError, match="without whitespace"):
        resolver.bind(CHANNEL_REF, path="bad path")

    for index, target_line in enumerate(
        ('store = "bad store"', 'path = "bad path"'),
        start=1,
    ):
        with pytest.raises(ChannelRefError, match="without whitespace"):
            ChannelResolver.from_text(
                f"""
[refs."{CHANNEL_REF}"]
{target_line}
""".lstrip(),
                root=tmp_path / f"bad-store-target-{index}",
            )


def test_resolver_rejects_malformed_url_targets(tmp_path):
    resolver = ChannelResolver()
    for url in (
        "service",
        "/service",
        "ftp://service",
        "http://",
    ):
        with pytest.raises(ChannelRefError, match="absolute HTTP"):
            resolver.bind(CHANNEL_REF, url=url)
    with pytest.raises(ChannelRefError, match="without whitespace"):
        resolver.bind(CHANNEL_REF, url="http://test server")

    for index, url in enumerate(("service", "/service", "ftp://service"), start=1):
        with pytest.raises(ChannelRefError, match="absolute HTTP"):
            ChannelResolver.from_text(
                f"""
[refs."{CHANNEL_REF}"]
url = "{url}"
""".lstrip(),
                root=tmp_path / f"bad-url-{index}",
            )


def test_resolver_returns_isolated_metadata_and_store_tables(tmp_path):
    resolver = ChannelResolver.from_text(
        f"""
[refs."{CHANNEL_REF}"]
store = ".lllm/network"

[refs."{CHANNEL_REF}".metadata]
owner = "demo"

[stores.default]
path = ".lllm/network"

[stores.default.metadata]
owner = "demo"
""".lstrip(),
        root=tmp_path,
    )

    binding = resolver.resolve(CHANNEL_REF)
    binding.metadata["metadata"]["owner"] = "changed"
    assert resolver.resolve(CHANNEL_REF).metadata == {"metadata": {"owner": "demo"}}

    stores = resolver.stores()
    stores["default"]["metadata"]["owner"] = "changed"
    assert resolver.stores() == {
        "default": {"path": ".lllm/network", "metadata": {"owner": "demo"}}
    }


def test_resolver_bind_accepts_and_isolates_mapping_metadata():
    headers = {"x-policy": "demo"}
    metadata = {"headers": MappingProxyType(headers)}
    resolver = ChannelResolver()
    resolver.bind(
        CHANNEL_REF,
        store=".lllm/network",
        metadata=MappingProxyType(metadata),
    )

    headers["x-policy"] = "changed"
    resolved = resolver.resolve(CHANNEL_REF)
    assert resolved.metadata == {"headers": {"x-policy": "demo"}}

    resolved.metadata["headers"]["x-policy"] = "mutated"
    assert resolver.resolve(CHANNEL_REF).metadata == {
        "headers": {"x-policy": "demo"}
    }


def test_resolver_rejects_non_table_metadata_config_values(tmp_path):
    with pytest.raises(ChannelRefError, match=r"\[refs\."):
        ChannelResolver.from_text(
            f"""
[refs."{CHANNEL_REF}"]
store = ".lllm/network"
metadata = "bad"
""".lstrip(),
            root=tmp_path / "bad-ref-metadata",
        )

    with pytest.raises(ChannelRefError, match=r"\[stores\.default\.metadata\]"):
        ChannelResolver.from_text(
            """
[stores.default]
path = ".lllm/network"
metadata = "bad"
""".lstrip(),
            root=tmp_path / "bad-store-metadata",
        )


def test_resolver_rejects_invalid_store_tables(tmp_path):
    with pytest.raises(ChannelRefError, match=r"\[stores\.default\]"):
        ChannelResolver.from_text(
            """
[stores]
default = ".lllm/network"
""".lstrip(),
            root=tmp_path / "bad-stores",
        )

    for index, text in enumerate(
        (
            """
[stores."bad/name"]
path = ".lllm/network"
""",
            """
[stores."bad%2Fname"]
path = ".lllm/network"
""",
            """
[stores."bad store"]
path = ".lllm/network"
""",
            """
[stores.default]
path = 123
""",
            """
[stores.default]
path = ""
""",
            """
[stores.default]
path = "bad path"
""",
        ),
        start=1,
    ):
        with pytest.raises(ChannelRefError):
            ChannelResolver.from_text(text.lstrip(), root=tmp_path / f"bad-store-{index}")


def test_resolver_store_lookup_rejects_malformed_names(tmp_path):
    resolver = ChannelResolver.from_text(
        """
[stores.default]
path = ".lllm/network"
""".lstrip(),
        root=tmp_path,
    )

    assert resolver.store("default") == {"path": ".lllm/network"}
    with pytest.raises(KeyError):
        resolver.store("missing")

    for invalid_name in (
        None,
        123,
        "",
        "   ",
        ".",
        "..",
        "bad/name",
        "bad name",
        "bad%2Fname",
    ):
        with pytest.raises(ChannelRefError, match="path-segment"):
            resolver.store(invalid_name)  # type: ignore[arg-type]


def test_path_value_rejects_malformed_config_paths(tmp_path):
    padded_root = tmp_path / "workspace"
    padded_config = tmp_path / "workspace" / ".lllm" / "config.toml"
    for value in ("", "   ", 123, b".lllm/config.toml", f" {padded_config} "):
        with pytest.raises(ValueError, match="config path|config root"):
            ChannelResolver.from_config(value)  # type: ignore[arg-type]

    for root in ("", "   ", 123, f" {padded_root} "):
        with pytest.raises(ValueError, match="config root"):
            ChannelResolver(root=root)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="config root"):
            ChannelResolver.from_text("", root=root)  # type: ignore[arg-type]

    assert not padded_root.exists()


def test_resolver_from_text_rejects_non_string_text(tmp_path):
    for index, value in enumerate((None, 123, b"[refs]\n"), start=1):
        root = tmp_path / f"workspace-{index}"
        with pytest.raises(ChannelRefError, match="config text"):
            ChannelResolver.from_text(value, root=root)  # type: ignore[arg-type]
        assert not root.exists()
