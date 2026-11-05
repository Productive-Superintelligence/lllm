from lllm import TacticResolver, as_tactic
from lllm.network import Event, LocalStore

from lllm.hub import LocalConfigResolver, LocalHub, load_manifest, validate_package
from test_lifecycle import make_combined_package


def test_downloaded_package_composes_tactic_and_network_channels(tmp_path):
    package = make_combined_package(tmp_path)
    hub = LocalHub(tmp_path / "hub")
    record = hub.publish(package)
    downloaded = hub.download("demo/combo", tmp_path / "downloaded")
    manifest = load_manifest(downloaded)
    config = LocalConfigResolver.from_text(
        hub.config_template("demo/combo"),
        root=tmp_path / "workspace",
    )

    def analyze(payload: dict) -> dict:
        return {"summary": payload["text"].upper()}

    tactic_ref = manifest.ref("tactic", "analyze")
    analyzer_ref = manifest.ref("service", "analyzer")
    events_ref = manifest.ref("channel", "events")
    analysis_ref = manifest.ref("channel", "analysis")
    config_tactic = TacticResolver.from_config(tmp_path / "workspace").resolve(
        tactic_ref
    )
    config_tactic_metadata = config_tactic.info().metadata

    tactic_resolver = TacticResolver()
    tactic_resolver.register(tactic_ref, as_tactic(analyze, input_type=dict, output_type=dict))

    store_path = tmp_path / config.resolve(events_ref).store
    store = LocalStore(store_path)
    store.create_channel({"name": "events"})
    store.create_channel({"name": "analysis"})
    raw = store.append_event(
        Event(
            channel="events",
            payload={"text": "hello"},
            correlation_id="demo-run",
        )
    )
    output = tactic_resolver.run(tactic_ref, raw.payload)
    analysis = store.append_event(
        Event(
            channel="analysis",
            kind="analysis",
            payload=output,
            correlation_id=raw.correlation_id,
            parent_ids=(raw.id,),
        )
    )

    assert record.validation.ok
    assert validate_package(downloaded).ok
    assert config_tactic_metadata["input"] == "analysis_input"
    assert config_tactic_metadata["output"] == "analysis_output"
    assert config_tactic_metadata["runtime"] == "python"
    assert config.resolve(analyzer_ref).metadata["subscribes"] == ["events"]
    assert config.resolve(analyzer_ref).metadata["publishes"] == ["analysis"]
    assert config.resolve(tactic_ref).url.endswith("/tactics/analyze")
    assert config.resolve(analysis_ref).store == ".lllm/network"
    assert output == {"summary": "HELLO"}
    assert store.query_events("analysis")[0].id == analysis.id
    assert analysis.parent_ids == (raw.id,)
