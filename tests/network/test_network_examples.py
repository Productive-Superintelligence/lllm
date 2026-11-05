import importlib.util
import re
from pathlib import Path

import pytest

from lllm.network import LocalStore


ROOT = Path(__file__).resolve().parents[2]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_documented_example_paths_exist():
    pattern = re.compile(r"`(examples/[^`]+)`")
    text_paths = [ROOT / "README.md"]
    text_paths.extend((ROOT / "docs").rglob("*.md"))

    missing = []
    for path in sorted(text_paths):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            example_path = ROOT / match.group(1)
            if not example_path.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {match.group(1)}")

    assert missing == []


def test_channel_processor_example_reads_and_writes_channels(tmp_path):
    module = load_module(
        ROOT / "examples" / "network" / "channel_processor" / "processor.py",
        "channel_processor",
    )
    store = LocalStore(tmp_path / "store")
    store.create_channel({"name": "raw"})
    store.create_channel({"name": "analysis"})
    sub = store.create_subscription("raw")
    raw = store.append_event(
        {
            "channel": "raw",
            "payload": {"frame": 1},
            "correlation_id": "episode-1",
        }
    )

    outputs = module.process_once(store, subscription_id=sub.id)

    assert len(outputs) == 1
    assert outputs[0].channel == "analysis"
    assert outputs[0].parent_ids == (raw.id,)
    assert outputs[0].correlation_id == "episode-1"
    assert store.query_events("analysis")[0].payload["source_event_id"] == raw.id


def test_first_channel_example_creates_and_reads_event(tmp_path):
    module = load_module(
        ROOT / "examples" / "network" / "first_channel" / "workflow.py",
        "first_channel_workflow",
    )
    store = LocalStore(tmp_path / "store")

    result = module.run_workflow(store)

    assert result["channel"].name == "events"
    assert result["event"].channel == "events"
    assert result["event"].payload == {"text": "hello"}
    assert result["events"] == (result["event"],)
    assert store.query_events("events", after_cursor=0) == result["events"]


def test_hub_manifest_example_builds_channel_manifest():
    module = load_module(
        ROOT / "examples" / "network" / "hub_manifest" / "package_manifest.py",
        "hub_manifest",
    )

    manifest = module.build_manifest()

    assert manifest["package"]["primary"] == "channels.raw"
    assert sorted(manifest["channels"]) == ["analysis", "raw"]
    assert sorted(manifest["snapshots"]) == ["latest_analysis"]
    assert manifest["channels"]["raw"]["schema"] == "raw_event"
    assert manifest["channels"]["raw"]["endpoints"] == [
        {
            "name": "channel_tail",
            "method": "GET",
            "path": "/channels/{name}/tail",
            "scope": "channel",
            "description": "Return the most recent events for a channel.",
            "tags": ["channels"],
        }
    ]
    assert manifest["snapshots"]["latest_analysis"]["schema"] == "analysis_event"
    assert manifest["snapshots"]["latest_analysis"]["channel"] == "analysis"
    assert manifest["snapshots"]["latest_analysis"]["endpoints"] == [
        {
            "name": "latest_analysis",
            "method": "GET",
            "path": "/snapshots/latest-analysis",
            "scope": "snapshot",
            "description": "Return the latest derived analysis state.",
            "tags": ["snapshots"],
        }
    ]
    assert manifest["runs"]["local"]["channels"] == ["raw", "analysis"]
    assert manifest["runs"]["local"]["snapshots"] == ["latest_analysis"]


def test_artifact_snapshot_example_writes_artifact_and_latest_state(tmp_path):
    module = load_module(
        ROOT / "examples" / "network" / "artifact_snapshot" / "workflow.py",
        "artifact_snapshot_workflow",
    )
    store = LocalStore(tmp_path / "store")

    result = module.run_workflow(store)

    assert result["artifact"].event_ids == (result["event"].id,)
    assert result["artifact"].channel == "frames"
    assert result["artifact_data"] == b"frame-one"
    assert result["snapshot"].name == "latest"
    assert result["snapshot"].source_event_id == result["event"].id
    assert result["snapshot"].value["artifact_id"] == result["artifact"].id
    assert store.get_snapshot("latest").value == result["snapshot"].value


def test_lllm_tactic_processor_example_composes_tactic_and_channels(tmp_path):
    pytest.importorskip("lllm")
    module = load_module(
        ROOT / "examples" / "network" / "lllm_tactic_processor" / "workflow.py",
        "lllm_tactic_processor_workflow",
    )
    store = LocalStore(tmp_path / "store")

    result = module.run_workflow(store)
    analysis = result["analysis"]

    assert result["raw"].channel == "raw"
    assert len(analysis) == 1
    assert analysis[0].channel == "analysis"
    assert analysis[0].parent_ids == (result["raw"].id,)
    assert analysis[0].correlation_id == "episode-1"
    assert analysis[0].payload == {"summary": "HELLO FROM THE NETWORK", "length": 22}
    assert result["tactic_info"].name == "analyze_message"
