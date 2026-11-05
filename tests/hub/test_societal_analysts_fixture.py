from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from lllm import RemoteTactic, TacticResolver, as_tactic
from lllm.services import create_service_app
from lllm.network import Event, LocalStore

from lllm.hub import (
    LocalConfigResolver,
    LocalHub,
    import_entrypoint,
    load_manifest,
    validate_package,
)


SOURCE_CHANNELS = (
    "finance_ticks",
    "business_events",
    "economic_indicators",
    "geopolitics_news",
    "social_culture",
    "market_timeseries",
)
OUTPUT_CHANNELS = ("analyst_records", "aggregate_reports", "decision_signals")


def test_societal_analysts_fixture_lifecycle_and_local_execution(tmp_path):
    source_package = make_societal_source_package(tmp_path)
    tactic_package = make_societal_tactic_package(tmp_path)
    market_package = make_societal_market_package(tmp_path)
    sentinel_package = make_societal_sentinel_package(tmp_path)
    packages = (source_package, tactic_package, market_package, sentinel_package)
    hub = LocalHub(tmp_path / "hub")

    for package in packages:
        report = validate_package(package)
        assert report.ok, report.issues
        hub.publish(package)

    downloaded_root = tmp_path / "downloaded"
    downloaded_source = hub.download("society/source-channels", downloaded_root)
    downloaded_tactics = hub.download("society/analyst-tactics", downloaded_root)
    downloaded_market = hub.download("society/market-analyst", downloaded_root)
    downloaded_sentinel = hub.download("society/society-sentinel", downloaded_root)

    source_manifest = load_manifest(downloaded_source)
    tactics_manifest = load_manifest(downloaded_tactics)
    market_manifest = load_manifest(downloaded_market)
    sentinel_manifest = load_manifest(downloaded_sentinel)
    downstream = make_societal_downstream_package(
        tmp_path,
        source_manifest.ref("schema", "decision_signal"),
        sentinel_manifest.ref("channel", "decision_signals"),
    )
    downstream_report = validate_package(downstream)
    downstream_record = hub.publish(downstream)
    downloaded_downstream = hub.download("society/decision-review", tmp_path / "review")

    card = hub.card("society/society-sentinel")
    agent_card = hub.agent_card("society/society-sentinel")
    config = hub.config_template("society/society-sentinel")
    resolver = LocalConfigResolver.from_text(config, root=tmp_path / "workspace")

    analysis_events, report_event, decision_event = run_societal_fixture(
        tmp_path,
        downloaded_tactics,
        downloaded_sentinel,
        source_manifest,
        tactics_manifest,
        sentinel_manifest,
    )

    keys = {record.key for record in hub.list()}
    assert {
        "society/source-channels@0.1.0",
        "society/analyst-tactics@0.1.0",
        "society/market-analyst@0.1.0",
        "society/society-sentinel@0.1.0",
        "society/decision-review@0.1.0",
    } <= keys
    for downloaded in (
        downloaded_source,
        downloaded_tactics,
        downloaded_market,
        downloaded_sentinel,
        downloaded_downstream,
    ):
        assert validate_package(downloaded).ok
    assert downstream_report.ok
    assert downstream_record.validation.ok

    assert "lllm://society/society-sentinel/tactics/aggregate_report" in card
    assert "lllm://society/society-sentinel/channels/aggregate_reports" in card
    assert "lllm://society/society-sentinel/channels/decision_signals" in card
    assert "Endpoint: `POST /reports/aggregate`" in card
    assert "Endpoint: `POST /decisions/simulate`" in agent_card
    assert "Deterministic dummy data only" in agent_card
    assert '[refs."lllm://society/society-sentinel/services/sentinel_api"]' in config
    assert '[refs."lllm://society/society-sentinel/channels/decision_signals"]' in config
    assert resolver.resolve(
        "lllm://society/society-sentinel/services/sentinel_api"
    ).url == "http://127.0.0.1:8130"
    assert resolver.resolve(
        "lllm://society/society-sentinel/tactics/aggregate_report"
    ).url == "http://127.0.0.1:8130/tactics/aggregate_report"
    assert resolver.resolve(
        "lllm://society/society-sentinel/tactics/simulate_decision"
    ).url == "http://127.0.0.1:8130/tactics/simulate_decision"
    assert resolver.resolve(
        "lllm://society/society-sentinel/tactics/decision_pipeline"
    ).url == "http://127.0.0.1:8130/tactics/decision_pipeline"
    assert resolver.resolve(
        "lllm://society/society-sentinel/channels/aggregate_reports"
    ).store == ".lllm/network"
    assert resolver.service("sentinel_api") == {"port": 8130}
    assert resolver.store("default") == {"path": ".lllm/network"}

    assert len(analysis_events) == 3
    assert {event.payload["runtime"] for event in analysis_events} == {
        "python",
        "pydantic-ai",
        "native",
    }
    assert report_event.payload["domains"] == [
        "finance",
        "economics",
        "culture",
    ]
    assert report_event.payload["source_record_count"] == 3
    assert decision_event.payload == {
        "signal": "hold",
        "reason": "simulation only",
        "confidence": 0.0,
    }


def run_societal_fixture(
    tmp_path: Path,
    downloaded_tactics: Path,
    downloaded_sentinel: Path,
    source_manifest,
    tactics_manifest,
    sentinel_manifest,
):
    store = LocalStore(tmp_path / "society-store")
    for name in (*SOURCE_CHANNELS, *OUTPUT_CHANNELS):
        store.create_channel({"name": name, "form": "log"})

    raw_events = [
        store.append_event(
            Event(
                channel="finance_ticks",
                source="fixture",
                kind="finance",
                payload={"domain": "finance", "symbol": "ACME", "value": 101.2},
                correlation_id="society-run",
            )
        ),
        store.append_event(
            Event(
                channel="economic_indicators",
                source="fixture",
                kind="economics",
                payload={"domain": "economics", "indicator": "cpi", "value": 2.4},
                correlation_id="society-run",
            )
        ),
        store.append_event(
            Event(
                channel="social_culture",
                source="fixture",
                kind="culture",
                payload={"domain": "culture", "topic": "tooling", "sentiment": "steady"},
                correlation_id="society-run",
            )
        ),
    ]

    tactic_specs = (
        ("finance_baseline", "FinanceBaseline", [raw_events[0]]),
        ("tool_augmented", "ToolAugmentedAnalyst", [raw_events[1]]),
        ("native_dialog", "NativeDialogAnalyst", [raw_events[2]]),
    )
    tactic_resolver = TacticResolver()
    for name, class_name, selected_events in tactic_specs:
        cls = import_entrypoint(
            f"society_analysts.tactics:{class_name}",
            base_dir=downloaded_tactics,
        )
        tactic_resolver.register(
            tactics_manifest.ref("tactic", name),
            as_tactic(
                cls().run,
                name=name,
                input_type=dict,
                output_type=dict,
                package_ref=tactics_manifest.ref("tactic", name),
            ),
        )

    analysis_events = []
    for name, _class_name, selected_events in tactic_specs:
        payload = {
            "domain": selected_events[0].payload["domain"],
            "records": [event.payload for event in selected_events],
        }
        output = tactic_resolver.run(tactics_manifest.ref("tactic", name), payload)
        analysis_events.append(
            store.append_event(
                Event(
                    channel="analyst_records",
                    source=name,
                    kind="analysis",
                    payload=output,
                    correlation_id="society-run",
                    parent_ids=tuple(event.id for event in selected_events),
                )
            )
        )

    aggregate_cls = import_entrypoint(
        "society_sentinel.tactics:AggregateReport",
        base_dir=downloaded_sentinel,
    )
    decision_cls = import_entrypoint(
        "society_sentinel.tactics:DecisionSignal",
        base_dir=downloaded_sentinel,
    )
    pipeline_cls = import_entrypoint(
        "society_sentinel.tactics:DecisionPipeline",
        base_dir=downloaded_sentinel,
    )
    aggregate_ref = sentinel_manifest.ref("tactic", "aggregate_report")
    decision_ref = sentinel_manifest.ref("tactic", "simulate_decision")
    pipeline_ref = sentinel_manifest.ref("tactic", "decision_pipeline")
    aggregate_tactic = as_tactic(
        aggregate_cls().run,
        name="aggregate_report",
        input_type=dict,
        output_type=dict,
        package_ref=aggregate_ref,
    )
    decision_tactic = as_tactic(
        decision_cls().run,
        name="simulate_decision",
        input_type=dict,
        output_type=dict,
        package_ref=decision_ref,
    )
    tactic_resolver.register(aggregate_ref, aggregate_tactic)
    tactic_resolver.register(decision_ref, decision_tactic)
    pipeline_tactic = as_tactic(
        pipeline_cls(tactic_resolver, decision_ref).run,
        name="decision_pipeline",
        input_type=dict,
        output_type=dict,
        package_ref=pipeline_ref,
    )
    tactic_resolver.register(pipeline_ref, pipeline_tactic)

    sentinel_app = create_service_app(
        {
            "aggregate_report": aggregate_tactic,
            "simulate_decision": decision_tactic,
            "decision_pipeline": pipeline_tactic,
        },
        title="Society Sentinel Fixture",
    )
    transport = httpx.ASGITransport(app=sentinel_app)
    service_resolver = TacticResolver()
    service_resolver.register(
        aggregate_ref,
        RemoteTactic(
            "http://sentinel/tactics/aggregate_report",
            name="aggregate_report",
            input_type=dict,
            output_type=dict,
            async_transport=transport,
        ),
    )
    service_resolver.register(
        pipeline_ref,
        RemoteTactic(
            "http://sentinel/tactics/decision_pipeline",
            name="decision_pipeline",
            input_type=dict,
            output_type=dict,
            async_transport=transport,
        ),
    )

    report = asyncio.run(
        service_resolver.arun(
            aggregate_ref,
            {"records": [event.payload for event in analysis_events]},
        )
    )
    report_event = store.append_event(
        Event(
            channel="aggregate_reports",
            source="aggregate_report",
            kind="report",
            payload=report,
            correlation_id="society-run",
            parent_ids=tuple(event.id for event in analysis_events),
        )
    )
    decision = asyncio.run(
        service_resolver.arun(
            pipeline_ref,
            {"report": report_event.payload},
        )
    )
    decision_event = store.append_event(
        Event(
            channel="decision_signals",
            source="decision_pipeline",
            kind="decision",
            payload=decision,
            correlation_id="society-run",
            parent_ids=(report_event.id,),
        )
    )
    snapshot = store.put_snapshot(
        {
            "name": "latest_report",
            "channel": "aggregate_reports",
            "value": report_event.payload,
            "source_event_id": report_event.id,
            "metadata": {
                "source_package": source_manifest.package.identifier,
                "analyst_package": tactics_manifest.package.identifier,
            },
        }
    )

    assert store.get_snapshot("latest_report").source_event_id == snapshot.source_event_id
    assert store.query_events("decision_signals")[0].id == decision_event.id
    return analysis_events, report_event, decision_event


def make_societal_source_package(tmp_path: Path) -> Path:
    package = tmp_path / "source-channels"
    module = package / "society_sources"
    module.mkdir(parents=True)
    (package / "README.md").write_text(
        "# Source Channels\n\nDeterministic societal analyst source channels.\n",
        encoding="utf-8",
    )
    (module / "__init__.py").write_text("", encoding="utf-8")
    (module / "schemas.py").write_text(
        """
from pydantic import BaseModel


class SourceRecord(BaseModel):
    domain: str
    value: object | None = None


class AnalystRecord(BaseModel):
    domain: str
    runtime: str
    summary: str
    score: float


class AggregateReport(BaseModel):
    domains: list[str]
    source_record_count: int
    summary: str


class DecisionSignal(BaseModel):
    signal: str
    reason: str
    confidence: float
""".lstrip(),
        encoding="utf-8",
    )
    (package / "lllm.toml").write_text(
        """
[package]
manifest_version = "0.1"
org = "society"
name = "source-channels"
version = "0.1.0"
kind = "channel"
primary = "channels.finance_ticks"
description = "Deterministic societal analyst source and output channels."

[card]
summary = "Dummy source channels for societal analyst composition tests."
tags = ["societal-analysts", "sssn", "dummy-data"]
safety = "Deterministic dummy data only; no real recommendations."

[docs.readme]
path = "README.md"
title = "README"

[schemas.source_record]
entry = "society_sources.schemas:SourceRecord"

[schemas.analyst_record]
entry = "society_sources.schemas:AnalystRecord"

[schemas.aggregate_report]
entry = "society_sources.schemas:AggregateReport"

[schemas.decision_signal]
entry = "society_sources.schemas:DecisionSignal"

[channels.finance_ticks]
schema = "source_record"
form = "time-series"

[channels.business_events]
schema = "source_record"
form = "log"

[channels.economic_indicators]
schema = "source_record"
form = "time-series"

[channels.geopolitics_news]
schema = "source_record"
form = "feed"

[channels.social_culture]
schema = "source_record"
form = "feed"

[channels.market_timeseries]
schema = "source_record"
form = "time-series"

[channels.analyst_records]
schema = "analyst_record"
form = "log"

[channels.aggregate_reports]
schema = "aggregate_report"
form = "latest-state"

[channels.decision_signals]
schema = "decision_signal"
form = "log"

[snapshots.latest_report]
schema = "aggregate_report"
channel = "aggregate_reports"

[runs.local]
channels = [
  "finance_ticks",
  "business_events",
  "economic_indicators",
  "geopolitics_news",
  "social_culture",
  "market_timeseries",
  "analyst_records",
  "aggregate_reports",
  "decision_signals",
]
snapshots = ["latest_report"]
""".lstrip(),
        encoding="utf-8",
    )
    return package


def make_societal_tactic_package(tmp_path: Path) -> Path:
    package = tmp_path / "analyst-tactics"
    module = package / "society_analysts"
    module.mkdir(parents=True)
    (package / "README.md").write_text(
        "# Analyst Tactics\n\nDeterministic dummy analyst tactics.\n",
        encoding="utf-8",
    )
    (module / "__init__.py").write_text("", encoding="utf-8")
    (module / "schemas.py").write_text(
        """
from pydantic import BaseModel


class AnalystInput(BaseModel):
    domain: str
    records: list[dict]


class AnalystOutput(BaseModel):
    domain: str
    runtime: str
    summary: str
    score: float
""".lstrip(),
        encoding="utf-8",
    )
    (module / "tactics.py").write_text(
        """
class FinanceBaseline:
    def run(self, input_value, *, context=None):
        records = input_value["records"]
        return {
            "domain": input_value["domain"],
            "runtime": "python",
            "summary": f"baseline:{records[0]['domain']}",
            "score": float(len(records)),
        }


class ToolAugmentedAnalyst:
    def run(self, input_value, *, context=None):
        records = input_value["records"]
        score = self.score_indicator(records[0])
        return {
            "domain": input_value["domain"],
            "runtime": "pydantic-ai",
            "summary": f"tool-score:{score}",
            "score": score,
        }

    def score_indicator(self, record):
        return float(record.get("value", 0)) / 10.0


class NativeDialogAnalyst:
    def run(self, input_value, *, context=None):
        records = input_value["records"]
        return {
            "domain": input_value["domain"],
            "runtime": "native",
            "summary": f"dialog-reviewed:{records[0]['domain']}",
            "score": 0.0,
        }
""".lstrip(),
        encoding="utf-8",
    )
    (module / "services.py").write_text(
        """
def create_finance_app():
    return {"service": "finance"}


def create_tool_app():
    return {"service": "tool"}


def create_native_app():
    return {"service": "native"}
""".lstrip(),
        encoding="utf-8",
    )
    (package / "lllm.toml").write_text(
        """
[package]
manifest_version = "0.1"
org = "society"
name = "analyst-tactics"
version = "0.1.0"
kind = "tactic"
primary = "tactics.finance_baseline"
description = "Deterministic LLLM analyst tactics for societal fixture tests."

[card]
summary = "Analyst tactics spanning Python, Pydantic-AI-style, and native-style runtimes."
tags = ["societal-analysts", "lllm", "dummy-data"]
safety = "Simulation only; outputs are fixture signals, not advice."

[docs.readme]
path = "README.md"
title = "README"

[schemas.analyst_input]
entry = "society_analysts.schemas:AnalystInput"

[schemas.analyst_output]
entry = "society_analysts.schemas:AnalystOutput"

[tactics.finance_baseline]
entry = "society_analysts.tactics:FinanceBaseline"
input = "analyst_input"
output = "analyst_output"
runtime = "python"
description = "Plain Python deterministic finance baseline."

[tactics.tool_augmented]
entry = "society_analysts.tactics:ToolAugmentedAnalyst"
input = "analyst_input"
output = "analyst_output"
runtime = "pydantic-ai"
description = "Pydantic-AI-style structured analyst with deterministic tool metadata."

[tactics.tool_augmented.metadata]
tools = ["score_indicator"]

[tactics.native_dialog]
entry = "society_analysts.tactics:NativeDialogAnalyst"
input = "analyst_input"
output = "analyst_output"
runtime = "native"
description = "Native-runtime-style prompt/dialog analyst fixture."

[services.finance_api]
entry = "society_analysts.services:create_finance_app"
tactic = "finance_baseline"
transport = "fastapi"

[services.finance_api.metadata]
port = 8110

[services.tool_api]
entry = "society_analysts.services:create_tool_app"
tactic = "tool_augmented"
transport = "fastapi"

[services.tool_api.metadata]
port = 8111

[services.native_api]
entry = "society_analysts.services:create_native_app"
tactic = "native_dialog"
transport = "fastapi"

[services.native_api.metadata]
port = 8112

[runs.local]
services = ["finance_api", "tool_api", "native_api"]
tactics = ["finance_baseline", "tool_augmented", "native_dialog"]
""".lstrip(),
        encoding="utf-8",
    )
    return package


def make_societal_market_package(tmp_path: Path) -> Path:
    package = tmp_path / "market-analyst"
    module = package / "market_analyst"
    module.mkdir(parents=True)
    (package / "README.md").write_text(
        "# Market Analyst\n\nCombined source-channel and analyst package.\n",
        encoding="utf-8",
    )
    (module / "__init__.py").write_text("", encoding="utf-8")
    (module / "tactics.py").write_text(
        """
class MarketBrief:
    def run(self, input_value, *, context=None):
        return {
            "domain": "market",
            "runtime": "python",
            "summary": "market fixture",
            "score": 1.0,
        }
""".lstrip(),
        encoding="utf-8",
    )
    (module / "services.py").write_text(
        """
def create_app():
    return {"service": "market"}
""".lstrip(),
        encoding="utf-8",
    )
    (package / "lllm.toml").write_text(
        """
[package]
manifest_version = "0.1"
org = "society"
name = "market-analyst"
version = "0.1.0"
kind = "app"
primary = "services.market_api"
description = "Combined LLLM and LLLM network package for market fixture data."

[card]
summary = "Combined analyst package using source channels and one tactic."
tags = ["societal-analysts", "combined"]
safety = "Fixture package only."

[docs.readme]
path = "README.md"
title = "README"

[tactics.market_brief]
entry = "market_analyst.tactics:MarketBrief"
input = "lllm://society/source-channels/schemas/source_record"
output = "lllm://society/source-channels/schemas/analyst_record"
runtime = "python"

[[tactics.market_brief.endpoints]]
name = "market_brief"
method = "POST"
path = "/market/brief"
mode = "run"

[channels.finance_ticks]
schema = "lllm://society/source-channels/schemas/source_record"
form = "time-series"

[channels.analyst_records]
schema = "lllm://society/source-channels/schemas/analyst_record"
form = "log"

[services.market_api]
entry = "market_analyst.services:create_app"
tactic = "market_brief"
subscribes = ["finance_ticks"]
publishes = ["analyst_records"]

[services.market_api.metadata]
port = 8120

[runs.local]
services = ["market_api"]
channels = ["finance_ticks", "analyst_records"]
""".lstrip(),
        encoding="utf-8",
    )
    return package


def make_societal_sentinel_package(tmp_path: Path) -> Path:
    package = tmp_path / "society-sentinel"
    module = package / "society_sentinel"
    module.mkdir(parents=True)
    (package / "README.md").write_text(
        "# Society Sentinel\n\nAggregates dummy analyst records and emits a simulated decision signal.\n",
        encoding="utf-8",
    )
    (module / "__init__.py").write_text("", encoding="utf-8")
    (module / "tactics.py").write_text(
        """
class AggregateReport:
    def run(self, input_value, *, context=None):
        records = input_value["records"]
        return {
            "domains": [record["domain"] for record in records],
            "source_record_count": len(records),
            "summary": " | ".join(record["summary"] for record in records),
        }


class DecisionSignal:
    def run(self, input_value, *, context=None):
        return {
            "signal": "hold",
            "reason": "simulation only",
            "confidence": 0.0,
        }


class DecisionPipeline:
    def __init__(self, resolver, decision_ref):
        self.resolver = resolver
        self.decision_ref = decision_ref

    def run(self, input_value, *, context=None):
        return self.resolver.run(self.decision_ref, input_value)
""".lstrip(),
        encoding="utf-8",
    )
    (module / "services.py").write_text(
        """
def create_app():
    return {"service": "sentinel"}
""".lstrip(),
        encoding="utf-8",
    )
    (package / "lllm.toml").write_text(
        """
[package]
manifest_version = "0.1"
org = "society"
name = "society-sentinel"
version = "0.1.0"
kind = "app"
primary = "services.sentinel_api"
description = "Aggregate report and simulated decision-signal package."

[card]
summary = "Societal analyst aggregate and decision-signal package."
tags = ["societal-analysts", "aggregate", "decision-signal"]
safety = "Deterministic dummy data only; no real recommendations."
latency = "Local fixture execution only."
suggested_commands = ["python examples/run_society_fixture.py"]

[docs.readme]
path = "README.md"
title = "README"

[tactics.aggregate_report]
entry = "society_sentinel.tactics:AggregateReport"
input = "lllm://society/source-channels/schemas/analyst_record"
output = "lllm://society/source-channels/schemas/aggregate_report"

[[tactics.aggregate_report.endpoints]]
name = "aggregate_report"
method = "POST"
path = "/reports/aggregate"
mode = "run"

[tactics.simulate_decision]
entry = "society_sentinel.tactics:DecisionSignal"
input = "lllm://society/source-channels/schemas/aggregate_report"
output = "lllm://society/source-channels/schemas/decision_signal"

[[tactics.simulate_decision.endpoints]]
name = "simulate_decision"
method = "POST"
path = "/decisions/simulate"
mode = "run"

[tactics.decision_pipeline]
entry = "society_sentinel.tactics:DecisionPipeline"
input = "lllm://society/source-channels/schemas/aggregate_report"
output = "lllm://society/source-channels/schemas/decision_signal"
description = "Pipeline tactic that calls simulate_decision through a resolved psi ref."

[channels.analyst_records]
schema = "lllm://society/source-channels/schemas/analyst_record"
form = "log"

[channels.aggregate_reports]
schema = "lllm://society/source-channels/schemas/aggregate_report"
form = "latest-state"

[channels.decision_signals]
schema = "lllm://society/source-channels/schemas/decision_signal"
form = "log"

[snapshots.latest_report]
schema = "lllm://society/source-channels/schemas/aggregate_report"
channel = "aggregate_reports"

[services.sentinel_api]
entry = "society_sentinel.services:create_app"
tactic = "aggregate_report"
tactics = ["aggregate_report", "simulate_decision", "decision_pipeline"]
subscribes = ["analyst_records"]
publishes = ["aggregate_reports", "decision_signals"]

[services.sentinel_api.metadata]
port = 8130

[runs.local]
services = ["sentinel_api"]
tactics = ["aggregate_report", "simulate_decision", "decision_pipeline"]
channels = ["analyst_records", "aggregate_reports", "decision_signals"]
snapshots = ["latest_report"]
""".lstrip(),
        encoding="utf-8",
    )
    return package


def make_societal_downstream_package(
    tmp_path: Path,
    decision_schema_ref: str,
    decision_channel_ref: str,
) -> Path:
    package = tmp_path / "decision-review"
    module = package / "decision_review"
    module.mkdir(parents=True)
    (package / "README.md").write_text(
        "# Decision Review\n\nComposes downloaded societal analyst refs.\n",
        encoding="utf-8",
    )
    (module / "__init__.py").write_text("", encoding="utf-8")
    (module / "services.py").write_text(
        """
def create_app():
    return {"service": "review"}
""".lstrip(),
        encoding="utf-8",
    )
    (package / "lllm.toml").write_text(
        f"""
[package]
manifest_version = "0.1"
org = "society"
name = "decision-review"
version = "0.1.0"
kind = "app"
primary = "services.review_api"
description = "Downstream package composed from downloaded societal analyst refs."

[card]
summary = "Downstream review package for simulated decision signals."
tags = ["societal-analysts", "downstream"]
safety = "Reviews fixture signals only."

[docs.readme]
path = "README.md"
title = "README"

[channels.decision_signals]
schema = "{decision_schema_ref}"
form = "log"

[services.review_api]
entry = "decision_review.services:create_app"
transport = "fastapi"
subscribes = ["decision_signals"]

[services.review_api.metadata]
source_channel = "{decision_channel_ref}"
port = 8140

[runs.local]
services = ["review_api"]
channels = ["decision_signals"]
""".lstrip(),
        encoding="utf-8",
    )
    return package
