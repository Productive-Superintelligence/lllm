import asyncio

import httpx

from lllm import RemoteTactic, Tactic, TacticResolver
from lllm.services import create_tactic_app
from pydantic import BaseModel
from lllm.network import AsyncRemoteStore, LocalStore
from lllm.network.server import create_app as create_network_app

from lllm.hub import LocalConfigResolver, LocalHub, load_manifest, validate_package
from test_lifecycle import make_combined_package


TACTIC_REF = "lllm://demo/http-combo/tactics/analyze"
RAW_CHANNEL_REF = "lllm://demo/http-combo/channels/raw"
ANALYSIS_CHANNEL_REF = "lllm://demo/http-combo/channels/analysis"


class AnalyzeInput(BaseModel):
    text: str


class AnalyzeOutput(BaseModel):
    summary: str


class AnalyzeTactic(Tactic[AnalyzeInput, AnalyzeOutput]):
    name = "analyze"
    input_type = AnalyzeInput
    output_type = AnalyzeOutput

    def _run(self, input_value, *, context=None):
        return AnalyzeOutput(summary=input_value.text.upper())


def test_http_refs_call_tactic_service_and_network_channel_service(tmp_path):
    config = LocalConfigResolver.from_text(
        f"""
[refs."{TACTIC_REF}"]
url = "http://lllm/run"

[refs."{RAW_CHANNEL_REF}"]
url = "http://sssn"

[refs."{ANALYSIS_CHANNEL_REF}"]
url = "http://sssn"
""".lstrip(),
        root=tmp_path / "workspace",
    )
    lllm_app = create_tactic_app(AnalyzeTactic())
    network_app = create_network_app(LocalStore(tmp_path / "store"))

    async def run():
        tactic_resolver = TacticResolver()
        tactic_resolver.register(
            TACTIC_REF,
            RemoteTactic(
                config.resolve(TACTIC_REF).url,
                name="analyze",
                input_type=AnalyzeInput,
                output_type=AnalyzeOutput,
                async_transport=httpx.ASGITransport(app=lllm_app),
            ),
        )
        raw_client = AsyncRemoteStore(
            config.resolve(RAW_CHANNEL_REF).url,
            transport=httpx.ASGITransport(app=network_app),
        )
        analysis_client = AsyncRemoteStore(
            config.resolve(ANALYSIS_CHANNEL_REF).url,
            transport=httpx.ASGITransport(app=network_app),
        )
        await raw_client.create_channel({"name": "raw"})
        await analysis_client.create_channel({"name": "analysis"})
        raw = await raw_client.append_event(
            {
                "channel": "raw",
                "payload": {"text": "hello"},
                "correlation_id": "run-1",
            }
        )
        output = await tactic_resolver.arun(TACTIC_REF, raw.payload)
        analysis = await analysis_client.append_event(
            {
                "channel": "analysis",
                "kind": "analysis",
                "payload": output.model_dump(mode="json"),
                "correlation_id": raw.correlation_id,
                "parent_ids": [raw.id],
            }
        )
        loaded = await analysis_client.query_events("analysis")
        return output, raw, analysis, loaded

    output, raw, analysis, loaded = asyncio.run(run())

    assert output == AnalyzeOutput(summary="HELLO")
    assert analysis.parent_ids == (raw.id,)
    assert loaded[0].id == analysis.id
    assert loaded[0].payload == {"summary": "HELLO"}


def test_downloaded_package_refs_bind_to_http_services(tmp_path):
    package = make_combined_package(tmp_path)
    hub = LocalHub(tmp_path / "hub")
    record = hub.publish(package)
    downloaded = hub.download("demo/combo", tmp_path / "downloaded")
    manifest = load_manifest(downloaded)

    tactic_ref = manifest.ref("tactic", "analyze")
    events_ref = manifest.ref("channel", "events")
    analysis_ref = manifest.ref("channel", "analysis")
    config = LocalConfigResolver.from_text(
        f"""
[refs."{tactic_ref}"]
url = "http://lllm/tactics/analyze"

[refs."{events_ref}"]
url = "http://sssn"

[refs."{analysis_ref}"]
url = "http://sssn"
""".lstrip(),
        root=tmp_path / "workspace",
    )
    lllm_app = create_tactic_app(AnalyzeTactic())
    network_app = create_network_app(LocalStore(tmp_path / "store"))

    async def run():
        tactic_resolver = TacticResolver()
        tactic_resolver.register(
            tactic_ref,
            RemoteTactic(
                config.resolve(tactic_ref).url,
                name="analyze",
                input_type=AnalyzeInput,
                output_type=AnalyzeOutput,
                async_transport=httpx.ASGITransport(app=lllm_app),
            ),
        )
        events_client = AsyncRemoteStore(
            config.resolve(events_ref).url,
            transport=httpx.ASGITransport(app=network_app),
        )
        analysis_client = AsyncRemoteStore(
            config.resolve(analysis_ref).url,
            transport=httpx.ASGITransport(app=network_app),
        )
        await events_client.create_channel({"name": "events"})
        await analysis_client.create_channel({"name": "analysis"})
        raw = await events_client.append_event(
            {
                "channel": "events",
                "payload": {"text": "downloaded"},
                "correlation_id": "downloaded-run",
            }
        )
        output = await tactic_resolver.arun(tactic_ref, raw.payload)
        analysis = await analysis_client.append_event(
            {
                "channel": "analysis",
                "kind": "analysis",
                "payload": output.model_dump(mode="json"),
                "correlation_id": raw.correlation_id,
                "parent_ids": [raw.id],
            }
        )
        loaded = await analysis_client.query_events("analysis")
        return output, raw, analysis, loaded

    output, raw, analysis, loaded = asyncio.run(run())

    assert record.validation.ok
    assert validate_package(downloaded).ok
    assert sorted(config.refs()) == [analysis_ref, events_ref, tactic_ref]
    assert output == AnalyzeOutput(summary="DOWNLOADED")
    assert analysis.parent_ids == (raw.id,)
    assert loaded[0].payload == {"summary": "DOWNLOADED"}
