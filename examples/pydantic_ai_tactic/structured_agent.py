import json
from typing import Any

from pydantic import BaseModel

from lllm.runtimes import PydanticAITactic, tactic_as_tool


class BriefInput(BaseModel):
    topic: str
    audience: str = "engineers"


class BriefOutput(BaseModel):
    title: str
    bullets: list[str]
    trace_id: str | None = None


class Result:
    def __init__(self, output: BriefOutput):
        self.output = output


class StreamResult:
    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def stream_output(self):
        yield from self._chunks


class FakeStructuredAgent:
    name = "brief-writer"
    output_type = BriefOutput

    def __init__(self):
        self.last_task: dict[str, Any] | None = None
        self.last_metadata: dict[str, Any] = {}

    def run_sync(self, task, *, metadata=None, **kwargs):
        data = self._decode_task(task)
        self.last_task = data
        self.last_metadata = dict(metadata or {})
        return Result(
            BriefOutput(
                title=f"{data['topic'].title()} for {data['audience']}",
                bullets=[
                    f"Define {data['topic']}",
                    f"Make it useful for {data['audience']}",
                ],
                trace_id=self.last_metadata.get("lllm_trace_id"),
            )
        )

    def run_stream_sync(self, task, *, metadata=None, **kwargs):
        data = self._decode_task(task)
        return StreamResult(
            [
                f"topic:{data['topic']}",
                f"audience:{data['audience']}",
            ]
        )

    def output_json_schema(self):
        return BriefOutput.model_json_schema()

    def _decode_task(self, task) -> dict[str, Any]:
        if isinstance(task, str):
            return json.loads(task)
        if isinstance(task, BaseModel):
            return task.model_dump(mode="json")
        return dict(task)


def build_tactic(agent: FakeStructuredAgent | None = None) -> PydanticAITactic:
    return PydanticAITactic(
        agent or FakeStructuredAgent(),
        input_type=BriefInput,
        output_type=BriefOutput,
        run_kwargs={"temperature": 0},
    )


def build_tool():
    return tactic_as_tool(
        build_tactic(),
        name="write brief",
        parameter_mode="kwargs",
    )
