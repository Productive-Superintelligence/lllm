from typing import Any

from lllm import CallContext
from lllm.runtimes import PydanticAITactic


class Result:
    def __init__(self, output: dict[str, Any]):
        self.output = output


class ConfiguredAgent:
    name = "configured-agent"
    output_type = dict

    def __init__(self, *, model: str, instrumented: bool = False):
        self.model = model
        self.instrumented = instrumented
        self.seen: dict[str, Any] = {}

    def run_sync(self, task, *, metadata=None, **kwargs):
        self.seen = {
            "task": task,
            "metadata": dict(metadata or {}),
            "kwargs": dict(kwargs),
            "model": self.model,
            "instrumented": self.instrumented,
        }
        return Result(
            {
                "task": task,
                "model": self.model,
                "trace_id": self.seen["metadata"].get("lllm_trace_id"),
                "durable_run_id": kwargs.get("durable_run_id"),
                "graph_node": kwargs.get("graph_node"),
            }
        )


def build_tactic(agent: ConfiguredAgent | None = None) -> PydanticAITactic:
    return PydanticAITactic(
        agent or ConfiguredAgent(model="fake-provider:small", instrumented=True),
        input_type=str,
        output_type=dict,
        run_kwargs={
            "model_settings": {"temperature": 0},
            "deps": {"vector_store": "fake"},
            "eval_hook": "offline-score",
            "tool_approval": "runtime-owned",
        },
    )


def run_demo() -> tuple[dict[str, Any], ConfiguredAgent]:
    agent = ConfiguredAgent(model="fake-provider:small", instrumented=True)
    tactic = build_tactic(agent)
    output = tactic.run(
        "summarize package refs",
        context=CallContext(trace_id="trace-surrounding", metadata={"caller": "demo"}),
        durable_run_id="durable-1",
        graph_node="plan.step",
    )
    return output, agent
