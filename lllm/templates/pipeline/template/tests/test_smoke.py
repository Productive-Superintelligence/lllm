from pathlib import Path

from lllm import build_tactic, resolve_config
from lllm.core.config import load_package
from lllm.core.const import APITypes
from lllm.core.dialog import Dialog
from lllm.core.runtime import Runtime
from lllm.invokers import register_invoker
from lllm.invokers.base import BaseInvoker


class DummyInvoker(BaseInvoker):
    def call(
        self,
        dialog: Dialog,
        model: str,
        model_args=None,
        parser_args=None,
        responder: str = "assistant",
        metadata=None,
        api_type: APITypes = APITypes.COMPLETION,
        stream_handler=None,
    ):
        raise AssertionError("Smoke test should not call the LLM.")


def test_project_loads_without_calling_llm():
    root = Path(__file__).resolve().parents[1]
    runtime = Runtime()
    load_package(root / "lllm.toml", runtime=runtime)

    assert runtime.has("{{package_name}}.prompts:pipeline/planner/system")
    assert runtime.has("{{package_name}}.prompts:pipeline/tasks/plan")
    assert runtime.has("{{package_name}}.configs:default")
    assert runtime.has("{{package_name}}.tactics:pipeline")

    config = resolve_config("{{package_name}}:default", runtime=runtime)
    config["invoker"] = "dummy"
    register_invoker("dummy", lambda cfg: DummyInvoker(cfg), overwrite=True)

    tactic = build_tactic(config, runtime=runtime)
    assert tactic.name == "pipeline"
    assert set(tactic.agents) == {"planner", "writer", "reviewer"}
