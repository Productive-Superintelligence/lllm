"""The project loads and builds its tactic without calling a model."""

from pathlib import Path

from lllm import Runtime, build_tactic, load_package, register_invoker, resolve_config


class NoModelInvoker:
    def __init__(self, config):
        self.config = config

    def call(self, *args, **kwargs):
        raise AssertionError("The smoke test must not call a model.")


def test_project_loads_without_calling_a_model():
    root = Path(__file__).resolve().parents[1]
    runtime = Runtime()
    load_package(root / "lllm.toml", runtime=runtime)

    config = resolve_config("{{package_name}}:default", runtime=runtime)
    config["invoker"] = "no-model"
    register_invoker("no-model", NoModelInvoker, overwrite=True)
    tactic = build_tactic(config, runtime=runtime)

    expected_agents = {agent["name"] for agent in config["agent_configs"]}
    assert set(tactic.agents) == expected_agents
