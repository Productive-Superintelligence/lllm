import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_echo_service_example_builds_tactic():
    module = load_module(ROOT / "examples" / "echo_service" / "tactics.py", "echo_tactics")
    tactic = module.build_tactic()

    assert tactic.run({"text": "hello"}).text == "HELLO"


def test_pydantic_ai_fake_agent_example_builds_tactic():
    module = load_module(
        ROOT / "examples" / "pydantic_ai_tactic" / "fake_agent.py",
        "fake_agent",
    )
    tactic = module.build_tactic()

    assert tactic.run("hello") == "HELLO"


def test_native_dialog_example_builds_transcript_and_retry_branch():
    module = load_module(
        ROOT / "examples" / "native_dialog" / "demo.py",
        "native_dialog_demo",
    )

    dialog = module.build_dialog()
    retry = module.build_retry_dialog()

    assert dialog.owner == "planner"
    assert dialog.head.content == "You are a careful planning assistant."
    assert dialog.tail.name == "label"
    assert dialog.tail.metadata["function_call"]["result"] == "Native Core"
    assert retry.parent is not None
    assert retry.depth == 1
