import os
from pathlib import Path

from lllm.hub import LocalHub, init_package, validate_package


def run_workflow(root: str | Path):
    """Run the local package lifecycle without a hosted hub."""

    root = Path(_path_value(root, "workflow root"))
    package = root / "demo-package"
    hub_root = root / ".lllm/hub"
    download_root = root / "downloaded"

    manifest_path = init_package(
        package,
        org="demo",
        name="echo",
        kind="tactic",
        force=True,
    )
    report = validate_package(package)
    hub = LocalHub(hub_root)
    record = hub.publish(package)
    downloaded = hub.download("demo/echo", download_root)
    card = hub.card("demo/echo")
    agent_card = hub.agent_card("demo/echo")
    config = hub.config_template("demo/echo")

    return {
        "manifest_path": manifest_path,
        "report": report,
        "record": record,
        "listed": hub.list(),
        "downloaded": downloaded,
        "card": card,
        "agent_card": agent_card,
        "config": config,
    }


def _path_value(value, label: str) -> str:
    try:
        text = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-empty path string") from exc
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    return text
