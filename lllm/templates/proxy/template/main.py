from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

from lllm import ProxyManager, build_tactic, load_runtime, resolve_config  # noqa: E402


def print_catalog() -> None:
    load_runtime(str(PROJECT_ROOT / "lllm.toml"))
    manager = ProxyManager(activate_proxies=["sample"])
    print(manager.retrieve_api_docs())


def create_tactic():
    """Build the project's tactic from configs/default.yaml.

    `lllm serve .` and the hub use this as the package's primary tactic.
    """

    load_runtime(str(PROJECT_ROOT / "lllm.toml"))
    return build_tactic(resolve_config("default"))


def main() -> None:
    if "--catalog" in sys.argv:
        print_catalog()
        return

    task = " ".join(arg for arg in sys.argv[1:] if arg != "--catalog").strip()
    task = task or "Summarize what data is available from the sample proxy."
    tactic = create_tactic()
    print(tactic(task))


if __name__ == "__main__":
    main()
