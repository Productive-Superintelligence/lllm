from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

from lllm import build_tactic, load_runtime, resolve_config  # noqa: E402


def create_tactic():
    load_runtime(str(PROJECT_ROOT / "lllm.toml"))
    config = resolve_config("default")
    return build_tactic(config)


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or "Describe this service in one sentence."
    tactic = create_tactic()
    print(tactic(task))


if __name__ == "__main__":
    main()
