from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

from lllm import build_tactic, load_runtime, resolve_config  # noqa: E402


def main() -> None:
    topic = " ".join(sys.argv[1:]).strip() or "agent evaluation methods"
    load_runtime(str(PROJECT_ROOT / "lllm.toml"))
    config = resolve_config("default")
    tactic = build_tactic(config)
    print(tactic(topic))


if __name__ == "__main__":
    main()
