from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

from lllm import build_tactic, load_runtime, resolve_config  # noqa: E402


def create_tactic():
    """Build the project's tactic from configs/default.yaml.

    `lllm serve .` and the hub use this as the package's primary tactic.
    """

    load_runtime(str(PROJECT_ROOT / "lllm.toml"))
    return build_tactic(resolve_config("default"))


def main() -> None:
    topic = " ".join(sys.argv[1:]).strip() or "agent evaluation methods"
    tactic = create_tactic()
    print(tactic(topic))


if __name__ == "__main__":
    main()
