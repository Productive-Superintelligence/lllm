from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("LLLM_CONFIG", str(PROJECT_ROOT / "lllm.toml"))

from lllm import build_tactic, resolve_config  # noqa: E402


def build_app_tactic():
    config = resolve_config("default")
    return build_tactic(config)


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or "Describe this service in one sentence."
    tactic = build_app_tactic()
    print(tactic(task))


if __name__ == "__main__":
    main()
