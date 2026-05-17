from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("LLLM_CONFIG", str(PROJECT_ROOT / "lllm.toml"))

from lllm import build_tactic, resolve_config  # noqa: E402


def main() -> None:
    task = " ".join(sys.argv[1:]).strip() or "Plan a useful starter project."
    config = resolve_config("default")
    tactic = build_tactic(config)
    print(tactic(task))


if __name__ == "__main__":
    main()
