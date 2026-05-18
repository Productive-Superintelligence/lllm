from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LLLM_CONFIG", str(PROJECT_ROOT / "lllm.toml"))

from lllm import build_tactic, load_resource, resolve_config  # noqa: E402


def main() -> None:
    config = resolve_config("default")
    tactic = build_tactic(config)
    topics = load_resource("data:topics.yaml")["topics"]
    for topic in topics:
        print("=" * 80)
        print(topic)
        print(tactic(topic))


if __name__ == "__main__":
    main()
