from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from lllm import build_tactic, load_resource, load_runtime, resolve_config  # noqa: E402


def main() -> None:
    load_runtime(str(PROJECT_ROOT / "lllm.toml"))
    config = resolve_config("default")
    tactic = build_tactic(config)
    topics = load_resource("data:topics.yaml")["topics"]
    for topic in topics:
        print("=" * 80)
        print(topic)
        print(tactic(topic))


if __name__ == "__main__":
    main()
