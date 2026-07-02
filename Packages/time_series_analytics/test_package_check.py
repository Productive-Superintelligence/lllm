"""Local integration check for the time_series_analytics LLLM package.

Run:
    python Packages/time_series_analytics/test_package_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from lllm import build_tactic, resolve_config
    from lllm.core.runtime import load_runtime

    toml = repo_root / "Packages" / "time_series_analytics" / "lllm.toml"
    runtime = load_runtime(
        toml_path=str(toml),
        name="time_series_analytics_check",
        discover_shared_packages=False,
    )

    checks = {
        "system/profiler": runtime.has("time_series_analytics.prompts:system/profiler"),
        "task/synthesize": runtime.has("time_series_analytics.prompts:task/synthesize"),
        "tactic": runtime.has("time_series_analytics.tactics:time_series_analysis"),
        "config/default": runtime.has("time_series_analytics.configs:default"),
    }
    print("Discovery checks:", checks)

    missing = [k for k, ok in checks.items() if not ok]
    if missing:
        print("FAILED: missing resources:", ", ".join(missing))
        return 1

    cfg = resolve_config("time_series_analytics:default", runtime=runtime)
    tactic = build_tactic(cfg, runtime=runtime)

    if tactic.name != "time_series_analysis":
        print(f"FAILED: unexpected tactic name: {tactic.name}")
        return 1

    print(f"PASS: Built tactic '{tactic.name}' successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
