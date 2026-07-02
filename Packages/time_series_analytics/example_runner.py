"""Minimal runner for the time_series_analytics package."""
from lllm import build_tactic, resolve_config

from tactics.time_series_analysis import TimeSeriesTask


def main() -> None:
    config = resolve_config("time_series_analytics:default")
    tactic = build_tactic(config)

    task = TimeSeriesTask(
        series_data=(
            "timestamp,value\n"
            "2026-04-01,122\n"
            "2026-04-02,125\n"
            "2026-04-03,123\n"
            "2026-04-04,180\n"
            "2026-04-05,127\n"
            "2026-04-06,129\n"
            "2026-04-07,130\n"
        ),
        timestamp_col="timestamp",
        value_col="value",
        horizon=5,
        frequency="D",
        objective="Find anomalies and forecast short-term demand.",
    )

    result = tactic(task)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
