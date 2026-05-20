"""Concrete time-series analysis agent built on top of this package.

Run from repo root:
    source .venv/bin/activate
    PYTHONPATH="$PWD" python Packages/time_series_analytics/concrete_time_series_agent.py
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lllm import build_tactic, resolve_config
from lllm.core.runtime import load_runtime

from tactics.time_series_analysis import TimeSeriesTask


def _read_csv_as_text(path: Path, max_rows: int = 300) -> str:
    """Read a CSV file and return a compact CSV text block."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        rows: list[dict[str, str]] = []
        for idx, row in enumerate(reader):
            rows.append(row)
            if idx + 1 >= max_rows:
                break

    out = []
    header = ",".join(reader.fieldnames)
    out.append(header)
    for row in rows:
        out.append(",".join(str(row.get(col, "")) for col in reader.fieldnames))
    return "\n".join(out)


@dataclass
class DemandForecastAgent:
    """Concrete agent for demand/sales time-series analysis."""

    config_name: str = "time_series_analytics:default"

    def __post_init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        toml = repo_root / "Packages" / "time_series_analytics" / "lllm.toml"
        self.runtime = load_runtime(
            toml_path=str(toml),
            name="time_series_analytics_runtime",
            discover_shared_packages=False,
        )
        cfg = resolve_config(self.config_name, runtime=self.runtime)
        self.tactic = build_tactic(cfg, runtime=self.runtime)

    def analyze_sales_csv(
        self,
        csv_path: str | Path,
        *,
        timestamp_col: str = "date",
        value_col: str = "sales",
        horizon: int = 14,
        frequency: str = "D",
        objective: str = "Detect anomalies and forecast upcoming demand.",
        max_rows: int = 300,
    ) -> dict:
        """Analyze a sales series and return structured JSON-like dict."""
        csv_path = Path(csv_path).resolve()
        series_data = _read_csv_as_text(csv_path, max_rows=max_rows)

        task = TimeSeriesTask(
            series_data=series_data,
            timestamp_col=timestamp_col,
            value_col=value_col,
            horizon=horizon,
            frequency=frequency,
            objective=objective,
        )
        result = self.tactic(task)
        return result.model_dump()


def _write_demo_csv(path: Path, rows: Iterable[tuple[str, int]]) -> None:
    """Create a small demo CSV so the script is runnable immediately."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "sales"])
        for date, sales in rows:
            writer.writerow([date, sales])


if __name__ == "__main__":
    # Quick demo data with one visible spike for anomaly detection.
    demo_rows = [
        ("2026-05-01", 120),
        ("2026-05-02", 125),
        ("2026-05-03", 122),
        ("2026-05-04", 121),
        ("2026-05-05", 180),
        ("2026-05-06", 124),
        ("2026-05-07", 127),
        ("2026-05-08", 129),
        ("2026-05-09", 130),
        ("2026-05-10", 131),
    ]

    demo_csv = Path(__file__).resolve().parent / "demo_sales.csv"
    if not demo_csv.exists():
        _write_demo_csv(demo_csv, demo_rows)

    agent = DemandForecastAgent(config_name="time_series_analytics:default")
    output = agent.analyze_sales_csv(
        demo_csv,
        timestamp_col="date",
        value_col="sales",
        horizon=7,
        frequency="D",
        objective="Detect anomalies and forecast next-week sales demand.",
    )
    print(json.dumps(output, indent=2))
