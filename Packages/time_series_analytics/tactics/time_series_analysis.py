"""Tactic orchestration for end-to-end time-series analysis."""
import logging

from pydantic import BaseModel, Field

from lllm import Tactic, load_prompt

# Import the statistical forecasting helper. Use a relative import for the
# normal package context, and fall back to loading by file path when this
# module is loaded in isolation by LLLM's resource discovery (which has no
# package context, so relative imports are unavailable).
try:  # pragma: no cover - exercised by both import paths
    from ._forecasting import run_statistical_forecast
except ImportError:
    import sys as _sys
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _spec = _ilu.spec_from_file_location(
        "time_series_analytics_forecasting",
        _Path(__file__).with_name("_forecasting.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    # Register before exec so dataclass/pydantic can resolve the module namespace
    # (this module is loaded in isolation by LLLM discovery, not as a package).
    _sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)
    run_statistical_forecast = _mod.run_statistical_forecast

logger = logging.getLogger(__name__)


class TimeSeriesTask(BaseModel):
    """Input payload for the time-series analysis tactic."""

    series_data: str = Field(
        description="Raw time-series content (CSV snippet, JSON lines, or compact table text)."
    )
    timestamp_col: str = Field(default="timestamp")
    value_col: str = Field(default="value")
    horizon: int = Field(default=12, ge=1, description="Number of future steps to forecast.")
    frequency: str = Field(
        default="D", description="Pandas-like frequency alias (D, W, M, H, etc.)."
    )
    objective: str = Field(
        default="Understand behavior and produce a forecast with anomalies."
    )
    confidence_level: float = Field(
        default=0.90, gt=0.0, lt=1.0,
        description="Two-sided confidence level for the prediction intervals.",
    )


class TimeSeriesAnalysisTactic(Tactic):
    """Pipeline: statistical forecast -> profile -> interpret -> synthesize.

    The numeric forecast and anomalies are produced by a real statistical model
    (see ``forecasting.run_statistical_forecast``). The LLM agents interpret and
    narrate those results; they never invent the numbers. After synthesis the
    computed forecast and anomalies are written back authoritatively so the
    output is guaranteed to match the statistical model.
    """

    name = "time_series_analysis"
    agent_group = ["profiler", "forecaster", "synthesizer"]

    def call(self, task: TimeSeriesTask):
        profiler = self.agents["profiler"]
        forecaster = self.agents["forecaster"]
        synthesizer = self.agents["synthesizer"]

        profile_prompt = load_prompt("task/profile")
        forecast_prompt = load_prompt("task/forecast")
        synthesize_prompt = load_prompt("task/synthesize")

        # --- 1) Real statistical forecast + anomaly detection (deterministic) ---
        stat = run_statistical_forecast(
            series_data=task.series_data,
            timestamp_col=task.timestamp_col,
            value_col=task.value_col,
            horizon=task.horizon,
            frequency=task.frequency,
            confidence_level=task.confidence_level,
        )
        forecast_text = stat.forecast_table_text()
        anomalies_text = stat.anomalies_text()
        diagnostics_text = stat.diagnostics_text()
        backtest_text = stat.backtest_text()

        # --- 2) Profile the raw data --------------------------------------------
        profiler.open("profile")
        profiler.receive_prompt(
            profile_prompt,
            {
                "series_data": task.series_data,
                "timestamp_col": task.timestamp_col,
                "value_col": task.value_col,
                "horizon": task.horizon,
                "frequency": task.frequency,
                "objective": task.objective,
            },
        )
        profile_report = profiler.respond().content

        # --- 3) Interpret the statistical forecast (no number invention) --------
        forecaster.open("forecast")
        forecaster.receive_prompt(
            forecast_prompt,
            {
                "objective": task.objective,
                "horizon": task.horizon,
                "frequency": task.frequency,
                "profile_report": profile_report,
                "forecast_method": stat.method,
                "diagnostics": diagnostics_text,
                "statistical_forecast": forecast_text,
                "detected_anomalies": anomalies_text,
                "backtest": backtest_text,
            },
        )
        forecast_report = forecaster.respond().content

        # --- 4) Synthesize the final report -------------------------------------
        synthesizer.open("synthesize")
        synthesizer.receive_prompt(
            synthesize_prompt,
            {
                "objective": task.objective,
                "horizon": task.horizon,
                "frequency": task.frequency,
                "profile_report": profile_report,
                "forecast_report": forecast_report,
                "forecast_method": stat.method,
                "diagnostics": diagnostics_text,
                "statistical_forecast": forecast_text,
                "detected_anomalies": anomalies_text,
                "backtest": backtest_text,
            },
        )
        response = synthesizer.respond()

        output_model = synthesize_prompt.format
        if isinstance(response.parsed, output_model):
            result = response.parsed
        else:
            result = output_model(**response.parsed)

        # --- 5) Make the statistical numbers authoritative ----------------------
        return self._apply_statistical_results(result, stat, output_model)

    @staticmethod
    def _apply_statistical_results(result, stat, output_model):
        """Overwrite forecast (and anomalies) with the computed statistical values."""
        import typing

        forecast_field = output_model.model_fields["forecast"]
        point_model = forecast_field.annotation.__args__[0]
        result.forecast = [point_model(**p) for p in stat.points]

        # Anomalies are also computed statistically; use them as the source of
        # truth so the report is internally consistent with the detector.
        if stat.anomalies:
            anomaly_field = output_model.model_fields["anomalies"]
            anomaly_model = anomaly_field.annotation.__args__[0]
            result.anomalies = [anomaly_model(**a) for a in stat.anomalies]

        # Backtest metrics (out-of-sample accuracy) are authoritative from code.
        if stat.backtest_metrics:
            bt_field = output_model.model_fields["backtest_metrics"]
            bt_args = [a for a in typing.get_args(bt_field.annotation) if a is not type(None)]
            if bt_args:
                bt_model = bt_args[0]
                allowed = set(bt_model.model_fields)
                result.backtest_metrics = bt_model(
                    **{k: v for k, v in stat.backtest_metrics.items() if k in allowed}
                )
        return result
