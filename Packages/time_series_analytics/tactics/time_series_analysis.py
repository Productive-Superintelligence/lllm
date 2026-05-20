"""Tactic orchestration for end-to-end time-series analysis."""
from pydantic import BaseModel, Field

from lllm import Tactic, load_prompt


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


class TimeSeriesAnalysisTactic(Tactic):
    """Three-agent pipeline: profile -> forecast -> synthesize."""

    name = "time_series_analysis"
    agent_group = ["profiler", "forecaster", "synthesizer"]

    def call(self, task: TimeSeriesTask):
        profiler = self.agents["profiler"]
        forecaster = self.agents["forecaster"]
        synthesizer = self.agents["synthesizer"]

        profile_prompt = load_prompt("task/profile")
        forecast_prompt = load_prompt("task/forecast")
        synthesize_prompt = load_prompt("task/synthesize")

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

        forecaster.open("forecast")
        forecaster.receive_prompt(
            forecast_prompt,
            {
                "series_data": task.series_data,
                "profile_report": profile_report,
                "horizon": task.horizon,
                "frequency": task.frequency,
                "objective": task.objective,
            },
        )
        forecast_report = forecaster.respond().content

        synthesizer.open("synthesize")
        synthesizer.receive_prompt(
            synthesize_prompt,
            {
                "profile_report": profile_report,
                "forecast_report": forecast_report,
                "horizon": task.horizon,
                "frequency": task.frequency,
                "objective": task.objective,
            },
        )
        response = synthesizer.respond()

        output_model = synthesize_prompt.format
        if isinstance(response.parsed, output_model):
            return response.parsed
        return output_model(**response.parsed)
