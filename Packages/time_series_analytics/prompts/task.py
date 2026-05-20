"""Task prompts and output schema for time-series analysis."""
from typing import List

from pydantic import BaseModel, Field

from lllm.core.prompt import Prompt


class DataQualityIssue(BaseModel):
    issue: str = Field(description="Short label of the issue.")
    severity: str = Field(description="One of low, medium, or high.")
    evidence: str = Field(description="Concrete evidence from the provided data.")


class AnomalyWindow(BaseModel):
    start: str = Field(description="Start timestamp or index for the anomaly window.")
    end: str = Field(description="End timestamp or index for the anomaly window.")
    reason: str = Field(description="Brief explanation of why this segment is anomalous.")


class ForecastPoint(BaseModel):
    step: int = Field(description="Forecast horizon step, starting at 1.")
    expected_value: float = Field(description="Expected value at this forecast step.")
    lower_bound: float = Field(description="Conservative lower estimate.")
    upper_bound: float = Field(description="Conservative upper estimate.")


class TimeSeriesAnalysisResult(BaseModel):
    summary: str = Field(description="3-5 sentence executive summary.")
    key_patterns: List[str] = Field(description="Main patterns in the series.")
    data_quality_issues: List[DataQualityIssue] = Field(
        description="Critical data quality observations."
    )
    anomalies: List[AnomalyWindow] = Field(
        description="Anomalous segments in the observed data."
    )
    forecast: List[ForecastPoint] = Field(
        description="Forecast points for the requested horizon."
    )
    recommendations: List[str] = Field(
        description="Actionable next steps for analysts or engineers."
    )
    confidence_note: str = Field(
        description="Confidence statement with caveats and assumptions."
    )


profile = Prompt(
    path="profile",
    prompt=(
        "Profile the following time-series data.\n"
        "Task objective: {objective}\n"
        "Timestamp column: {timestamp_col}\n"
        "Value column: {value_col}\n"
        "Expected frequency: {frequency}\n"
        "Forecast horizon: {horizon}\n\n"
        "Data sample:\n{series_data}\n\n"
        "Return structured bullet points for data quality and pattern clues."
    ),
    metadata={"stage": "profile"},
)

forecast = Prompt(
    path="forecast",
    prompt=(
        "Generate a forecast plan and numeric outlook.\n"
        "Objective: {objective}\n"
        "Horizon: {horizon}\n"
        "Frequency: {frequency}\n\n"
        "Profiler findings:\n{profile_report}\n\n"
        "Data sample:\n{series_data}\n\n"
        "Produce a concise forecast rationale plus horizon-wise estimates."
    ),
    metadata={"stage": "forecast"},
)

synthesize = Prompt(
    path="synthesize",
    prompt=(
        "Create the final time-series analysis output.\n"
        "Objective: {objective}\n"
        "Horizon: {horizon}\n"
        "Frequency: {frequency}\n\n"
        "Profiler report:\n{profile_report}\n\n"
        "Forecaster report:\n{forecast_report}\n\n"
        "Return JSON that matches the required schema."
    ),
    format=TimeSeriesAnalysisResult,
    metadata={"stage": "synthesize"},
)
