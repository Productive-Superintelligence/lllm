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
        "Interpret the statistical forecast below. Do NOT recompute or change any numbers.\n"
        "Objective: {objective}\n"
        "Horizon: {horizon}\n"
        "Frequency: {frequency}\n\n"
        "Profiler findings:\n{profile_report}\n\n"
        "Forecasting method used: {forecast_method}\n\n"
        "Model diagnostics:\n{diagnostics}\n\n"
        "Forecast (computed by the statistical model):\n{statistical_forecast}\n\n"
        "Anomalies (computed by the statistical detector):\n{detected_anomalies}\n\n"
        "Explain what this forecast implies, how reliable it is, the assumptions and\n"
        "failure modes, and how the anomalies relate to it. Numbers are fixed."
    ),
    metadata={"stage": "forecast"},
)

synthesize = Prompt(
    path="synthesize",
    prompt=(
        "Create the final time-series analysis output as JSON matching the schema.\n"
        "Objective: {objective}\n"
        "Horizon: {horizon}\n"
        "Frequency: {frequency}\n\n"
        "Profiler report:\n{profile_report}\n\n"
        "Forecast interpretation:\n{forecast_report}\n\n"
        "Forecasting method: {forecast_method}\n"
        "Model diagnostics:\n{diagnostics}\n\n"
        "Authoritative forecast points (use these EXACTLY in the 'forecast' field):\n"
        "{statistical_forecast}\n\n"
        "Authoritative anomalies (use these EXACTLY in the 'anomalies' field):\n"
        "{detected_anomalies}\n\n"
        "Do not alter the forecast numbers or anomaly windows. Write the narrative\n"
        "fields (summary, key_patterns, data_quality_issues, recommendations,\n"
        "confidence_note) grounded in the profiler findings and the interpretation."
    ),
    format=TimeSeriesAnalysisResult,
    metadata={"stage": "synthesize"},
)
