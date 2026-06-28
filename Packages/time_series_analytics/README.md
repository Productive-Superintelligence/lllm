# time_series_analytics

Reusable time-series analysis package for the `lllm` framework.

The numeric forecast and anomaly detection are produced by a real statistical
model (Holt-Winters / exponential smoothing via `statsmodels`, with numpy
fallbacks and a robust MAD-based anomaly detector). The LLM agents **interpret**
those results and write the narrative report; they never invent the numbers.

Pipeline: statistical forecast -> profile (LLM) -> interpret (LLM) -> synthesize (LLM).
After synthesis, the computed forecast/anomalies are written back authoritatively,
so the output is guaranteed to match the statistical model.

## Dependencies

- Core framework: `pydantic`, `pyyaml`, `litellm` (+ `boto3` for Amazon Bedrock).
- Forecasting: `numpy`, `pandas`, `statsmodels` (`scipy` optional, for exact
  prediction-interval z-values).

## Included resources

- `prompts/system.py`: role prompts for profiler, forecast-interpreter, synthesizer.
- `prompts/task.py`: task prompts and `TimeSeriesAnalysisResult` schema.
- `tactics/time_series_analysis.py`: `TimeSeriesAnalysisTactic` pipeline.
- `tactics/_forecasting.py`: statistical forecasting + anomaly detection (`run_statistical_forecast`).
- `configs/default.yaml`: baseline config for low-cost runs (Haiku everywhere).
- `configs/balanced.yaml`: mixed-model config (Haiku for profile/synthesis,
  Sonnet for the forecast-interpretation step).
- `configs/high_accuracy.yaml`: higher-quality config (Sonnet everywhere).

## Quick usage

```python
from lllm import build_tactic, resolve_config
from tactics.time_series_analysis import TimeSeriesTask

config = resolve_config("time_series_analytics:balanced")
tactic = build_tactic(config)

task = TimeSeriesTask(
    series_data="timestamp,value\n2026-01-01,100\n2026-01-02,103\n2026-01-03,98",
    timestamp_col="timestamp",
    value_col="value",
    horizon=7,
    frequency="D",
    objective="Detect anomalies and forecast next week.",
    confidence_level=0.90,  # prediction-interval width
)

result = tactic(task)
print(result.model_dump_json(indent=2))
```

## Usage scripts

- `concrete_time_series_agent.py`: `DemandForecastAgent` — a ready-made class that
  analyzes a sales/demand CSV file (`analyze_sales_csv(...)`) and returns the result
  as a dict; creates a `demo_sales.csv` and runs a demo when executed directly.
- `example_runner.py`: minimal end-to-end runner on an inline CSV sample.
- `time_series-testcase.ipynb`: notebook walkthrough with a demo run plus cells that
  validate the reported forecast matches the statistical model.

## Package integration notes

- Namespace is `time_series_analytics` from `lllm.toml`.
- Configs use package-qualified `system_prompt_path` for dependency-safe loading.
- Configs target Amazon Bedrock cross-region inference profiles; adjust the
  `model_name` ids per account/region (`aws bedrock list-inference-profiles`).
- Install by dropping folder into `lllm_packages/` or using `lllm pkg install`.

