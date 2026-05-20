# time_series_analytics

Reusable time-series analysis package for the `lllm` framework.

## Included resources

- `prompts/system.py`: role prompts for profiler, forecaster, synthesizer.
- `prompts/task.py`: task prompts and `TimeSeriesAnalysisResult` schema.
- `tactics/time_series_analysis.py`: `TimeSeriesAnalysisTactic` pipeline.
- `configs/default.yaml`: baseline config for low-cost runs.
- `configs/high_accuracy.yaml`: inherited config for higher-quality runs.

## Quick usage

```python
from lllm import build_tactic, resolve_config
from tactics.time_series_analysis import TimeSeriesTask

config = resolve_config("time_series_analytics:default")
tactic = build_tactic(config)

task = TimeSeriesTask(
    series_data="timestamp,value\n2026-01-01,100\n2026-01-02,103\n2026-01-03,98",
    timestamp_col="timestamp",
    value_col="value",
    horizon=7,
    frequency="D",
    objective="Detect anomalies and forecast next week."
)

result = tactic(task)
print(result.model_dump_json(indent=2))
```

## Package integration notes

- Namespace is `time_series_analytics` from `lllm.toml`.
- Configs use package-qualified `system_prompt_path` for dependency-safe loading.
- Install by dropping folder into `lllm_packages/` or using `lllm pkg install`.
