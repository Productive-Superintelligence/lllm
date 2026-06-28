"""Statistical forecasting and anomaly detection for time-series analysis.

This module does the *numeric* work that an LLM is bad at: it fits a real
statistical model (Holt-Winters exponential smoothing via statsmodels, with
numpy fallbacks) to produce point forecasts and prediction intervals, and it
flags anomalies using a robust (median/MAD) z-score.

The LLM agents downstream are responsible only for *interpreting* these
results, never for inventing the numbers.

The module degrades gracefully:
- pandas/numpy are required (lightweight, already used by the notebook).
- statsmodels is used when available; otherwise an OLS linear-trend fallback
  produces the point forecast.
"""
from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# Z-multipliers for common two-sided confidence levels.
_Z_TABLE = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}

# Approximate number of periods in one seasonal cycle, keyed by the leading
# letters of a pandas-style frequency alias (D, W, M, H, ...).
_SEASONAL_PERIODS = {
    "H": 24,   # hourly -> daily cycle
    "T": 60, "MIN": 60,
    "S": 60,
    "B": 5,    # business days -> weekly cycle
    "D": 7,    # daily -> weekly cycle
    "W": 52,   # weekly -> yearly cycle
    "M": 12, "MS": 12,
    "Q": 4, "QS": 4,
    "Y": 1, "A": 1, "YS": 1,
}


@dataclass
class StatisticalForecast:
    """Numeric forecast + anomalies produced by a statistical model."""

    points: List[Dict] = field(default_factory=list)      # step/expected_value/lower_bound/upper_bound
    anomalies: List[Dict] = field(default_factory=list)   # start/end/reason
    diagnostics: Dict = field(default_factory=dict)
    method: str = ""
    confidence_level: float = 0.90

    # -- prompt-friendly renderings -----------------------------------------
    def forecast_table_text(self) -> str:
        if not self.points:
            return "(no forecast produced)"
        pct = int(round(self.confidence_level * 100))
        lines = [f"step  expected  lower({pct}%)  upper({pct}%)"]
        for p in self.points:
            lines.append(
                f"{p['step']:>4}  {p['expected_value']:>8.3f}  "
                f"{p['lower_bound']:>9.3f}  {p['upper_bound']:>9.3f}"
            )
        return "\n".join(lines)

    def anomalies_text(self) -> str:
        if not self.anomalies:
            return "(no anomalies detected by the statistical detector)"
        return "\n".join(
            f"- {a['start']}..{a['end']}: {a['reason']}" for a in self.anomalies
        )

    def diagnostics_text(self) -> str:
        order = [
            "n_observations", "model", "trend", "seasonality", "seasonal_periods",
            "residual_std", "in_sample_mae", "confidence_level",
        ]
        keys = order + [k for k in self.diagnostics if k not in order]
        return "\n".join(
            f"- {k}: {self.diagnostics[k]}" for k in keys if k in self.diagnostics
        )


def _z_for(confidence_level: float) -> float:
    """Return the two-sided normal multiplier for a confidence level."""
    if confidence_level in _Z_TABLE:
        return _Z_TABLE[confidence_level]
    try:
        from scipy.stats import norm  # type: ignore

        return float(norm.ppf(0.5 + confidence_level / 2.0))
    except Exception:
        # Nearest tabulated value.
        nearest = min(_Z_TABLE, key=lambda c: abs(c - confidence_level))
        return _Z_TABLE[nearest]


def _seasonal_periods(frequency: str) -> int:
    if not frequency:
        return 1
    letters = "".join(ch for ch in str(frequency).upper() if ch.isalpha())
    if not letters:
        return 1
    # Try the full token first (e.g. "MIN", "MS"), then the leading letter.
    return _SEASONAL_PERIODS.get(letters, _SEASONAL_PERIODS.get(letters[0], 1))


def _parse_series(
    series_data: str, timestamp_col: str, value_col: str
) -> Tuple[List[str], np.ndarray]:
    """Parse CSV-like text into (timestamps, values), dropping non-numeric rows."""
    df = pd.read_csv(io.StringIO(series_data))
    df.columns = [str(c).strip() for c in df.columns]

    # Resolve the value column, falling back to the last numeric-looking column.
    vcol = value_col if value_col in df.columns else None
    if vcol is None:
        numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not numeric_cols:
            raise ValueError(
                f"Could not find value column '{value_col}' or any numeric column "
                f"in data with columns {list(df.columns)}"
            )
        vcol = numeric_cols[-1]

    values = pd.to_numeric(df[vcol], errors="coerce")
    mask = values.notna()
    df = df[mask].reset_index(drop=True)
    values = values[mask].to_numpy(dtype=float)

    if timestamp_col in df.columns:
        timestamps = df[timestamp_col].astype(str).tolist()
    else:
        timestamps = [str(i) for i in range(len(values))]

    return timestamps, values


def _fit_and_forecast(
    values: np.ndarray, horizon: int, seasonal_periods: int, use_seasonal: bool
) -> Tuple[np.ndarray, np.ndarray, str, Dict]:
    """Return (point_forecast, in_sample_residuals, method, diagnostics)."""
    n = len(values)
    diagnostics: Dict = {}

    # --- Preferred path: statsmodels Holt-Winters -------------------------
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if use_seasonal and n >= 2 * seasonal_periods + 1:
                model = ExponentialSmoothing(
                    values, trend="add", seasonal="add",
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                )
                method = f"Holt-Winters (additive trend + additive seasonality, m={seasonal_periods})"
                diagnostics["trend"] = "additive"
                diagnostics["seasonality"] = f"additive (period={seasonal_periods})"
            elif n >= 4:
                model = ExponentialSmoothing(
                    values, trend="add", initialization_method="estimated"
                )
                method = "Holt's linear trend (additive trend, no seasonality)"
                diagnostics["trend"] = "additive"
                diagnostics["seasonality"] = "none (insufficient data for a full seasonal cycle)"
            else:
                model = ExponentialSmoothing(values, initialization_method="estimated")
                method = "Simple exponential smoothing (level only)"
                diagnostics["trend"] = "none"
                diagnostics["seasonality"] = "none"

            fit = model.fit()
            point = np.asarray(fit.forecast(horizon), dtype=float)
            fitted = np.asarray(fit.fittedvalues, dtype=float)
            resid = values[-len(fitted):] - fitted if len(fitted) else values - np.mean(values)
            diagnostics["model"] = "statsmodels.ExponentialSmoothing"
            return point, resid, method, diagnostics
    except Exception as exc:  # pragma: no cover - fallback path
        diagnostics["statsmodels_error"] = f"{type(exc).__name__}: {exc}"

    # --- Fallback: OLS linear trend (or mean) via numpy -------------------
    x = np.arange(n, dtype=float)
    if n >= 2:
        slope, intercept = np.polyfit(x, values, 1)
        future_x = np.arange(n, n + horizon, dtype=float)
        point = slope * future_x + intercept
        resid = values - (slope * x + intercept)
        method = "OLS linear trend (numpy fallback; statsmodels unavailable)"
        diagnostics["trend"] = "linear (OLS)"
        diagnostics["seasonality"] = "none"
    else:
        level = float(values[-1]) if n else 0.0
        point = np.full(horizon, level, dtype=float)
        resid = np.zeros(max(n, 1))
        method = "Naive last-value (insufficient data)"
        diagnostics["trend"] = "none"
        diagnostics["seasonality"] = "none"
    diagnostics["model"] = "numpy"
    return point, resid, method, diagnostics


def _detect_anomalies(
    timestamps: List[str], values: np.ndarray, threshold: float = 3.5
) -> List[Dict]:
    """Flag anomalous points using a robust (median/MAD) z-score on detrended data."""
    n = len(values)
    if n < 3:
        return []

    # Detrend with a centered rolling median so trend/level shifts don't trigger.
    window = min(7, n if n % 2 else n - 1)
    window = max(window, 3)
    s = pd.Series(values)
    baseline = s.rolling(window=window, center=True, min_periods=1).median().to_numpy()
    resid = values - baseline

    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    if mad > 1e-9:
        robust_z = 0.6745 * (resid - med) / mad
        scale_note = "MAD"
    else:
        std = float(np.std(resid, ddof=1)) if n > 1 else 0.0
        if std <= 1e-9:
            return []
        robust_z = (resid - med) / std
        scale_note = "std"

    flagged = np.where(np.abs(robust_z) > threshold)[0]
    if len(flagged) == 0:
        return []

    # Group consecutive indices into windows.
    anomalies: List[Dict] = []
    group = [int(flagged[0])]
    for idx in flagged[1:]:
        if idx == group[-1] + 1:
            group.append(int(idx))
        else:
            anomalies.append(_window(timestamps, values, baseline, robust_z, group, scale_note))
            group = [int(idx)]
    anomalies.append(_window(timestamps, values, baseline, robust_z, group, scale_note))
    return anomalies


def _window(timestamps, values, baseline, robust_z, group, scale_note) -> Dict:
    i0, i1 = group[0], group[-1]
    peak = max(group, key=lambda i: abs(robust_z[i]))
    direction = "above" if values[peak] >= baseline[peak] else "below"
    return {
        "start": timestamps[i0],
        "end": timestamps[i1],
        "reason": (
            f"Observed {values[peak]:.2f} vs local level ~{baseline[peak]:.2f} "
            f"({abs(robust_z[peak]):.1f} robust \u03c3 {direction} expected, {scale_note}-based)."
        ),
    }


def run_statistical_forecast(
    series_data: str,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    horizon: int = 12,
    frequency: str = "D",
    confidence_level: float = 0.90,
) -> StatisticalForecast:
    """Fit a statistical model and return numeric forecast + anomalies.

    Raises ValueError only if the series cannot be parsed at all.
    """
    timestamps, values = _parse_series(series_data, timestamp_col, value_col)
    n = len(values)
    if n == 0:
        raise ValueError("No numeric observations found in the provided series data.")

    m = _seasonal_periods(frequency)
    use_seasonal = m >= 2 and n >= 2 * m + 1

    point, resid, method, diagnostics = _fit_and_forecast(values, horizon, m, use_seasonal)

    # Residual-based prediction intervals that widen with the horizon. Use a
    # robust scale (MAD-based) so a detected outlier doesn't balloon the bands;
    # fall back to classical std, then to the std of first differences.
    resid = np.asarray(resid, dtype=float)
    resid = resid[~np.isnan(resid)]
    resid_std = 0.0
    if resid.size > 1:
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        if mad > 1e-9:
            resid_std = 1.4826 * mad  # robust estimate of the standard deviation
        else:
            resid_std = float(np.std(resid, ddof=1))
    if resid_std <= 1e-9:
        diffs = np.diff(values)
        resid_std = float(np.std(diffs, ddof=1)) if diffs.size > 1 else max(abs(np.mean(values)) * 0.05, 1.0)

    z = _z_for(confidence_level)
    non_negative = bool(np.min(values) >= 0)

    points: List[Dict] = []
    for i in range(horizon):
        width = z * resid_std * math.sqrt(i + 1)
        expected = float(point[i])
        lower = expected - width
        upper = expected + width
        if non_negative:
            lower = max(0.0, lower)
        points.append({
            "step": i + 1,
            "expected_value": round(expected, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
        })

    anomalies = _detect_anomalies(timestamps, values)

    in_sample_mae = float(np.mean(np.abs(resid))) if resid.size else 0.0
    diagnostics.update({
        "n_observations": n,
        "seasonal_periods": m if use_seasonal else 1,
        "residual_std": round(resid_std, 4),
        "in_sample_mae": round(in_sample_mae, 4),
        "confidence_level": confidence_level,
        "non_negative_clamped": non_negative,
        "n_anomalies": len(anomalies),
    })

    return StatisticalForecast(
        points=points,
        anomalies=anomalies,
        diagnostics=diagnostics,
        method=method,
        confidence_level=confidence_level,
    )
