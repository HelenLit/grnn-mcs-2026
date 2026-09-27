"""Exploratory Data Analysis (EDA) Module for GRNN Forecasting UI.

Implements descriptive statistics calculation per series and multi-series
aggregate summaries:
  - Per series: count, min, max, mean, median, mode, std, Q1, Q3, IQR,
    skewness, kurtosis, outliers_count (|z| > 3).
  - Multi-series datasets: aggregate summary across series plus a
    series-length distribution row (min/median/max length).
  - Output is numeric tables only, no plots.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st

from src.ui.data_loader import SeriesData
import src.ui.strings_ua as ua


@dataclass
class SeriesEDA:
    """Descriptive statistics for a single time series.

    Fields:
        series_id: Identifier of the series.
        count: Number of observation points (N).
        min: Minimum observation value.
        max: Maximum observation value.
        mean: Arithmetic mean.
        median: Sample median (50th percentile).
        mode: Sample mode.
        std: Sample standard deviation (ddof=1).
        q1: 25th percentile (Q1).
        q3: 75th percentile (Q3).
        iqr: Interquartile Range (Q3 - Q1).
        skewness: Fisher-Pearson skewness coefficient.
        kurtosis: Fisher excess kurtosis.
        outliers_count: Count of extreme points with |z| > 3.
    """
    series_id: str
    count: int
    min: float
    max: float
    mean: float
    median: float
    mode: float
    std: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    outliers_count: int

    def to_dict(self, localized_keys: bool = False) -> dict[str, Any]:
        """Convert statistics to a dictionary with optional Ukrainian localized keys."""
        raw = asdict(self)
        if not localized_keys:
            return raw
        return {ua.EDA_COLS.get(k, k): v for k, v in raw.items()}


def compute_series_eda(series: SeriesData) -> SeriesEDA:
    """Calculate descriptive statistics for a single SeriesData object.

    Args:
        series: SeriesData instance.

    Returns:
        SeriesEDA dataclass with computed metrics.
    """
    y = np.asarray(series.values, dtype=np.float64)
    n = len(y)

    min_val = float(np.min(y))
    max_val = float(np.max(y))
    mean_val = float(np.mean(y))
    median_val = float(np.median(y))

    # Mode: use scipy mode
    mode_res = stats.mode(y, keepdims=False)
    mode_val = float(np.asarray(mode_res.mode).reshape(-1)[0])

    # Sample standard deviation (ddof=1 for n > 1)
    std_val = float(np.std(y, ddof=1)) if n > 1 else 0.0

    # Quartiles & IQR
    q1_val = float(np.percentile(y, 25))
    q3_val = float(np.percentile(y, 75))
    iqr_val = float(q3_val - q1_val)

    # Skewness & Kurtosis (require std > 0 and sufficient sample size)
    if std_val > 1e-12 and n >= 3:
        skew_val = float(stats.skew(y, bias=False))
    else:
        skew_val = 0.0

    if std_val > 1e-12 and n >= 4:
        kurt_val = float(stats.kurtosis(y, bias=False, fisher=True))
    else:
        kurt_val = 0.0

    # Outliers count (|z| > 3)
    if std_val > 1e-12:
        z_scores = np.abs((y - mean_val) / std_val)
        outliers_count = int(np.sum(z_scores > 3.0))
    else:
        outliers_count = 0

    return SeriesEDA(
        series_id=series.series_id,
        count=n,
        min=min_val,
        max=max_val,
        mean=mean_val,
        median=median_val,
        mode=mode_val,
        std=std_val,
        q1=q1_val,
        q3=q3_val,
        iqr=iqr_val,
        skewness=skew_val,
        kurtosis=kurt_val,
        outliers_count=outliers_count,
)


def compute_dataset_eda(
    series_list: list[SeriesData],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute comprehensive EDA across all series in the dataset.

    Args:
        series_list: List of SeriesData objects.

    Returns:
        tuple of:
          - per_series_df: DataFrame of per-series statistics with Ukrainian column names.
          - aggregate_df: DataFrame summarizing statistical distributions across series.
          - meta: Dictionary with dataset summary statistics.
    """
    if not series_list:
        return pd.DataFrame(), pd.DataFrame(), {}

    eda_records: list[SeriesEDA] = [compute_series_eda(s) for s in series_list]

    # Build per-series table with localized column names
    per_series_rows = [rec.to_dict(localized_keys=True) for rec in eda_records]
    per_series_df = pd.DataFrame(per_series_rows)

    # Dataset length distribution
    lengths = [s.length for s in series_list]
    min_len = int(np.min(lengths))
    max_len = int(np.max(lengths))
    median_len = float(np.median(lengths))
    total_points = sum(lengths)
    total_outliers = sum(rec.outliers_count for rec in eda_records)

    meta = {
        "n_series": len(series_list),
        "min_len": min_len,
        "median_len": median_len,
        "max_len": max_len,
        "total_points": total_points,
        "total_outliers": total_outliers,
    }

    # Aggregate summary across series (numeric columns only)
    numeric_cols = [
        "count", "min", "max", "mean", "median", "mode",
        "std", "q1", "q3", "iqr", "skewness", "kurtosis", "outliers_count",
    ]
    raw_df = pd.DataFrame([rec.to_dict(localized_keys=False) for rec in eda_records])

    agg_data = {
        "Показник": [
            "Середнє за рядами (Mean)",
            "Медіана за рядами (Median)",
            "Мінімум (Min)",
            "Максимум (Max)",
            "Станд. відхилення (Std)",
        ]
    }

    for col in numeric_cols:
        col_label = ua.EDA_COLS.get(col, col)
        vals = raw_df[col].to_numpy(dtype=np.float64)
        agg_data[col_label] = [
            float(np.mean(vals)),
            float(np.median(vals)),
            float(np.min(vals)),
            float(np.max(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        ]

    aggregate_df = pd.DataFrame(agg_data)

    return per_series_df, aggregate_df, meta


def render_eda_tab(series_list: list[SeriesData]) -> None:
    """Render the complete EDA panel view inside the Streamlit tab.

    Args:
        series_list: Active list of SeriesData.
    """
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.subheader(ua.EDA_HEADER)

    if not series_list:
        st.info("Немає активних даних для аналізу. Будь ласка, завантажте CSV або оберіть вибірку M4.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Compute EDA data
    per_series_df, aggregate_df, meta = compute_dataset_eda(series_list)

    # 1. Dataset summary header metrics
    st.markdown(
        f"**{ua.EDA_DATASET_SUMMARY_TEMPLATE.format(n_series=meta['n_series'], min_len=meta['min_len'], median_len=meta['median_len'], max_len=meta['max_len'])}** "
        f"· Всього спостережень: `{meta['total_points']:,}` · Викидів (|z| > 3): `{meta['total_outliers']}`"
)

    st.markdown("---")

    # 2. Multi-series aggregate distribution table (if dataset has > 1 series)
    if meta["n_series"] > 1:
        st.markdown("#### Зведена описова статистика за всіма рядами")
        # Format numeric floats nicely
        styled_agg = aggregate_df.copy()
        for col in styled_agg.columns:
            if col != "Показник":
                styled_agg[col] = styled_agg[col].apply(lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))
        st.dataframe(styled_agg, use_container_width=True, hide_index=True)
        st.markdown("---")

    # 3. Per-series statistics table
    st.markdown("#### Деталізована статистика по кожному ряду")
    styled_per_series = per_series_df.copy()
    float_cols = [
        c for c in styled_per_series.columns
        if c not in (ua.EDA_COLS["series_id"], ua.EDA_COLS["count"], ua.EDA_COLS["outliers_count"])
    ]
    for col in float_cols:
        styled_per_series[col] = styled_per_series[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (float, int)) else str(x))

    st.dataframe(styled_per_series, use_container_width=True, hide_index=True)
    if len(series_list) > 1:
        st.caption(f"Показано статистику для {len(series_list)} рядів.")

    # 4. CSV Download Button
    csv_bytes = per_series_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"💾 {ua.BTN_DOWNLOAD_EDA}",
        data=csv_bytes,
        file_name="eda.csv",
        mime="text/csv",
        use_container_width=True,
)

    st.markdown('</div>', unsafe_allow_html=True)
