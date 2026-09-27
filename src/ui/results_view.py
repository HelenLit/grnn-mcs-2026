"""Results View, Deterministic Exemplar Plotting, and ZIP Export Module.

Implements results presentation:
  - Deterministic plot selection (max 3): best/median/worst by sMAPE (Evaluation)
    or min/median/max by series length (Forecast).
  - High-quality matplotlib charts using the official color tokens.
  - Aggregate and per-series metric tables (sMAPE, MASE, OWA, MAE, RMSE, selected sigma).
  - Reproducibility ZIP package (forecasts.csv, metrics_per_series.csv,
    metrics_aggregate.csv, eda.csv, run_config.json).
"""

from __future__ import annotations

import io
import json
from typing import Any
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import src.ui.strings_ua as ua
import src.ui.theme as theme
from src.ui.run_engine import RunResults, SeriesForecastResult


def select_exemplar_plots(results: RunResults) -> list[tuple[SeriesForecastResult, str]]:
    """Deterministically select 1 to 3 exemplar series for plotting.

    Selection Rules:
      - Total series == 1: exactly 1 plot (best/min length).
      - Total series == 2: exactly 2 plots (best + worst / min + max length).
      - Total series >= 3: exactly 3 plots.
        * Evaluation mode: Best / Median / Worst series by per-series sMAPE.
        * Forecast mode: Shortest / Median / Longest series by training length N.

    Args:
        results: RunResults object containing series_results.

    Returns:
        List of tuples (SeriesForecastResult, criterion_label_ukrainian).
    """
    series_list = results.series_results
    n = len(series_list)
    if n == 0:
        return []

    is_eval = (results.mode == "evaluation")

    if n == 1:
        criterion = ua.PLOT_CRITERION_BEST if is_eval else ua.PLOT_CRITERION_MIN_LEN
        return [(series_list[0], criterion)]

    if is_eval:
        # Sort deterministically by sMAPE ascending (lower is better)
        # Fallback to series_id for tie breaking
        sorted_series = sorted(
            series_list,
            key=lambda r: (r.smape if r.smape is not None else float("inf"), r.series_id),
)

        if n == 2:
            return [
                (sorted_series[0], ua.PLOT_CRITERION_BEST),
                (sorted_series[1], ua.PLOT_CRITERION_WORST),
            ]

        # n >= 3: exactly 3 series (best, median, worst)
        best = sorted_series[0]
        worst = sorted_series[-1]

        # Choose the series with sMAPE closest to the median sMAPE
        all_smapes = [r.smape for r in sorted_series if r.smape is not None]
        median_target = float(np.median(all_smapes)) if all_smapes else 0.0

        # Candidates strictly between best and worst if n >= 3
        candidates = sorted_series[1:-1]
        median_series = min(
            candidates,
            key=lambda r: abs((r.smape if r.smape is not None else 0.0) - median_target),
)

        return [
            (best, ua.PLOT_CRITERION_BEST),
            (median_series, ua.PLOT_CRITERION_MEDIAN),
            (worst, ua.PLOT_CRITERION_WORST),
        ]
    else:
        # Forecast mode: sort by length ascending
        sorted_series = sorted(
            series_list,
            key=lambda r: (len(r.history), r.series_id),
)

        if n == 2:
            return [
                (sorted_series[0], ua.PLOT_CRITERION_MIN_LEN),
                (sorted_series[1], ua.PLOT_CRITERION_MAX_LEN),
            ]

        min_s = sorted_series[0]
        max_s = sorted_series[-1]
        candidates = sorted_series[1:-1]
        median_target = float(np.median([len(r.history) for r in sorted_series]))
        median_s = min(
            candidates,
            key=lambda r: abs(len(r.history) - median_target),
)

        return [
            (min_s, ua.PLOT_CRITERION_MIN_LEN),
            (median_s, ua.PLOT_CRITERION_MEDIAN_LEN),
            (max_s, ua.PLOT_CRITERION_MAX_LEN),
        ]


def create_series_plot(
    result: SeriesForecastResult,
    criterion: str,
    mode: str = "evaluation",
) -> matplotlib.figure.Figure:
    """Create a publication-quality Matplotlib chart matching UI theme tokens.

    Args:
        result: SeriesForecastResult for one series.
        criterion: Ukrainian criterion description (e.g. 'найкращий sMAPE').
        mode: 'evaluation' or 'forecast'.

    Returns:
        Matplotlib Figure object.
    """
    n_train = len(result.history)
    horizon = len(result.forecast)

    t_train = np.arange(1, n_train + 1)
    t_fc = np.arange(n_train + 1, n_train + horizon + 1)

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)
    fig.patch.set_facecolor(theme.MAIN_BACKGROUND)
    ax.set_facecolor("#ffffff")

    # 1. History line (Charcoal #1f1f1f)
    ax.plot(
        t_train,
        result.history,
        color=theme.PLOT_COLORS["history"],
        linewidth=1.8,
        label=ua.PLOT_LEGEND_HISTORY,
        zorder=3,
)

    # Bridge segment connecting history to forecast/actual for visual continuity
    t_bridge = np.array([t_train[-1], t_fc[0]])
    last_hist_val = result.history[-1]

    # 2. Actual holdout line in Evaluation mode (Emerald Green #4CC88B)
    if mode == "evaluation" and result.actual is not None:
        ax.plot(
            t_bridge,
            np.array([last_hist_val, result.actual[0]]),
            color=theme.PLOT_COLORS["actual"],
            linewidth=1.5,
            linestyle=":",
            alpha=0.6,
            zorder=4,
)
        ax.plot(
            t_fc,
            result.actual,
            color=theme.PLOT_COLORS["actual"],
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=ua.PLOT_LEGEND_ACTUAL,
            zorder=5,
)

    # 3. Model forecast line (Dodger Blue #3A8BFF)
    ax.plot(
        t_bridge,
        np.array([last_hist_val, result.forecast[0]]),
        color=theme.PLOT_COLORS["forecast"],
        linewidth=1.5,
        linestyle="--",
        alpha=0.6,
        zorder=4,
)
    ax.plot(
        t_fc,
        result.forecast,
        color=theme.PLOT_COLORS["forecast"],
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=5,
        label=ua.PLOT_LEGEND_FORECAST,
        zorder=6,
)

    # 4. Train/Test Boundary marker
    ax.axvline(
        x=n_train + 0.5,
        color="#888888",
        linestyle=":",
        linewidth=1.2,
        label=ua.PLOT_LEGEND_BOUNDARY,
        zorder=2,
)

    # Formatting and title
    if mode == "evaluation" and result.smape is not None:
        title_text = f"Ряд {result.series_id} ({criterion}: sMAPE = {result.smape:.2f}%, σ = {result.sigma:.3f})"
    else:
        title_text = f"Ряд {result.series_id} ({criterion}: N = {n_train}, σ = {result.sigma:.3f})"

    ax.set_title(title_text, fontsize=11, fontweight="bold", color=theme.PRIMARY_TEXT, pad=10)
    ax.set_xlabel("Часовий індекс (t)", fontsize=9, color=theme.PRIMARY_TEXT)
    ax.set_ylabel("Рівень ряду (y)", fontsize=9, color=theme.PRIMARY_TEXT)

    # Polish grid & spines
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#dddddd")

    ax.legend(loc="upper left", framealpha=0.9, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=8.5)
    plt.tight_layout()

    return fig


def build_forecasts_dataframe(results: RunResults) -> pd.DataFrame:
    """Build a DataFrame containing point forecasts for all series."""
    rows = []
    for r in results.series_results:
        row = {"series_id": r.series_id}
        for h_step, val in enumerate(r.forecast, start=1):
            row[f"F{h_step}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_series_metrics_dataframe(results: RunResults) -> pd.DataFrame:
    """Build a detailed per-series metrics table for UI display and export."""
    rows = []
    for r in results.series_results:
        row = {
            "ID ряду": r.series_id,
            "N (історія)": len(r.history),
            "Обрана σ": round(r.sigma, 4),
        }
        if results.mode == "evaluation":
            row["sMAPE (%)"] = round(r.smape, 3) if r.smape is not None else None
            row["MASE"] = round(r.mase, 3) if r.mase is not None else None
            row["MAE"] = round(r.mae, 3) if r.mae is not None else None
            row["RMSE"] = round(r.rmse, 3) if r.rmse is not None else None
            row["MedAE"] = round(r.medae, 3) if r.medae is not None else None
            row["MAPE (%)"] = round(r.mape, 3) if r.mape is not None else None
            row["Naive2 sMAPE"] = round(r.naive2_smape, 3) if r.naive2_smape is not None else None
            row["Naive2 MASE"] = round(r.naive2_mase, 3) if r.naive2_mase is not None else None

        row["Час (с)"] = round(r.runtime_s, 4)
        rows.append(row)

    return pd.DataFrame(rows)


def build_aggregate_metrics_dataframe(results: RunResults) -> pd.DataFrame:
    """Build aggregate summary metrics table."""
    if results.mode != "evaluation" or not results.aggregate_metrics:
        return pd.DataFrame()

    rows = []
    for metric_name, val in results.aggregate_metrics.items():
        desc = {
            "sMAPE": "Symmetric Mean Absolute Percentage Error (%)",
            "MASE": "Mean Absolute Scaled Error (baseline mase_m)",
            "OWA": "Overall Weighted Average vs Naive2",
            "MAE": "Mean Absolute Error",
            "RMSE": "Root Mean Squared Error",
        }.get(metric_name, metric_name)

        unit = "%" if metric_name == "sMAPE" else ""
        rows.append({
            "Метрика": metric_name,
            "Значення": f"{val:.4f}{unit}",
            "Опис": desc,
        })

    return pd.DataFrame(rows)


def build_run_config_dict(results: RunResults) -> dict[str, Any]:
    """Assemble reproducibility manifest contract dictionary."""
    import scipy
    import pyGRNN

    package_versions = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "streamlit": st.__version__,
        "pyGRNN": getattr(pyGRNN, "__version__", "0.1.2"),
    }

    return {
        "application": "GRNN Forecasting Demo (Streamlit)",
        "timestamp": results.timestamp,
        "mode": results.mode,
        "is_m4": results.is_m4,
        "parameters": results.parameters.to_dict(),
        "dataset_info": results.dataset_info,
        "total_series_processed": len(results.series_results),
        "total_runtime_s": results.total_runtime_s,
        "aggregate_metrics": results.aggregate_metrics,
        "naive2_aggregate": results.naive2_aggregate,
        "package_versions": package_versions,
    }


def generate_results_zip(results: RunResults, eda_df: pd.DataFrame | None = None) -> bytes:
    """Package complete run artifacts into an in-memory ZIP archive.

    Contains:
      - forecasts.csv
      - metrics_per_series.csv (Evaluation only)
      - metrics_aggregate.csv (Evaluation only)
      - eda.csv (if available)
      - run_config.json

    Returns:
        Bytes of the constructed ZIP archive.
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Forecasts
        forecasts_df = build_forecasts_dataframe(results)
        zf.writestr("forecasts.csv", forecasts_df.to_csv(index=False).encode("utf-8"))

        # 2. Per-series metrics (Evaluation mode)
        if results.mode == "evaluation":
            per_series_df = build_per_series_metrics_dataframe(results)
            zf.writestr("metrics_per_series.csv", per_series_df.to_csv(index=False).encode("utf-8"))

            # 3. Aggregate metrics
            agg_df = build_aggregate_metrics_dataframe(results)
            if not agg_df.empty:
                zf.writestr("metrics_aggregate.csv", agg_df.to_csv(index=False).encode("utf-8"))

        # 4. EDA table if provided
        if eda_df is not None and not eda_df.empty:
            zf.writestr("eda.csv", eda_df.to_csv(index=False).encode("utf-8"))

        # 5. Reproducibility config manifest
        config_dict = build_run_config_dict(results)
        config_json = json.dumps(config_dict, indent=2, ensure_ascii=False)
        zf.writestr("run_config.json", config_json.encode("utf-8"))

    buffer.seek(0)
    return buffer.getvalue()


def render_results_tab(results: RunResults | None) -> None:
    """Render the full interactive results tab in Streamlit."""
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.subheader(ua.RESULTS_HEADER)

    if results is None:
        st.info("Натисніть кнопку **«Запустити прогнозування»** на панелі ліворуч для виконання моделювання.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Configuration summary
    st.markdown(
        f"**Параметри запуску:** Режим: `{results.mode}`, "
        f"L={results.parameters.lags}, M={results.parameters.models}, "
        f"Агрегація: `{results.parameters.aggregation}`, Оптимізатор: `{results.parameters.optimizer}`, "
        f"h={results.parameters.horizon} · Оброблено рядів: `{len(results.series_results)}` · Час: `{results.total_runtime_s:.2f} с.`"
)

    # 1. Aggregate KPI Cards (Evaluation mode)
    if results.mode == "evaluation" and results.aggregate_metrics:
        st.markdown(f"#### {ua.METRICS_AGGREGATE_HEADER}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("sMAPE (%)", f"{results.aggregate_metrics.get('sMAPE', 0.0):.3f}%")
        c2.metric("MASE", f"{results.aggregate_metrics.get('MASE', 0.0):.3f}")
        c3.metric("OWA (vs Naive2)", f"{results.aggregate_metrics.get('OWA', 0.0):.3f}")
        c4.metric("MAE", f"{results.aggregate_metrics.get('MAE', 0.0):.3f}")
        c5.metric("RMSE", f"{results.aggregate_metrics.get('RMSE', 0.0):.3f}")
    elif results.mode == "forecast":
        st.info(f"ℹ️ {ua.FORECAST_NO_METRICS_NOTE}")

    st.markdown("---")

    # 2. Deterministic Exemplar Plots (max 3)
    st.markdown(f"#### {ua.PLOTS_HEADER} (детермінована вибірка)")
    exemplars = select_exemplar_plots(results)

    if exemplars:
        for res, criterion in exemplars:
            fig = create_series_plot(result=res, criterion=criterion, mode=results.mode)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("---")

    # 3. Tabular Results View
    st.markdown(f"#### {ua.METRICS_PER_SERIES_HEADER}")
    metrics_table = build_per_series_metrics_dataframe(results)
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    # Forecasts table
    with st.expander("🔍 Переглянути числові прогнози по горизонтах (F1...Fh)"):
        fc_df = build_forecasts_dataframe(results)
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_download_tab(results: RunResults | None, eda_df: pd.DataFrame | None = None) -> None:
    """Render the Export & ZIP download tab."""
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.subheader("Завантаження результатів моделювання")
    st.caption(ua.DOWNLOAD_ALL_HELP)

    if results is None:
        st.button(
            f"📦 {ua.BTN_DOWNLOAD_ALL}",
            disabled=True,
            help="Доступно після успішного виконання моделювання",
            use_container_width=True,
)
        st.info("Виконайте прогнозування, щоб сформувати архів результатів.")
    else:
        zip_bytes = generate_results_zip(results=results, eda_df=eda_df)
        file_timestamp = results.timestamp.replace(":", "-").split(".")[0]
        zip_name = f"grnn_results_{results.mode}_{file_timestamp}.zip"

        st.download_button(
            label=f"📦 {ua.BTN_DOWNLOAD_ALL}",
            data=zip_bytes,
            file_name=zip_name,
            mime="application/zip",
            type="primary",
            use_container_width=True,
)

        st.success(
            f"✅ **Архів підготовлено:** `{zip_name}` ({len(zip_bytes) / 1024:.1f} KB). "
            "Містить прогнози, метрики, зведення та файл конфігурації `run_config.json`."
)

    st.markdown('</div>', unsafe_allow_html=True)
