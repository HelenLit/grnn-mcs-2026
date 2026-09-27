"""Run Engine for GRNN Forecasting Demo UI.

Executes the official forecasting and evaluation pipeline per series:
  differencing (always ON) -> MaxAbs feature scaling -> sigma selection with the
  chosen optimizer in deployment space -> interleaved ensemble -> aggregation ->
  recursive multi-step forecasting with Tukey IQR clipping -> invert differencing ->
  scoring (sMAPE, MASE, OWA via Naive2, MAE, RMSE).

Supports live progress updates, cancellation, and atomic result replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.data.preparation import (
    create_sliding_window,
    fit_feature_pipeline,
    invert_difference,
)
from src.evaluation.comparison import fit_model
from src.evaluation.m4_harness import (
    compute_clip_bounds,
    recursive_forecast_regressor,
)
from src.evaluation.m4_metrics import (
    mase,
    naive2_forecast_yearly,
    owa,
    smape,
)
from src.evaluation.metrics import compute_metrics
from src.models.grnn import GRNN
from src.models.grnn_ensemble_avg import GRNNEnsembleAvg
from src.models.grnn_ensemble_robust import GRNNRobustEnsemble
from src.models.sigma_optimizers import ALL_METHODS
from src.ui.data_loader import SeriesData
from src.ui.parameters import UIParameters


@dataclass
class SeriesForecastResult:
    """Detailed forecast outcome and metrics for one time series.

    Fields:
        series_id: Identifier of the series.
        history: 1D array of historical training points in level space.
        forecast: 1D array of forecasted level values over the horizon.
        actual: 1D array of ground truth test values (or None in forecast mode).
        sigma: Optimized smoothing parameter sigma.
        category: Domain or category label.
        m: Seasonality period (1 for Yearly).
        smape: sMAPE score in percent (or None in forecast mode).
        mase: MASE score (or None in forecast mode).
        mae: Mean Absolute Error (or None in forecast mode).
        rmse: Root Mean Squared Error (or None in forecast mode).
        medae: Median Absolute Error (or None in forecast mode).
        mape: Mean Absolute Percentage Error (or None in forecast mode).
        naive2_smape: Baseline Naive2 sMAPE on the same holdout (or None).
        naive2_mase: Baseline Naive2 MASE on the same holdout (or None).
        runtime_s: Execution wall clock time in seconds for this series.
    """
    series_id: str
    history: np.ndarray
    forecast: np.ndarray
    actual: np.ndarray | None = None
    sigma: float = 1.0
    category: str = ""
    m: int = 1
    smape: float | None = None
    mase: float | None = None
    mae: float | None = None
    rmse: float | None = None
    medae: float | None = None
    mape: float | None = None
    naive2_smape: float | None = None
    naive2_mase: float | None = None
    runtime_s: float = 0.0


@dataclass
class RunResults:
    """Atomic container for the entire forecasting run results and metadata.

    Fields:
        parameters: UIParameters configuration snapshot.
        mode: "evaluation" or "forecast".
        is_m4: Whether run was executed on M4 Yearly benchmark.
        series_results: List of per-series results.
        aggregate_metrics: Summary metrics across all series (sMAPE, MASE, OWA, etc.).
        naive2_aggregate: Aggregate Naive2 benchmark scores.
        total_runtime_s: Total elapsed wall clock time in seconds.
        timestamp: ISO format string timestamp of run completion.
        dataset_info: Metadata dictionary describing active dataset.
    """
    parameters: UIParameters
    mode: str
    is_m4: bool
    series_results: list[SeriesForecastResult]
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    naive2_aggregate: dict[str, float] = field(default_factory=dict)
    total_runtime_s: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_info: dict[str, Any] = field(default_factory=dict)


def _construct_model(models: int, aggregation: str, sigma: float) -> object:
    """Build GRNN or GRNN Ensemble instance."""
    if models == 1:
        return GRNN(sigma=sigma)
    if aggregation == "median":
        return GRNNRobustEnsemble(n_models=models, sigma=sigma)
    if aggregation == "mean":
        return GRNNEnsembleAvg(n_models=models, sigma=sigma)
    raise ValueError(f"Unknown aggregation method: '{aggregation}'")


def forecast_single_series(
    series: SeriesData,
    params: UIParameters,
    is_m4: bool = False,
) -> SeriesForecastResult:
    """Execute the full forecasting pipeline on a single time series.

    Args:
        series: SeriesData object containing historical values and optional test set.
        params: Configured UIParameters.
        is_m4: True if running on M4 Yearly dataset.

    Returns:
        SeriesForecastResult with forecasts, metrics, and selected sigma.
    """
    start_time = time.perf_counter()

    # 1. Determine train/test split and horizon
    if is_m4:
        train_values = np.asarray(series.values, dtype=np.float64)
        test_actual = np.asarray(series.test, dtype=np.float64) if series.test is not None else None
        horizon = 6
        eval_mode = True
    else:
        horizon = params.horizon
        eval_mode = (params.mode == "evaluation")
        if eval_mode:
            train_values = np.asarray(series.values[:-horizon], dtype=np.float64)
            test_actual = np.asarray(series.values[-horizon:], dtype=np.float64)
        else:
            train_values = np.asarray(series.values, dtype=np.float64)
            test_actual = None

    # 2. First differencing (always ON)
    diffs = np.diff(train_values)
    anchor = float(train_values[-1])

    # 3. Sliding window lag embedding
    raw_X, train_y = create_sliding_window(diffs, lag_order=params.lags)

    # 4. Feature scaling in deployment space (MaxAbsScaler)
    scaled_train_X, pipeline = fit_feature_pipeline(raw_X, scaler_type="MaxAbsScaler")

    # 5. Sigma Optimization in deployment scaled space
    opt_selector = ALL_METHODS[params.optimizer]
    agg_name = "single" if params.models == 1 else params.aggregation

    search_res = opt_selector(
        scaled_train_X,
        train_y,
        n_models=params.models,
        aggregate=agg_name,
        sigma_bounds=(0.1, 10.0),
        validation_size=0.2,
)
    selected_sigma = float(search_res.sigma)

    # 6. Fit Model
    model = _construct_model(
        models=params.models,
        aggregation=params.aggregation,
        sigma=selected_sigma,
)
    fit_model(model, scaled_train_X, train_y)

    # 7. Recursive Multi-step Forecast with Tukey IQR clipping bounds
    clip_bounds = compute_clip_bounds(train_y)
    last_window = diffs[-params.lags:]
    raw_diff_forecast = recursive_forecast_regressor(
        model=model,
        pipeline=pipeline,
        last_window=last_window,
        horizon=horizon,
        clip_bounds=clip_bounds,
)

    # 8. Invert differencing back to level space
    level_forecast = invert_difference(raw_diff_forecast, anchor)[1:]

    # 9. Compute metrics in Evaluation mode
    s_smape: float | None = None
    s_mase: float | None = None
    s_mae: float | None = None
    s_rmse: float | None = None
    s_medae: float | None = None
    s_mape: float | None = None
    naive2_s: float | None = None
    naive2_m: float | None = None

    if eval_mode and test_actual is not None:
        s_smape = smape(test_actual, level_forecast)
        s_mase = mase(insample=train_values, y_true=test_actual, y_pred=level_forecast, m=series.m)
        own_metrics = compute_metrics(test_actual, level_forecast)
        s_mae = own_metrics["MAE"]
        s_rmse = own_metrics["RMSE"]
        s_medae = own_metrics["MedAE"]
        s_mape = own_metrics["MAPE"]

        # Baseline Naive2 forecast on the same holdout for OWA computation
        naive2_fc = naive2_forecast_yearly(train_values, horizon=horizon)
        naive2_s = smape(test_actual, naive2_fc)
        naive2_m = mase(insample=train_values, y_true=test_actual, y_pred=naive2_fc, m=series.m)

    elapsed_s = time.perf_counter() - start_time

    return SeriesForecastResult(
        series_id=series.series_id,
        history=train_values,
        forecast=level_forecast,
        actual=test_actual,
        sigma=selected_sigma,
        category=series.category,
        m=series.m,
        smape=s_smape,
        mase=s_mase,
        mae=s_mae,
        rmse=s_rmse,
        medae=s_medae,
        mape=s_mape,
        naive2_smape=naive2_s,
        naive2_mase=naive2_m,
        runtime_s=elapsed_s,
)


def execute_run(
    series_list: list[SeriesData],
    params: UIParameters | None = None,
    parameters: UIParameters | None = None,
    is_m4: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    dataset_info: dict[str, Any] | None = None,
) -> RunResults:
    """Run the batch forecasting pipeline across all series.

    Guarantees atomic result creation: only completes and returns
    RunResults if all series succeed and no cancellation is requested.

    Args:
        series_list: List of active SeriesData objects.
        params: UIParameters instance (or alias parameters).
        parameters: Alias for params.
        is_m4: True if M4 dataset.
        progress_callback: Callback func(current_idx_1based, total_count, series_id).
        cancel_check: Callable returning True if cancellation was requested.
        dataset_info: Optional metadata dict.

    Returns:
        RunResults dataclass container.

    Raises:
        InterruptedError: If run was cancelled mid-execution.
        RuntimeError: If execution fails on any series.
    """
    active_params = params if params is not None else parameters
    if active_params is None:
        raise ValueError("Must provide params or parameters to execute_run.")

    total_series = len(series_list)
    if total_series == 0:
        raise ValueError("Cannot execute run on an empty dataset.")

    start_run_time = time.perf_counter()
    results: list[SeriesForecastResult] = []

    for i, s in enumerate(series_list):
        if cancel_check is not None and cancel_check():
            raise InterruptedError(f"Обчислення скасовано користувачем на ряді {i + 1} з {total_series}.")

        if progress_callback is not None:
            progress_callback(i + 1, total_series, s.series_id)

        try:
            res = forecast_single_series(series=s, params=active_params, is_m4=is_m4)
            results.append(res)
        except Exception as e:
            raise RuntimeError(f"Помилка при прогнозуванні ряду '{s.series_id}': {e}") from e

    total_elapsed = time.perf_counter() - start_run_time

    # Calculate aggregate metrics if in evaluation mode
    aggregate_metrics: dict[str, float] = {}
    naive2_aggregate: dict[str, float] = {}
    eval_mode = (is_m4 or active_params.mode == "evaluation")

    if eval_mode and results:
        valid_smapes = [r.smape for r in results if r.smape is not None]
        valid_mases = [r.mase for r in results if r.mase is not None]
        valid_maes = [r.mae for r in results if r.mae is not None]
        valid_rmses = [r.rmse for r in results if r.rmse is not None]
        valid_n2_smapes = [r.naive2_smape for r in results if r.naive2_smape is not None]
        valid_n2_mases = [r.naive2_mase for r in results if r.naive2_mase is not None]

        mean_smape = float(np.mean(valid_smapes)) if valid_smapes else 0.0
        mean_mase = float(np.mean(valid_mases)) if valid_mases else 0.0
        mean_mae = float(np.mean(valid_maes)) if valid_maes else 0.0
        mean_rmse = float(np.mean(valid_rmses)) if valid_rmses else 0.0

        n2_smape = float(np.mean(valid_n2_smapes)) if valid_n2_smapes else 0.0
        n2_mase = float(np.mean(valid_n2_mases)) if valid_n2_mases else 0.0

        agg_owa = float(owa(mean_smape, mean_mase, n2_smape, n2_mase)) if (n2_smape > 0 and n2_mase > 0) else 1.0

        aggregate_metrics = {
            "sMAPE": mean_smape,
            "MASE": mean_mase,
            "OWA": agg_owa,
            "MAE": mean_mae,
            "RMSE": mean_rmse,
        }
        naive2_aggregate = {
            "sMAPE": n2_smape,
            "MASE": n2_mase,
        }

    return RunResults(
        parameters=active_params,

        mode="evaluation" if eval_mode else "forecast",
        is_m4=is_m4,
        series_results=results,
        aggregate_metrics=aggregate_metrics,
        naive2_aggregate=naive2_aggregate,
        total_runtime_s=total_elapsed,
        dataset_info=dataset_info or {},
)
