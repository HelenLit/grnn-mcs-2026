"""M4-Yearly per-series evaluation harness.

For each sampled M4 series and each registry model key (`src/evaluation/comparison.py`):
  1. Take the official train part; optionally first-difference it.
  2. GRNN family + SVR + MLP ("lag-feature regressors"): build lag features
     via a sliding window, select sigma (and, for ensembles, n_models) on a
     within-train validation split (no peeking at held-out data), fit on
     the FULL lagged train set, then forecast the h-step horizon
     RECURSIVELY — one-step-ahead, feeding each prediction back into the lag
     window (`recursive_forecast_regressor`).
  3. ARIMA: fits directly on the (optionally differenced) train series and
     forecasts the whole horizon in one call — statsmodels handles multi-step
     natively, so it bypasses the lag-feature / recursive-loop machinery
     entirely (it is not a lag-feature regressor).
  4. If differencing was applied, invert the horizon forecast back to levels
     via cumulative sum anchored on the last raw training value
     (`invert_difference`).
  5. Score in level space: official sMAPE/MASE (src/evaluation/m4_metrics)
     plus the internal MAE/RMSE/MedAE/MAPE/R2 (src/evaluation/metrics). OWA is
     an aggregate-level statistic (needs a Naive2 denominator averaged over
     the same sample) and is computed by the caller, not per series.

Reuses `make_model`/`fit_model`/the sigma-selection dispatch from
`comparison.py` unchanged — no model code is forked between the synthetic
and real-data evaluation paths.

Design decision on pairing: every model is attempted on every series that
passes the minimal length gate (works for all model types uniformly). Beyond
that, an ensemble's configured M values may all be infeasible for a very
short series (`decompose_into_subsequences` requires n_models <= available
samples at every internal validation split) — such (series, model) cells are
skipped and logged, not silently dropped. This means per-model result counts
can differ slightly; the paired significance tests must restrict to the
intersection of series with a valid row for every model being compared.
"""
import math
import time
from typing import Any
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.data.m4 import M4Series
from src.data.preparation import create_sliding_window
from src.data.preparation import difference_series
from src.data.preparation import fit_feature_pipeline
from src.data.preparation import invert_difference
from src.evaluation.comparison import ENSEMBLE_MODEL_KEYS
from src.evaluation.comparison import build_sigma_grid
from src.evaluation.comparison import compute_validation_rmse
from src.evaluation.comparison import fit_model
from src.evaluation.comparison import make_model
from src.evaluation.comparison import model_needs_sigma
from src.evaluation.comparison import select_sigma_dispatch
from src.evaluation.comparison import sigma_config_key
from src.evaluation.metrics import compute_metrics
from src.evaluation.m4_metrics import mase
from src.evaluation.m4_metrics import smape

# Exceptions treated as "this (series, model) cell is infeasible" rather than
# a harness bug: tiny M4 series routinely produce singular feature matrices,
# non-convergent ARIMA fits, or decomposition-size assertions.
INFEASIBLE_ERRORS = (AssertionError, ValueError, np.linalg.LinAlgError, RuntimeError)
GRNN_FAMILY_MODEL_KEYS = ("single_grnn", "grnn_avg", "grnn_robust", "grnn_bagging")


class SkippedCell(NamedTuple):
    """One logged (series, model) cell that could not be evaluated."""
    series_id: str
    model_key: str
    reason: str


def score_forecasts(
    forecasts: dict[str, np.ndarray],
    series_list: list[M4Series],
    source: str,
) -> pd.DataFrame:
    """Score point forecasts for loaded M4 series with the official metrics.

    This intentionally does not know whether `forecasts` came from our GRNN
    runner or an official M4 submission. OWA remains an aggregate-level metric
    and is computed by the caller over identical series subsets.
    """
    rows = []
    for series in series_list:
        if series.series_id not in forecasts:
            raise KeyError(f"{source}: missing forecast for {series.series_id}")

        forecast = np.asarray(forecasts[series.series_id], dtype=np.float64)
        if len(forecast) != len(series.test):
            raise ValueError(
                f"{source}/{series.series_id}: forecast length {len(forecast)} "
                f"!= test length {len(series.test)}"
            )

        rows.append({
            "series_id": series.series_id,
            "domain": series.category,
            "source": source,
            "smape": smape(series.test, forecast),
            "mase": mase(series.train, series.test, forecast, series.m),
        })

    return pd.DataFrame(
        rows,
        columns=["series_id", "domain", "source", "smape", "mase"],
    )


def build_working_series(
    train: np.ndarray, difference: bool
) -> tuple[np.ndarray, float | None]:
    """Returns the series a model should train on, plus the inversion anchor.

    Args:
        train: Official training values of one M4 series.
        difference: Whether to first-difference.

    Returns:
        (working_series, anchor). anchor is None when difference is False;
        otherwise it is the last RAW training level, needed to invert a
        difference-space forecast horizon back to levels.
    """
    if not difference:
        return train, None
    diffs, _ = difference_series(train)
    return diffs, float(train[-1])


def compute_clip_bounds(
    train_y: np.ndarray, margin_factor: float = 3.0
) -> tuple[float, float]:
    """Sane forecast bounds derived from the training target's robust spread.

    Defense-in-depth against recursive-forecast blowup: a regressor with an
    unbounded output layer (MLP) can predict a wild value that, fed back
    into the lag window, compounds across the remaining recursive steps into
    a numerical explosion (observed on an unclipped M4 pilot run: MASE ~4.8e9
    on 2/100 series). GRNN (kernel-weighted average of training targets, always
    within [min(train_y), max(train_y)]) and SVR with an RBF kernel are
    naturally near-bounded already, so this clip is close to a no-op for
    them; it is the real guard for MLP.

    Uses median +/- margin_factor * IQR (Tukey's "far outlier" fence at the
    default margin_factor=3.0), NOT min/max: on a short, volatile M4 series
    the single largest observed difference can itself be a genuine one-off
    swing (e.g. a late train-window jump of +800 on a series whose typical
    diff is ~50) — a min/max-derived bound would inherit that swing and
    become a near no-op exactly where clipping is needed most. The
    median/IQR bound stays tight even when a couple of training points are
    extreme, at the (deliberate) cost of not fully trusting a lone recent
    large swing to repeat every step of the recursive horizon.

    Args:
        train_y: The lag-windowed training target array the model was fit on
            (difference space if differencing was applied upstream).
        margin_factor: Multiples of the IQR allowed beyond the median before
            clipping. 3.0 = Tukey's far-outlier fence.

    Returns:
        (low, high) clip bounds.
    """
    median = float(np.median(train_y))
    q1, q3 = np.percentile(train_y, [25, 75])
    iqr = float(q3 - q1)
    if iqr == 0.0:
        # Degenerate (near-constant) training target — fall back to a margin
        # relative to the median's magnitude, or a unit margin at zero.
        iqr = abs(median) if median != 0.0 else 1.0
    margin = margin_factor * iqr
    return median - margin, median + margin


def recursive_forecast_regressor(
    model: object,
    pipeline,
    last_window: np.ndarray,
    horizon: int,
    clip_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """One-step-ahead recursive forecast for a lag-feature regressor.

    At each step, transforms the current raw lag window through the fitted
    `FeaturePipeline`, predicts one value, appends it to the window (so the
    next step's window is the same length, shifted by one), and repeats.
    Used for every lag-feature model (GRNN family, SVR, MLP) — not for ARIMA.

    Args:
        model: A fitted registry model exposing `.predict(X) -> array`.
        pipeline: `FeaturePipeline` fitted on the training lag matrix.
        last_window: The last `lag_order` raw values of the working series.
        horizon: Number of steps to forecast.
        clip_bounds: Optional (low, high) — each step's prediction is clipped
            into this range BEFORE being appended to the window, so a wild
            prediction cannot compound across subsequent recursive steps
            (see `compute_clip_bounds`). None disables clipping.

    Returns:
        Array of `horizon` forecasts, in the same space as `last_window`
        (difference space if differencing was applied upstream).
    """
    lag_order = len(last_window)
    window = list(last_window)
    forecasts = np.empty(horizon, dtype=np.float64)
    for step in range(horizon):
        raw_row = np.array(window[-lag_order:], dtype=np.float64).reshape(1, -1)
        X_row = pipeline.transform(raw_row)
        y_hat = float(np.asarray(model.predict(X_row)).reshape(-1)[0])
        if clip_bounds is not None:
            y_hat = min(max(y_hat, clip_bounds[0]), clip_bounds[1])
        forecasts[step] = y_hat
        window.append(y_hat)
    return forecasts


def select_model_config(
    model_key: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    config: dict[str, Any],
    validation_size: float,
) -> tuple[float, int | float] | None:
    """Selects sigma (and, for ensembles, n_models) via within-train validation.

    For ensembles, every configured M is scored by validation RMSE
    (`compute_validation_rmse`) and the best is kept; M values infeasible at
    this series' length (raises `AssertionError` inside decomposition) are
    skipped, not treated as errors.

    **Selection runs in the DEPLOYMENT feature space.** Sigma is a kernel
    width, so it is only meaningful on the same scale as the distances the
    fitted model will actually see. Selecting on raw lag windows while
    deploying on MaxAbs-scaled features makes the objective flat: on
    M4-scale data the Gaussian kernel underflows to exactly 0.0 for every
    sigma in [0.1, 10] (median inter-point distance ~205 raw vs ~0.88
    scaled), so an optimizer would return its tie-breaking convention rather
    than a minimum on most series. The scaler is fitted on the FIT split
    alone, so the held-out validation windows never influence the transform
    that judges them. Same rule as `select_config_with_method` in
    `compare_sigma_optimizers`.

    Args:
        model_key: Registry model key.
        train_X: Lag-windowed training features, raw (pre-scaling); this
            function applies the deployment transform itself.
        train_y: Lag-windowed training targets.
        config: Full loaded `config.yaml`.
        validation_size: Fraction held out for sigma/M selection.

    Returns:
        (sigma, n_models) with n_models = NaN for non-ensemble models, or
        None if no configured M is feasible for this series length.
    """
    if not model_needs_sigma(model_key):
        return float("nan"), float("nan")

    # Same temporal split rule as `compute_validation_rmse` and
    # `_temporal_split`: the transform must not see the windows it judges.
    n_fit = len(train_y) - math.ceil(len(train_y) * validation_size)
    if n_fit < 1:
        return None
    _, selection_pipeline = fit_feature_pipeline(
        train_X[:n_fit], config["preparation"]["scaler"]
    )
    train_X = selection_pipeline.transform(train_X)

    sigma_grid = build_sigma_grid(config[sigma_config_key(model_key)])

    if model_key not in ENSEMBLE_MODEL_KEYS:
        sigma = select_sigma_dispatch(
            model_key=model_key,
            n_models=1,
            train_X=train_X,
            train_y=train_y,
            sigma_values=sigma_grid,
            config=config,
        )
        return sigma, float("nan")

    m_values = config[sigma_config_key(model_key)]["M_values"]
    best: tuple[float, int, float] | None = None
    for n_models in m_values:
        try:
            sigma = select_sigma_dispatch(
                model_key=model_key,
                n_models=n_models,
                train_X=train_X,
                train_y=train_y,
                sigma_values=sigma_grid,
                config=config,
            )
            val_rmse = compute_validation_rmse(
                model_key=model_key,
                n_models=int(n_models),
                sigma=sigma,
                train_X=train_X,
                train_y=train_y,
                validation_size=validation_size,
                config=config,
            )
        except INFEASIBLE_ERRORS:
            continue
        if best is None or val_rmse < best[2]:
            best = (sigma, n_models, val_rmse)

    if best is None:
        return None
    sigma, n_models, _ = best
    return sigma, n_models


def evaluate_series(
    series: M4Series,
    model_key: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Evaluates one (series, model) pair. Returns None if infeasible.

    Args:
        series: One M4 series with its official train/test split.
        model_key: Registry model key (`src.evaluation.comparison.MODEL_KEYS`).
        config: Full loaded `config.yaml`.

    Returns:
        A result dict (series_id, model, n_models, sigma, sMAPE, MASE, our
        metrics, wall clock), or None if the series is too short or no
        configured ensemble M is feasible at this series' length.
    """
    m4_config = config["realdata"]["m4_yearly"]
    prep_config = config["preparation"]
    horizon = m4_config["horizon"]
    lag_order = prep_config["lag_order"]
    difference = prep_config["difference"]
    validation_size = config["evaluation"]["validation_size"]

    working_series, anchor = build_working_series(series.train, difference)
    if len(working_series) <= lag_order:
        return None

    started = time.perf_counter()

    if model_key == "arima":
        sigma, n_models = float("nan"), float("nan")
        model = make_model(model_key, n_models=n_models, sigma=sigma, config=config)
        # ARIMAWrapper.fit only reads train_y; ARIMAWrapper.predict only reads
        # len(test_X) (see src/models/baselines.py) — both placeholders below
        # are shape-only, never inspected for content.
        fit_model(model, np.empty((len(working_series), 1)), working_series)
        raw_forecast = np.asarray(
            model.predict(np.empty((horizon, 1)))
        ).reshape(-1)
    else:
        X, y = create_sliding_window(working_series, lag_order)
        selected = select_model_config(model_key, X, y, config, validation_size)
        if selected is None:
            return None
        sigma, n_models = selected

        X_transformed, pipeline = fit_feature_pipeline(X, prep_config["scaler"])
        model = make_model(model_key, n_models=n_models, sigma=sigma, config=config)
        fit_model(model, X_transformed, y)

        last_window = working_series[-lag_order:]
        clip_bounds = compute_clip_bounds(y)
        raw_forecast = recursive_forecast_regressor(
            model, pipeline, last_window, horizon, clip_bounds=clip_bounds
        )

    if difference:
        levels = invert_difference(raw_forecast, anchor)[1:]
    else:
        levels = raw_forecast

    elapsed = time.perf_counter() - started
    own_metrics = compute_metrics(series.test, levels)

    return {
        "series_id": series.series_id,
        "model": model_key,
        "n_models": n_models,
        "sigma": sigma,
        "train_length": len(series.train),
        "smape": smape(series.test, levels),
        "mase": mase(series.train, series.test, levels, m4_config["mase_m"]),
        **own_metrics,
        "wall_clock_s": elapsed,
    }


def evaluate_grnn_series(
    series: M4Series,
    model_key: str,
    config: dict[str, Any],
    difference: bool,
    bagging_seed: int | None = None,
    selected_config: tuple[float, int | float] | None = None,
) -> dict[str, Any] | None:
    """Evaluate one M4 series with one GRNN-family model.

    Unlike `evaluate_series`, it never reads
    `config["preparation"]["difference"]`; the caller passes the
    differencing variant explicitly so both off/on rows are always produced.
    """
    if model_key not in GRNN_FAMILY_MODEL_KEYS:
        raise ValueError(f"'{model_key}' is not a GRNN-family model")

    m4_config = config["realdata"]["m4_yearly"]
    prep_config = config["preparation"]
    horizon = m4_config["horizon"]
    lag_order = prep_config["lag_order"]
    validation_size = config["evaluation"]["validation_size"]

    working_series, anchor = build_working_series(series.train, difference)
    if len(working_series) <= lag_order:
        return None

    started = time.perf_counter()
    X, y = create_sliding_window(working_series, lag_order)
    selected = selected_config
    if selected is None:
        selected = select_model_config(model_key, X, y, config, validation_size)
    if selected is None:
        return None
    sigma, n_models = selected

    X_transformed, pipeline = fit_feature_pipeline(X, prep_config["scaler"])

    model_config = config
    if model_key == "grnn_bagging" and bagging_seed is not None:
        model_config = {
            **config,
            "ensemble_bagging": {
                **config["ensemble_bagging"],
                "random_seed": int(bagging_seed),
            },
        }

    model = make_model(model_key, n_models=n_models, sigma=sigma, config=model_config)
    fit_model(model, X_transformed, y)

    raw_forecast = recursive_forecast_regressor(
        model,
        pipeline,
        working_series[-lag_order:],
        horizon,
        clip_bounds=compute_clip_bounds(y),
    )

    if difference:
        levels = invert_difference(raw_forecast, anchor)[1:]
    else:
        levels = raw_forecast

    elapsed = time.perf_counter() - started
    own_metrics = compute_metrics(series.test, levels)
    result = {
        "series_id": series.series_id,
        "domain": series.category,
        "model": model_key,
        "diff": "on" if difference else "off",
        "n_train": len(series.train),
        "sigma": sigma,
        "M": n_models,
        "seed": bagging_seed if bagging_seed is not None else config["realdata"]["m4_yearly"]["random_seed"],
        "runtime_ms": elapsed * 1000.0,
        "smape": smape(series.test, levels),
        "mase": mase(series.train, series.test, levels, series.m),
        **own_metrics,
    }
    for index, value in enumerate(levels, start=1):
        result[f"F{index}"] = float(value)
    return result


def run_m4_harness(
    series_list: list[M4Series],
    model_keys: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[SkippedCell]]:
    """Evaluates every (series, model) pair, in a paired series x model design.

    Args:
        series_list: Sampled M4 series (same list offered to every model).
        model_keys: Registry model keys to evaluate.
        config: Full loaded `config.yaml`.

    Returns:
        (results_frame, skipped) where `skipped` logs every (series, model)
        cell that could not be evaluated and why — no silent truncation.
    """
    rows = []
    skipped: list[SkippedCell] = []
    for series in series_list:
        for model_key in model_keys:
            try:
                result = evaluate_series(series, model_key, config)
            except INFEASIBLE_ERRORS as error:
                skipped.append(SkippedCell(series.series_id, model_key, repr(error)))
                continue
            if result is None:
                skipped.append(
                    SkippedCell(series.series_id, model_key, "infeasible (length/M)")
                )
                continue
            rows.append(result)

    columns = [
        "series_id", "model", "n_models", "sigma", "train_length",
        "smape", "mase", "MAE", "RMSE", "MedAE", "MAPE", "R2", "wall_clock_s",
    ]
    return pd.DataFrame(rows, columns=columns), skipped
