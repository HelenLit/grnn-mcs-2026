"""Sigma-optimizer comparison for the FULL GRNN family on M4 Yearly.

Compares five sigma-selection procedures (grid, gradient, brent,
coarse2fine, silverman) for the full GRNN family on the M4 Yearly
subset. Sigma AND the ensemble size M are selected jointly in the SAME
deployment feature space the fitted model forecasts in
(`fit_feature_pipeline`, fitted on the fit split only — no validation
leakage): sigma is a kernel width, so it is only meaningful on the scale
of the distances the deployed model actually computes. On raw M4-scale
distances the Gaussian kernel underflows for every sigma in [0.1, 10],
which makes the validation objective numerically flat and reduces every
"optimizer" to its tie-breaking convention. The script therefore:

  * logs per-cell objective diagnostics (flatness probe, kernel-underflow
    fraction at the selected sigma) so a degenerate objective cannot
    pass silently;
  * reports the effective Wilcoxon n (non-tied pairs) and the SIGNED
    median sigma shift, not only |delta sigma|;
  * can optionally include reference rows from previously stored pipeline
    runs (--grid-tag / --gradient-tag; rows named `grid_frozen` /
    `gradient_frozen`) — these are integrity-hashed, never re-executed,
    and gated on their stored aggregate OWA. On a fresh clone with no
    stored runs, use --no-frozen.

Stages:
  [load-frozen]     frozen reference rows + integrity hashes + OWA gate.
  [run-new]         grid/gradient/brent/coarse2fine/silverman on the full
                    23,000 Yearly series, both differencing variants, joint
                    (sigma, M) selection in deployment space (bagging keeps
                    5-seed averaging). Resumable at
                    (series x model x method x diff) granularity.
  [microbenchmark]  all five methods timed on a fixed 500-series subsample.
  [stats]           OWA per domain, paired Wilcoxon on MASE vs fresh grid
                    (Holm), sigma agreement vs fresh grid.
  [report]          CSV tables + markdown report + plots + manifest.

Usage:
    python -m src.scripts.compare_sigma_optimizers --tag sigma_opt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from src.data.m4 import M4Series
from src.data.m4 import load_m4_csv
from src.data.preparation import create_sliding_window
from src.data.preparation import fit_feature_pipeline
from src.evaluation.comparison import compute_validation_rmse
from src.evaluation.comparison import sigma_config_key
from src.evaluation.m4_harness import INFEASIBLE_ERRORS
from src.evaluation.m4_harness import build_working_series
from src.evaluation.m4_harness import evaluate_grnn_series
from src.evaluation.m4_metrics import mase
from src.evaluation.m4_metrics import smape
from src.models.sigma_optimizers import ALL_METHODS
from src.models.sigma_optimizers import SigmaSearchResult
from src.models.sigma_optimizers import diagnose_objective

REPO_ROOT = Path(__file__).resolve().parents[2]
M4_RAW_DIR = REPO_ROOT / "data" / "raw" / "m4"
M4_RESULTS_DIR = REPO_ROOT / "reports" / "results" / "m4"
FINAL_DIR = REPO_ROOT / "reports" / "final"
PLOTS_DIR = REPO_ROOT / "plots" / "m4"

MODEL_KEYS = ("single_grnn", "grnn_avg", "grnn_robust", "grnn_bagging")
MODEL_LABELS = {
    "single_grnn": "GRNN",
    "grnn_avg": "GRNN-Mean",
    "grnn_robust": "GRNN-Median",
    "grnn_bagging": "GRNN-Bag",
}
# Objective aggregation per model — mirrors select_sigma_dispatch: bagging
# selects sigma on the plain single-GRNN objective, like the frozen runs.
MODEL_AGGREGATE = {
    "single_grnn": "single",
    "grnn_avg": "mean",
    "grnn_robust": "median",
    "grnn_bagging": "single",
}
BAGGING_RUNS = 5  # bagging rows are averages over 5 whole runs
# Fresh methods: executed by THIS script, sigma selected in deployment
# feature space. Frozen references: loaded from previously stored runs
# (raw-space selection) — quality columns only, never re-executed.
FRESH_METHOD_ORDER = ("grid", "gradient", "brent", "coarse2fine", "silverman")
FROZEN_METHODS = ("grid_frozen", "gradient_frozen")
METHOD_ORDER = FROZEN_METHODS + FRESH_METHOD_ORDER
# Methods actually present in THIS run. Reassigned once in main() when the
# frozen reference CSVs are absent (--no-frozen); every stats/report/plot
# helper reads this instead of the static METHOD_ORDER.
ACTIVE_METHODS: tuple[str, ...] = METHOD_ORDER
METHOD_COLORS = {
    "grid_frozen": "#9a9a9a", "gradient_frozen": "#c8c8c8",  # gray = frozen
    "grid": "#4878a8", "gradient": "#e1812c", "brent": "#559e55",
    "coarse2fine": "#b04a4a", "silverman": "#8064a2",
}
BASELINE_METHOD = "grid"  # fresh grid — same feature space as every rival
DOMAIN_ORDER = ["Micro", "Finance", "Macro", "Industry", "Other", "Demographic", "All"]
RESULT_COLUMNS = (
    "series_id", "domain", "model", "method", "diff", "n_train",
    "sigma", "M", "val_mse", "n_evals", "objective_flat", "underflow_frac",
    "smape", "mase", "runtime_ms",
)
SKIP_COLUMNS = ("series_id", "domain", "model", "method", "diff", "reason")

plt.rcParams.update({"figure.dpi": 300, "font.size": 9, "svg.hashsalt": "m4"})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    with open(REPO_ROOT / "src" / "config" / "config.yaml") as file_obj:
        return yaml.safe_load(file_obj)


def load_yearly_series() -> list[M4Series]:
    return load_m4_csv(
        train_path=M4_RAW_DIR / "Train" / "Yearly-train.csv",
        test_path=M4_RAW_DIR / "Test" / "Yearly-test.csv",
        info_path=M4_RAW_DIR / "M4-info.csv",
    )


def append_row(path: Path, columns: tuple[str, ...], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(columns))
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def existing_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["series_id", "model", "method", "diff"])
    return {
        (str(r.series_id), str(r.model), str(r.method), str(r.diff))
        for r in frame.itertuples(index=False)
    }


class SelectionOutcome:
    """One completed (sigma, M) selection with its cost bookkeeping."""

    def __init__(
        self, search: SigmaSearchResult, n_models: float,
        n_calls: int, total_evals: int, select_seconds: float,
        objective_flat: bool | None = None,
        underflow_frac: float | None = None,
    ) -> None:
        self.search = search
        self.n_models = n_models  # NaN for single (no M sweep)
        self.n_calls = n_calls
        self.total_evals = total_evals
        # Pure sigma-search wall time (selector calls only, excluding the
        # M-validation gate) — the microbenchmark's per-call numerator.
        self.select_seconds = select_seconds
        # Objective health at the selected (sigma, M) —
        # None when diagnostics were skipped (microbenchmark timing path).
        self.objective_flat = objective_flat
        self.underflow_frac = underflow_frac


def select_config_with_method(
    method: str,
    model_key: str,
    working_series: np.ndarray,
    config: dict[str, Any],
    diagnose: bool = True,
) -> SelectionOutcome | None:
    """Joint (sigma, M) selection in the DEPLOYMENT feature space.

    `evaluate_grnn_series` forecasts on MaxAbs-scaled features, so selecting
    sigma on raw lag windows would optimize a geometry the deployed model
    never uses (and the kernel underflows on raw M4-scale distances). The
    windows are therefore transformed with the SAME `fit_feature_pipeline`
    the harness applies at forecast time — fitted on the fit split only, so
    the held-out validation windows never influence the transform — and both
    the sigma search and the `compute_validation_rmse` M-gate operate on the
    transformed matrix.

    Ensembles: the optimizer proposes sigma for each configured M, and M is
    chosen by the same `compute_validation_rmse` gate the frozen runs used.
    single_grnn / grnn_bagging: one selector call on the 'single' objective.
    Returns None when the series is too short or no M is feasible.
    """
    prep_config = config["preparation"]
    lag_order = prep_config["lag_order"]
    validation_size = config["evaluation"]["validation_size"]
    model_cfg = config[sigma_config_key(model_key)]
    bounds = (float(model_cfg["sigma_min"]), float(model_cfg["sigma_max"]))
    aggregate = MODEL_AGGREGATE[model_key]

    if len(working_series) <= lag_order:
        return None
    raw_X, train_y = create_sliding_window(working_series, lag_order)
    # Same temporal split rule as _temporal_split / compute_validation_rmse:
    # the pipeline must not see the validation windows it will judge.
    n_fit = len(train_y) - math.ceil(len(train_y) * validation_size)
    if n_fit < 1:
        return None
    _, sel_pipeline = fit_feature_pipeline(
        raw_X[:n_fit],
        prep_config["scaler"],
    )
    train_X = sel_pipeline.transform(raw_X)
    selector = ALL_METHODS[method]

    if model_key == "single_grnn":
        try:
            started = time.perf_counter()
            search = selector(
                train_X, train_y, n_models=1, aggregate="single",
                sigma_bounds=bounds, validation_size=validation_size,
            )
            select_seconds = time.perf_counter() - started
        except INFEASIBLE_ERRORS:
            return None
        flat, underflow = _maybe_diagnose(
            diagnose, train_X, train_y, 1, "single", search.sigma,
            bounds, validation_size,
        )
        return SelectionOutcome(
            search, float("nan"), 1, search.n_evals, select_seconds,
            flat, underflow,
        )

    # Ensembles (incl. bagging): sigma per candidate M on the model's own
    # objective; bagging's 'single' objective is M-independent, so one call
    # serves every M. M itself is chosen by the frozen-run validation gate.
    best: tuple[SigmaSearchResult, int, float] | None = None
    n_calls, total_evals = 0, 0
    select_seconds = 0.0
    shared_search: SigmaSearchResult | None = None
    for n_models in model_cfg["M_values"]:
        try:
            if aggregate == "single":
                if shared_search is None:
                    started = time.perf_counter()
                    shared_search = selector(
                        train_X, train_y, n_models=1, aggregate="single",
                        sigma_bounds=bounds, validation_size=validation_size,
                    )
                    select_seconds += time.perf_counter() - started
                    n_calls += 1
                    total_evals += shared_search.n_evals
                search = shared_search
            else:
                started = time.perf_counter()
                search = selector(
                    train_X, train_y,
                    n_models=int(n_models),
                    aggregate=aggregate,
                    sigma_bounds=bounds,
                    validation_size=validation_size,
                )
                select_seconds += time.perf_counter() - started
                n_calls += 1
                total_evals += search.n_evals
            val_rmse = compute_validation_rmse(
                model_key=model_key,
                n_models=int(n_models),
                sigma=search.sigma,
                train_X=train_X,
                train_y=train_y,
                validation_size=validation_size,
                config=config,
            )
        except INFEASIBLE_ERRORS:
            continue
        if best is None or val_rmse < best[2]:
            best = (search, int(n_models), val_rmse)
    if best is None:
        return None
    diag_models = 1 if aggregate == "single" else best[1]
    flat, underflow = _maybe_diagnose(
        diagnose, train_X, train_y, diag_models, aggregate, best[0].sigma,
        bounds, validation_size,
    )
    return SelectionOutcome(
        best[0], float(best[1]), n_calls, total_evals, select_seconds,
        flat, underflow,
    )


def _maybe_diagnose(
    diagnose: bool,
    train_X: np.ndarray,
    train_y: np.ndarray,
    n_models: int,
    aggregate: str,
    sigma: float,
    bounds: tuple[float, float],
    validation_size: float,
) -> tuple[bool | None, float | None]:
    """Objective health probe at the selected config (skipped when timing)."""
    if not diagnose:
        return None, None
    diag = diagnose_objective(
        train_X, train_y, n_models=int(n_models), aggregate=aggregate,
        sigma=sigma, sigma_bounds=bounds, validation_size=validation_size,
    )
    return diag.flat, diag.underflow_frac


def bagging_seeds(config: dict[str, Any]) -> list[int]:
    seed = int(config["realdata"]["m4_yearly"]["random_seed"])
    return [seed + 1000 * index for index in range(BAGGING_RUNS)]


def evaluate_cell(
    series: M4Series,
    model_key: str,
    difference: bool,
    outcome: SelectionOutcome,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Fit/forecast/score one cell; bagging averages 5 whole runs."""
    selected = (outcome.search.sigma, outcome.n_models)
    if model_key != "grnn_bagging":
        return evaluate_grnn_series(
            series=series, model_key=model_key, config=config,
            difference=difference, selected_config=selected,
        )
    seed_rows = []
    for seed in bagging_seeds(config):
        row = evaluate_grnn_series(
            series=series, model_key=model_key, config=config,
            difference=difference, bagging_seed=seed, selected_config=selected,
        )
        if row is None:
            return None
        seed_rows.append(row)
    # sMAPE/MASE are nonlinear in the forecast, so they must be recomputed
    # from the seed-averaged forecast rather than averaged per-seed.
    first = seed_rows[0].copy()
    forecast_columns = tuple(f"F{index}" for index in range(1, 7))
    avg_forecast = np.array(
        [np.mean([row[column] for row in seed_rows]) for column in forecast_columns],
        dtype=np.float64,
    )
    for column, value in zip(forecast_columns, avg_forecast):
        first[column] = float(value)
    first["smape"] = smape(series.test, avg_forecast)
    first["mase"] = mase(series.train, series.test, avg_forecast, series.m)
    return first


# ---------------------------------------------------------------------------
# Stage: run-new
# ---------------------------------------------------------------------------
def stage_run_new(
    series_list: list[M4Series],
    methods: list[str],
    config: dict[str, Any],
    tag: str,
) -> Path:
    results_path = M4_RESULTS_DIR / f"sigma_opt_results_{tag}.csv"
    skipped_path = M4_RESULTS_DIR / f"sigma_opt_skipped_{tag}.csv"
    completed = existing_keys(results_path)
    total = len(series_list) * len(methods) * len(MODEL_KEYS) * 2
    log("run-new", f"target: {len(series_list)} series x {len(methods)} methods "
                   f"x {len(MODEL_KEYS)} models x 2 diff = {total} cells "
                   f"({len(completed)} already done)")

    started = time.perf_counter()
    done_this_run = 0
    for method in methods:
        for model_key in MODEL_KEYS:
            for difference in (False, True):
                diff_label = "on" if difference else "off"
                for index, series in enumerate(series_list, start=1):
                    key = (series.series_id, model_key, method, diff_label)
                    if key in completed:
                        continue
                    cell_started = time.perf_counter()
                    try:
                        working, _ = build_working_series(series.train, difference)
                        outcome = select_config_with_method(
                            method, model_key, working, config
                        )
                        if outcome is None:
                            raise ValueError("infeasible (length/M)")
                        row = evaluate_cell(
                            series, model_key, difference, outcome, config
                        )
                        if row is None:
                            raise ValueError("infeasible (length/M)")
                        append_row(results_path, RESULT_COLUMNS, {
                            "series_id": series.series_id,
                            "domain": series.category,
                            "model": model_key,
                            "method": method,
                            "diff": diff_label,
                            "n_train": len(series.train),
                            "sigma": outcome.search.sigma,
                            "M": outcome.n_models,
                            "val_mse": outcome.search.val_mse,
                            "n_evals": outcome.total_evals,
                            "objective_flat": outcome.objective_flat,
                            "underflow_frac": outcome.underflow_frac,
                            "smape": row["smape"],
                            "mase": row["mase"],
                            "runtime_ms":
                                (time.perf_counter() - cell_started) * 1e3,
                        })
                        completed.add(key)
                        done_this_run += 1
                    except INFEASIBLE_ERRORS as error:
                        append_row(skipped_path, SKIP_COLUMNS, {
                            "series_id": series.series_id,
                            "domain": series.category,
                            "model": model_key,
                            "method": method,
                            "diff": diff_label,
                            "reason": repr(error),
                        })
                    if index % 2000 == 0:
                        elapsed = time.perf_counter() - started
                        rate = done_this_run / elapsed if elapsed > 0 else 0.0
                        remaining = total - len(completed)
                        eta = remaining / rate if rate > 0 else float("nan")
                        log("run-new", f"{method} {MODEL_LABELS[model_key]} "
                            f"diff={diff_label} {index}/{len(series_list)} "
                            f"({len(completed)}/{total} cells, "
                            f"{len(completed)/total:.0%}) · elapsed {elapsed:.0f}s"
                            f" · eta {eta:.0f}s")
    log("run-new", f"done: +{done_this_run} cells in "
                   f"{time.perf_counter() - started:.0f}s -> {results_path}")
    return results_path


# ---------------------------------------------------------------------------
# Stage: microbenchmark  (per-call inference time, ALL FIVE methods)
# ---------------------------------------------------------------------------
def stage_microbenchmark(
    series_list: list[M4Series],
    config: dict[str, Any],
    bench_size: int,
) -> pd.DataFrame:
    # Deterministic stride over the full set (no RNG) — domain-representative,
    # unlike a head slice (the first 500 Yearly IDs are all one domain).
    stride = max(1, len(series_list) // bench_size)
    subsample = series_list[::stride][:bench_size]
    log("microbenchmark", f"{len(subsample)} series (stride {stride}), diff=on, "
        f"{len(MODEL_KEYS)} models x {len(FRESH_METHOD_ORDER)} methods")
    rows = []
    for model_key in MODEL_KEYS:
        for method in FRESH_METHOD_ORDER:
            total_ms, call_ms, evals_per_call = [], [], []
            for series in subsample:
                try:
                    working, _ = build_working_series(series.train, True)
                    started = time.perf_counter()
                    outcome = select_config_with_method(
                        method, model_key, working, config, diagnose=False
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1e3
                    if outcome is None or outcome.n_calls == 0:
                        continue
                    total_ms.append(elapsed_ms)
                    call_ms.append(outcome.select_seconds * 1e3 / outcome.n_calls)
                    evals_per_call.append(outcome.total_evals / outcome.n_calls)
                except INFEASIBLE_ERRORS:
                    continue
            rows.append({
                "model": MODEL_LABELS[model_key],
                "method": method,
                "n_series": len(total_ms),
                "mean_call_ms": float(np.mean(call_ms)),
                "median_call_ms": float(np.median(call_ms)),
                "mean_selection_ms": float(np.mean(total_ms)),
                "mean_evals_per_call": float(np.mean(evals_per_call)),
            })
            log("microbenchmark",
                f"{MODEL_LABELS[model_key]:12s} {method:12s} "
                f"per-call {rows[-1]['mean_call_ms']:8.2f} ms · "
                f"full selection {rows[-1]['mean_selection_ms']:8.2f} ms · "
                f"evals/call {rows[-1]['mean_evals_per_call']:6.1f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage: stats
# ---------------------------------------------------------------------------
def load_frozen(tag: str, method: str) -> pd.DataFrame:
    frame = pd.read_csv(M4_RESULTS_DIR / f"grnn_results_{tag}.csv")
    frame = frame[frame["model"].isin(MODEL_KEYS)]
    frame = frame[["series_id", "domain", "model", "diff", "sigma", "M",
                   "smape", "mase"]]
    frame = frame.assign(method=method)
    log("load-frozen", f"{method}: {len(frame)} GRNN-family rows from tag '{tag}'")
    return frame


def load_naive2(
    series_list: list[M4Series],
    scores_path: Path = M4_RESULTS_DIR / "submission_scores_full_mb12_combined.csv",
) -> tuple[pd.DataFrame, str]:
    """Naive2 per-series scores (the OWA denominators) + their provenance.

    Preferred source: the official-submission scores CSV produced by
    `run_m4_pipeline --stages anchors` (anchor-gated against published OWA).
    Fallback when that file is absent: compute Naive2 directly from the raw
    data. For the Yearly subset this is exact, not an approximation — the
    seasonal adjustment of Naive2 is a no-op at period m=1, so the forecast
    is the last training value repeated across the horizon, scored with the
    same smape/mase code as everything else.
    """
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        frame = scores[scores["source"] == "Naive2"][
            ["series_id", "domain", "smape", "mase"]
        ]
        return frame, f"official submission scores ({scores_path.name})"
    log("load-frozen", f"{scores_path.name} not found — computing Naive2 "
        f"denominators from raw data (exact for Yearly: m=1, no seasonality)")
    rows = [{
        "series_id": s.series_id,
        "domain": s.category,
        "smape": smape(s.test, np.full(len(s.test), s.train[-1])),
        "mase": mase(s.train, s.test,
                     np.full(len(s.test), s.train[-1]), s.m),
    } for s in series_list]
    return pd.DataFrame(rows), "computed from raw data (Naive2 fallback)"


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def stage_stats(combined: pd.DataFrame, naive2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_key in MODEL_KEYS:
        for diff in ("on", "off"):
            subset = combined[
                (combined["model"] == model_key) & (combined["diff"] == diff)
            ]
            per_method = {
                m: g.set_index("series_id") for m, g in subset.groupby("method")
            }
            if set(per_method) != set(ACTIVE_METHODS):
                log("stats", f"{model_key} diff={diff}: methods missing "
                    f"({sorted(set(ACTIVE_METHODS) - set(per_method))}) — skipped")
                continue
            shared = set.intersection(*(set(g.index) for g in per_method.values()))
            log("stats", f"{MODEL_LABELS[model_key]} diff={diff}: paired "
                         f"intersection {len(shared)} series")
            ids = sorted(shared)
            n2 = naive2[naive2["series_id"].isin(shared)]
            grid_part = per_method[BASELINE_METHOD].loc[ids]

            for method in ACTIVE_METHODS:
                part = per_method[method].loc[ids]
                # Frozen references selected sigma on RAW features — those
                # sigmas live in a different geometry, so agreement columns
                # would be meaningless for them.
                sigma_comparable = method not in FROZEN_METHODS
                for domain in DOMAIN_ORDER:
                    if domain == "All":
                        p, g, n = part, grid_part, n2
                    else:
                        p = part[part["domain"] == domain]
                        g = grid_part[grid_part["domain"] == domain]
                        n = n2[n2["domain"] == domain]
                    if p.empty:
                        continue
                    owa = 0.5 * (p["smape"].mean() / n["smape"].mean()
                                 + p["mase"].mean() / n["mase"].mean())
                    owa_grid = 0.5 * (g["smape"].mean() / n["smape"].mean()
                                      + g["mase"].mean() / n["mase"].mean())
                    delta_mase = p["mase"].to_numpy() - g["mase"].to_numpy()
                    # Effective n: Wilcoxon discards zero differences, so the
                    # nominal n overstates the evidence when most pairs tie.
                    n_eff = int((delta_mase != 0.0).sum())
                    if method == BASELINE_METHOD or n_eff == 0:
                        p_value = 1.0
                    else:
                        _, p_value = wilcoxon(p["mase"], g["mase"])
                    sigma_delta = p["sigma"].to_numpy() - g["sigma"].to_numpy()
                    rows.append({
                        "model": MODEL_LABELS[model_key],
                        "domain": domain,
                        "method": method,
                        "diff": diff,
                        "n": len(p),
                        "n_eff": n_eff,
                        "owa": owa,
                        "delta_owa_vs_grid": owa - owa_grid,
                        "median_delta_mase": float(np.median(delta_mase)),
                        "p_value": p_value,
                        "sigma_changed_share": float(
                            (np.round(sigma_delta, 4) != 0.0).mean()
                        ) if sigma_comparable else np.nan,
                        "median_abs_dsigma": float(
                            np.median(np.abs(sigma_delta))
                        ) if sigma_comparable else np.nan,
                        "median_signed_dsigma": float(
                            np.median(sigma_delta)
                        ) if sigma_comparable else np.nan,
                        "flat_share": float(p["objective_flat"].mean())
                        if "objective_flat" in p else np.nan,
                        "mean_underflow": float(p["underflow_frac"].mean())
                        if "underflow_frac" in p else np.nan,
                    })
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    non_grid = table["method"] != BASELINE_METHOD
    groups = table[non_grid].groupby(["model", "domain", "diff"]).groups
    for _, idx in groups.items():
        table.loc[idx, "p_holm"] = holm(table.loc[idx, "p_value"].tolist())
    table.loc[~non_grid, "p_holm"] = 1.0
    table["significant"] = table["p_holm"] < 0.05
    return table


# ---------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------
THEORY = """\
**Selection space.** σ is a kernel width: it is only meaningful on the
scale of the distances the deployed model actually computes, and the
deployed model forecasts on MaxAbs-scaled features. On raw M4-scale
distances the Gaussian kernel underflows for every σ ∈ [0.1, 10], making
the validation objective numerically FLAT on most series — each
"optimizer" would merely report its tie-breaking convention
(grid → 0.1, brent → its first bracket edge 0.316, …). Both the σ
search and the M-gate therefore operate on the SAME deployment feature
space (`fit_feature_pipeline`, fitted on the fit split only — no
validation leakage). All five methods below are executed fresh in that
space; `grid_frozen` / `gradient_frozen` are optional reference rows from
previously stored raw-space runs — NOT optimizer baselines.

All fresh methods minimise the same objective — the validation MSE of the
model's own aggregation (single / mean / median; bagging selects on the
single-GRNN objective, as in the frozen runs) on the same temporal 80/20
split — and differ only in how the minimum over σ ∈ [0.1, 10] is located:

- **grid** — exhaustive 0.1-step scan (100 evals/call); the paired baseline.
- **gradient** — multi-start L-BFGS-B with the analytic dMSE/dsigma in
  log-sigma space. Scales to multi-parameter kernels.
- **brent** — multi-start bounded Brent (golden section + parabolic
  interpolation); derivative-free, hence formally correct for the median's
  piecewise-smooth objective.
- **coarse2fine** — 3-level refined grid (12 log points, two 9-point zooms).
- **silverman** — analytic bandwidth rule `c*s*n^(-1/(d+4))` on the
  transformed features; zero objective evaluations, the "no optimisation"
  reference.

Excluded as irrelevant here: Bayesian optimisation (the objective costs
milliseconds — surrogate-model overhead dominates), evolutionary/annealing
methods (200+ evals for a 1-D problem, RNG threatens determinism),
golden-section alone (dominated by Brent).

**Reading the frozen reference columns.** `grid_frozen` differences vs the
fresh methods conflate the selection feature space with the optimizer, so
they measure the effect of deployment-space selection, not optimizer
quality. A further caveat: the frozen grid run selected σ for `grnn_avg`
on the median-ensemble objective, which affects `grid_frozen` only; all
fresh methods are objective-consistent for every model.

**Per-cell diagnostics.** Every fresh cell logs whether the
objective was numerically flat across {0.1, 1, 10} and the fraction of
validation kernels that underflow at the selected σ; the per-model tables
report their averages. A healthy comparison requires flat-share ≈ 0.
"""


def fmt_combined_owa_table(table: pd.DataFrame, diff: str) -> list[str]:
    """Merged OWA-by-domain table, methods as columns, best per row in bold.

    A trailing `*` marks a Holm-significant paired difference vs the grid
    baseline (per-series Wilcoxon on MASE). Ties within 5e-5 are all bolded.
    """
    lines = [
        "| Model | Domain | " + " | ".join(ACTIVE_METHODS) + " |",
        "| --- | --- |" + " --- |" * len(ACTIVE_METHODS),
    ]
    subset = table[table["diff"] == diff]
    for model_key in MODEL_KEYS:
        model = MODEL_LABELS[model_key]
        for domain in DOMAIN_ORDER:
            cells = {}
            for method in ACTIVE_METHODS:
                r = subset[(subset["model"] == model)
                           & (subset["domain"] == domain)
                           & (subset["method"] == method)]
                if not r.empty:
                    cells[method] = (float(r.iloc[0].owa),
                                     bool(r.iloc[0].significant))
            if not cells:
                continue
            best = min(owa for owa, _ in cells.values())
            rendered = []
            for method in ACTIVE_METHODS:
                if method not in cells:
                    rendered.append("—")
                    continue
                owa, significant = cells[method]
                text = f"{owa:.4f}" + ("*" if significant else "")
                if owa - best <= 5e-5:
                    text = f"**{text}**"
                rendered.append(text)
            lines.append(f"| {model} | {domain} | " + " | ".join(rendered) + " |")
    return lines


def fmt_quality_table(table: pd.DataFrame, model: str, diff: str,
                      domains: list[str]) -> list[str]:
    lines = [
        "| Domain | Method | OWA | ΔOWA vs grid | median ΔMASE | p (Holm) "
        "| sig. | n_eff | σ changed | med Δσ (signed) | flat | underflow |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- |",
    ]

    def pct(value: float) -> str:
        return "—" if pd.isna(value) else f"{value:.0%}"

    subset = table[(table["model"] == model) & (table["diff"] == diff)]
    for domain in domains:
        for method in ACTIVE_METHODS:
            r = subset[(subset["domain"] == domain) & (subset["method"] == method)]
            if r.empty:
                continue
            r = r.iloc[0]
            signed = ("—" if pd.isna(r.median_signed_dsigma)
                      else f"{r.median_signed_dsigma:+.3f}")
            lines.append(
                f"| {domain} | {method} | {r.owa:.4f} | "
                f"{r.delta_owa_vs_grid:+.4f} | {r.median_delta_mase:+.5f} | "
                f"{r.p_holm:.2e} | {'**yes**' if r.significant else 'no'} | "
                f"{int(r.n_eff)} | {pct(r.sigma_changed_share)} | {signed} | "
                f"{pct(r.flat_share)} | {pct(r.mean_underflow)} |"
            )
    return lines


def write_markdown(
    table: pd.DataFrame, cost: pd.DataFrame, n_clipped: int, path: Path
) -> None:
    model_labels = [MODEL_LABELS[k] for k in MODEL_KEYS]
    conclusions = []
    for model in model_labels:
        on_all = table[(table["diff"] == "on") & (table["domain"] == "All")
                       & (table["model"] == model)]
        if on_all.empty:
            continue

        def owa_of(method: str) -> float | None:
            r = on_all[on_all["method"] == method]
            return None if r.empty else float(r["owa"].iloc[0])

        grid_owa = owa_of(BASELINE_METHOD)
        frozen_owa = owa_of("grid_frozen")
        if grid_owa is not None and frozen_owa is not None:
            fix_gain = grid_owa - frozen_owa
            conclusions.append(
                f"- **{model}** selection-space fix (fresh grid vs frozen "
                f"raw-space grid): ΔOWA = {fix_gain:+.4f} "
                + ("— deployment-space selection helps."
                   if fix_gain < -0.005 else
                   "— deployment-space selection hurts here; investigate "
                   "before concluding." if fix_gain > 0.005 else
                   "— the fix is quality-neutral for this model.")
            )
        searched = on_all[
            ~on_all["method"].isin(("silverman",) + FROZEN_METHODS)
        ]
        spread = searched["owa"].max() - searched["owa"].min()
        silverman_owa = owa_of("silverman")
        silverman_gap = (
            None if silverman_owa is None or grid_owa is None
            else silverman_owa - grid_owa
        )
        line = (
            f"- **{model}**: fresh searched methods span ΔOWA = {spread:.4f} "
            + ("(flat optimum — the search procedure does not matter)"
               if spread < 0.005 else
               "(the search procedure measurably affects quality)")
        )
        if silverman_gap is not None:
            if silverman_gap > 0.005:
                note = "validation-based selection earns its cost."
            elif silverman_gap < -0.005:
                note = ("the closed-form rule outperforms validation "
                        "selection — check the flat/underflow diagnostics "
                        "before interpreting.")
            else:
                note = "even the closed-form rule is competitive."
            line += f"; silverman is {silverman_gap:+.4f} OWA vs grid — {note}"
        conclusions.append(line)
    fresh_on = table[
        (table["diff"] == "on") & (table["domain"] == "All")
        & (~table["method"].isin(FROZEN_METHODS))
        & (table["method"] != "silverman")
    ]
    if not fresh_on.empty and fresh_on["flat_share"].notna().any():
        worst_flat = float(fresh_on["flat_share"].max())
        mean_uf = float(fresh_on["mean_underflow"].mean())
        conclusions.append(
            f"- Objective health (diff=on): worst per-model flat-objective "
            f"share {worst_flat:.1%}, mean kernel-underflow fraction at the "
            f"selected σ {mean_uf:.1%}. "
            + ("The objective is informative — optimizer differences are "
               "meaningful." if worst_flat < 0.05 else
               "WARNING: a substantial share of cells still has a flat "
               "objective — treat optimizer deltas as tie-breaking noise "
               "there.")
        )
    if not cost["mean_call_ms"].isna().all():
        fastest = cost.sort_values("mean_call_ms").iloc[0]
        conclusions.append(
            f"- Cheapest per call overall: {fastest.method} on {fastest.model} "
            f"({fastest.mean_call_ms:.2f} ms/call); see the cost table for the "
            f"full method x model timing."
        )
    conclusions.append(
        f"- Silverman σ clipped to bounds on {n_clipped} (series, model, diff) "
        f"cells."
    )

    lines = [
        "# Sigma-Optimizer Comparison — GRNN family, M4 Yearly (full 23k)",
        "",
        "All five methods were executed FRESH with σ selected in the "
        "deployment feature space; `grid_frozen` / "
        "`gradient_frozen` are previously stored raw-space runs, "
        "shown as references only. Wilcoxon signed-rank "
        "on paired per-series MASE vs the FRESH grid baseline within each "
        "model, Holm within model/domain/diff.",
        "",
        "## Methods (brief)", "", THEORY,
        "## Combined OWA by domain — differencing ON (the operating regime)",
        "",
        "**Bold** = best OWA in the row (ties within 5e-5 all bolded); "
        "`*` = Holm-significant paired difference vs the FRESH grid "
        "baseline (Wilcoxon on per-series MASE). `grid_frozen` / "
        "`gradient_frozen` = frozen raw-space references.", "",
        *fmt_combined_owa_table(table, "on"), "",
        "## Combined OWA by domain — differencing OFF (appendix; "
        "interpolator collapse regime)", "",
        *fmt_combined_owa_table(table, "off"), "",
    ]
    lines += [
        "## Per-model detail — differencing ON "
        "(ΔOWA, ΔMASE, p-values, σ agreement)", "",
    ]
    for model in model_labels:
        lines += [f"### {model}", "",
                  *fmt_quality_table(table, model, "on", DOMAIN_ORDER), ""]
    lines += [
        "## Cost — microbenchmark (500 series, differencing on)", "",
        "`per call` = pure σ-search wall time of ONE selector invocation "
        "(ensembles make one call per candidate M ∈ {3,5,7}; single/bagging "
        "make one call total). `full selection` = the whole per-series (σ, M) "
        "selection including the M-validation gate, which is identical across "
        "methods.", "",
        "| Model | Method | mean ms / call | median ms / call | "
        "mean ms / full selection | mean evals / call |",
        "| --- | --- | --- | --- | --- | --- |",
        *(f"| {r.model} | {r.method} | {r.mean_call_ms:.2f} | "
          f"{r.median_call_ms:.2f} | {r.mean_selection_ms:.2f} | "
          f"{r.mean_evals_per_call:.1f} |" for r in cost.itertuples()),
        "",
        "Timing measures σ-selection only, on the SAME precomputed-distance "
        "objective for every method — it isolates optimizer overhead. The "
        "grid implementation of the stored reference runs recomputes "
        "distances per σ and is slower than the 'grid' row here.",
        "",
        "## Conclusions", "", *conclusions, "",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_owa(table: pd.DataFrame, path: Path) -> None:
    data = table[table["diff"] == "on"]
    domains = [d for d in DOMAIN_ORDER if d in data["domain"].unique()]
    n_methods = len(ACTIVE_METHODS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, model_key in zip(axes.ravel(), MODEL_KEYS, strict=True):
        model = MODEL_LABELS[model_key]
        sub_model = data[data["model"] == model]
        ax.set_title(model)
        if sub_model.empty:
            continue
        width = 0.8 / n_methods
        for mi, method in enumerate(ACTIVE_METHODS):
            sub = sub_model[sub_model["method"] == method]
            sub = sub.set_index("domain").reindex(domains)
            x = np.arange(len(domains)) + (mi - (n_methods - 1) / 2) * width
            ax.bar(x, sub["owa"], width * 0.9, color=METHOD_COLORS[method],
                   label=method if model_key == MODEL_KEYS[0] else None)
        ax.set_xticks(np.arange(len(domains)))
        ax.set_xticklabels(domains, fontsize=8)
        lo, hi = sub_model["owa"].min(), sub_model["owa"].max()
        ax.set_ylim(lo - 0.02, hi + 0.02)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("OWA (lower is better)")
    axes[1, 0].set_ylabel("OWA (lower is better)")
    fig.suptitle("Sigma optimizers — OWA by domain and model "
                 "(differencing on; gray = frozen raw-space references)")
    fig.legend(loc="lower center", ncol=len(ACTIVE_METHODS), fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def plot_cost(table: pd.DataFrame, cost: pd.DataFrame, path: Path) -> None:
    on_all = table[(table["diff"] == "on") & (table["domain"] == "All")]
    merged = on_all.merge(cost, on=["model", "method"])
    if merged.empty or merged["mean_call_ms"].isna().all():
        return
    colors = {"GRNN": "#4878a8", "GRNN-Mean": "#e1812c",
              "GRNN-Median": "#559e55", "GRNN-Bag": "#b04a4a"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model, part in merged.groupby("model"):
        ax.scatter(part["mean_call_ms"], part["owa"], s=45,
                   color=colors[model], label=model)
        for r in part.itertuples():
            ax.annotate(r.method, (r.mean_call_ms, r.owa),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("mean σ-selection time per call, ms (log)")
    ax.set_ylabel("OWA, All domains (differencing on)")
    ax.set_title("Efficiency frontier: per-call search cost vs forecast quality")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="sigma_opt")
    parser.add_argument("--methods", default=",".join(FRESH_METHOD_ORDER),
                        help=f"comma list from: {', '.join(FRESH_METHOD_ORDER)} "
                             f"(default runs ALL FIVE fresh, in deployment "
                             f"feature space)")
    parser.add_argument("--grid-tag", default="main")
    parser.add_argument("--gradient-tag", default="main_gradsigma")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N series (0 = all; smoke)")
    parser.add_argument("--bench-size", type=int, default=500)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--no-frozen", action="store_true",
                        help="run without the frozen reference CSVs "
                             "(grnn_results_<grid-tag>/<gradient-tag>): drops "
                             "the *_frozen report columns and the 0.8734 "
                             "anchor gate; never regenerates frozen runs")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}; valid: {list(ALL_METHODS)}")

    overall_started = time.perf_counter()
    config = load_config()

    frozen_paths = {
        "grid": M4_RESULTS_DIR / f"grnn_results_{args.grid_tag}.csv",
        "gradient": M4_RESULTS_DIR / f"grnn_results_{args.gradient_tag}.csv",
    }
    missing_frozen = [str(p) for p in frozen_paths.values() if not p.exists()]
    if missing_frozen and not args.no_frozen:
        raise SystemExit(
            "frozen reference CSVs not found:\n  "
            + "\n  ".join(missing_frozen)
            + "\n\nThe comparison is self-contained without them (all five "
              "methods run fresh); the frozen files only add the *_frozen "
              "reference columns and the 0.8734 anchor gate. Either copy the "
              "files from the machine that produced them, or rerun with "
              "--no-frozen to proceed without references.\n"
              "They are NEVER regenerated implicitly — they are frozen "
              "reference runs. To regenerate deliberately "
              "(hours each, and only with the code state that produced them):\n"
              "  config.yaml evaluation.sigma_search: grid    -> python -m "
              "src.scripts.run_m4_pipeline --stages grnn --full --tag "
              f"{args.grid_tag}\n"
              "  config.yaml evaluation.sigma_search: gradient -> python -m "
              "src.scripts.run_m4_pipeline --stages grnn --full --tag "
              f"{args.gradient_tag}"
        )
    global ACTIVE_METHODS
    # --no-frozen is authoritative: it excludes the references even when the
    # frozen CSVs are present, so a references-free run is reproducible on
    # any machine regardless of which files happen to exist locally.
    skip_frozen = bool(missing_frozen) or args.no_frozen
    if skip_frozen:
        log("load-frozen", "--no-frozen: running WITHOUT the frozen "
            "reference columns and WITHOUT the 0.8734 anchor gate")
        ACTIVE_METHODS = FRESH_METHOD_ORDER
        frozen_hashes_before: dict[str, str] = {}
        frozen = None
    else:
        frozen_hashes_before = {
            k: sha256_path(p) for k, p in frozen_paths.items()
        }
        frozen = pd.concat([
            load_frozen(args.grid_tag, "grid_frozen"),
            load_frozen(args.gradient_tag, "gradient_frozen"),
        ])

    series_list = load_yearly_series()
    ids_digest = hashlib.sha256(
        ",".join(s.series_id for s in series_list).encode()
    ).hexdigest()[:12]
    log("load-frozen", f"{len(series_list)} Yearly series loaded; "
                       f"series-id set hash {ids_digest}")
    if args.limit:
        series_list = series_list[:args.limit]
        log("load-frozen", f"--limit {args.limit}: smoke run on first "
                           f"{len(series_list)} series")

    results_path = stage_run_new(series_list, methods, config, args.tag)

    if args.skip_bench:
        cost = pd.DataFrame([
            {"model": MODEL_LABELS[k], "method": m, "n_series": 0,
             "mean_call_ms": np.nan, "median_call_ms": np.nan,
             "mean_selection_ms": np.nan, "mean_evals_per_call": np.nan}
            for k in MODEL_KEYS for m in FRESH_METHOD_ORDER
        ])
    else:
        cost = stage_microbenchmark(series_list, config, args.bench_size)

    new_results = pd.read_csv(results_path)
    wanted_ids = {s.series_id for s in series_list}
    new_results = new_results[new_results["series_id"].isin(wanted_ids)]
    n_clipped = int(
        ((new_results["method"] == "silverman")
         & ((new_results["sigma"] <= 0.1) | (new_results["sigma"] >= 10.0))).sum()
    )
    # objective_flat lands as bool/object from CSV appends — normalize to
    # float so stats can average it (frozen rows get NaN via the concat).
    new_results["objective_flat"] = (
        new_results["objective_flat"].astype(str).str.lower()
        .map({"true": 1.0, "false": 0.0})
    )
    combined_parts = [
        new_results[["series_id", "domain", "model", "diff", "sigma", "M",
                     "objective_flat", "underflow_frac", "smape", "mase",
                     "method"]],
    ]
    if frozen is not None:
        combined_parts.insert(0, frozen[frozen["series_id"].isin(wanted_ids)])
    combined = pd.concat(combined_parts)
    naive2, naive2_source = load_naive2(series_list)
    log("stats", f"Naive2 denominators: {naive2_source}")
    table = stage_stats(combined, naive2)

    # Validity gates: NaN-free new results; frozen reference matches its
    # stored aggregate OWA.
    nan_rows = int(new_results[["smape", "mase"]].isna().any(axis=1).sum())
    assert nan_rows == 0, f"{nan_rows} new result rows contain NaN metrics"
    expected_rows = len(series_list) * len(methods) * len(MODEL_KEYS) * 2
    log("stats", f"new result rows: {len(new_results)} / {expected_rows} "
                 f"expected (difference = infeasible cells, see skipped CSV)")
    if not args.limit and not skip_frozen:
        anchor = table[
            (table["model"] == "GRNN-Median") & (table["domain"] == "All")
            & (table["diff"] == "on") & (table["method"] == "grid_frozen")
        ]["owa"].iloc[0]
        assert abs(anchor - 0.8734) < 5e-4, (
            f"anchor gate failed: frozen grid GRNN-Median OWA {anchor:.4f} "
            f"!= published 0.8734 — inputs or pairing are broken"
        )
        log("stats", f"anchor gate passed: grid_frozen GRNN-Median All/on OWA "
                     f"{anchor:.4f} (expected 0.8734)")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # One CSV per optimization method: OWA split by domain (x model x diff).
    method_csv_columns = [
        "model", "domain", "diff", "n", "n_eff", "owa", "delta_owa_vs_grid",
        "median_delta_mase", "p_value", "p_holm", "significant",
        "sigma_changed_share", "median_abs_dsigma", "median_signed_dsigma",
        "flat_share", "mean_underflow",
    ]
    method_paths = []
    for method in ACTIVE_METHODS:
        part = table[table["method"] == method][method_csv_columns]
        method_path = FINAL_DIR / f"m4_sigma_optimizer_{method}_{args.tag}.csv"
        part.to_csv(method_path, index=False)
        method_paths.append(method_path)
        log("report", f"{method}: {len(part)} rows -> {method_path.name}")
    cost_path = FINAL_DIR / f"m4_sigma_optimizers_cost_{args.tag}.csv"
    report_path = FINAL_DIR / f"m4_sigma_optimizers_{args.tag}.md"
    cost.to_csv(cost_path, index=False)
    write_markdown(table, cost, n_clipped, report_path)
    plot_owa(table, PLOTS_DIR / f"sigma_optimizers_owa_{args.tag}.png")
    plot_cost(table, cost, PLOTS_DIR / f"sigma_optimizers_cost_{args.tag}.png")

    if not skip_frozen:
        frozen_hashes_after = {
            k: sha256_path(p) for k, p in frozen_paths.items()
        }
        assert frozen_hashes_before == frozen_hashes_after, (
            "frozen input CSVs changed during the run — reproducibility "
            "violation"
        )
    flat_col = new_results["objective_flat"]
    manifest = {
        "tag": args.tag,
        "generated_unix": time.time(),
        "protocol_version": 2,
        "selection_feature_space": {
            "scaler": config["preparation"]["scaler"],
            "pipeline_fit": "fit split only (no validation leakage)",
            "note": "sigma selected in the deployment feature space; the "
                    "*_frozen reference rows come from stored raw-space runs",
        },
        "objective_health": {
            "flat_share": float(flat_col.mean()),
            "mean_underflow_frac":
                float(new_results["underflow_frac"].mean()),
        },
        "baseline": "grid (fresh, deployment feature space)",
        "methods_fresh": methods,
        "models": list(MODEL_KEYS),
        "limit": args.limit,
        "n_series": len(series_list),
        "n_new_result_rows": int(len(new_results)),
        "frozen_references_present": not skip_frozen,
        "naive2_source": naive2_source,
        "frozen_inputs_sha256": frozen_hashes_before,
        "code_sha256": {
            name: sha256_path(REPO_ROOT / name)
            for name in ("src/models/sigma_optimizers.py",
                         "src/scripts/compare_sigma_optimizers.py")
        },
        "config_snapshot": {
            key: config[key] for key in
            ("grnn", "ensemble_robust", "ensemble_avg", "ensemble_bagging")
        } | {
            "preparation_lag_order": config["preparation"]["lag_order"],
            "evaluation_validation_size":
                config["evaluation"]["validation_size"],
            "bagging_runs": BAGGING_RUNS,
        },
        "silverman_clip_cells": n_clipped,
        "output_tables": [p.name for p in method_paths] + [cost_path.name],
        "total_runtime_s": time.perf_counter() - overall_started,
    }
    manifest_path = M4_RESULTS_DIR / f"manifest_sigma_opt_{args.tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    summary = table[(table["diff"] == "on") & (table["domain"] == "All")]
    print()
    print(summary[["model", "method", "owa", "delta_owa_vs_grid", "p_holm",
                   "significant", "n_eff", "flat_share",
                   "mean_underflow"]].to_string(index=False))
    log("done", f"report: {report_path} (+ {len(method_paths)} per-method CSVs, "
                f"{cost_path.name}, 2 plots, {manifest_path.name}) in "
                f"{time.perf_counter() - overall_started:.0f}s")


if __name__ == "__main__":
    main()
