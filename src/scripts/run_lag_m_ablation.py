"""Sensitivity grid over lag order L and ensemble size M.

Justifies the `lag_order: 3` and `M_values: [3, 5, 7]` defaults in
`config.yaml` empirically. Sigma and M are selected per series by within-train
validation while the lag order is a fixed design choice; this grid sweeps both
so that no design parameter rests on fiat alone.

**Which models.** Only `single_grnn`, `grnn_avg` and `grnn_robust`.
`grnn_bagging` is deliberately excluded: there M counts bootstrap replicates,
not interleaved sub-sequences, so putting it on the same M axis would place two
different quantities under one label. `single_grnn` has no M and enters the
table as the M = 1 reference column, which is what makes the grid readable —
without it a reader cannot tell whether ensembling helps at all.

**What varies and what does not.** L and M are FIXED per cell. Sigma continues
to be selected per series by the same within-train validation as the main
pipeline. One thing changes at a time, or the result is uninterpretable.

**Coverage is exact, not approximate.** A cell is only run if all 23,000 series
survive it under BOTH differencing variants, so every cell is computed on the
identical sample and cells are comparable. The binding constraint chain is:

    differencing   -> n - 1 usable points
    sliding window -> N = (n - 1) - L windows
    validation     -> ceil(0.2 * N) held out
    decomposition  -> requires N - ceil(0.2 * N) >= M

At the shortest M4 Yearly series (n = 13) this gives
`M_max = (12 - L) - ceil(0.2 * (12 - L))` — which is exactly 7 at L = 3, i.e.
the default `M_values: [3, 5, 7]` sits precisely at the ceiling imposed by ONE
series.

**M = 2 is a correctness test, not filler.** The median of two numbers equals
their mean, so `grnn_robust` at M = 2 must reproduce `grnn_avg` at M = 2 to
floating-point identity. A mismatch means a bug in the decomposition or the
aggregation. `--check-m2` reports it.

**No post-hoc selection.** Every feasible cell is reported. The grid is a
sensitivity analysis, so it must never be used to pick the best cell and
adopt it silently — if a cell other than (L=3, M=3) wins, changing the default
configuration is a separate decision with its own run.

Outputs are written under their own tag and never touch the main results.

Usage:
    python -m src.scripts.run_lag_m_ablation --tag ablation_extended
    python -m src.scripts.run_lag_m_ablation --scope core       # L>=3, M>=3
    python -m src.scripts.run_lag_m_ablation --limit 50         # smoke
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.m4 import M4Series
from src.evaluation.m4_harness import INFEASIBLE_ERRORS
from src.evaluation.m4_harness import evaluate_grnn_series
from src.evaluation.m4_metrics import mase as mase_metric
from src.evaluation.m4_metrics import naive2_forecast_yearly
from src.evaluation.m4_metrics import owa as owa_metric
from src.evaluation.m4_metrics import smape as smape_metric
from src.scripts.run_m4_pipeline import M4_FINAL_DIR
from src.scripts.run_m4_pipeline import M4_RESULTS_DIR
from src.scripts.run_m4_pipeline import GENERATED_TABLES_DIR
from src.scripts.run_m4_pipeline import code_state_fingerprint
from src.scripts.run_m4_pipeline import feature_pipeline_manifest
from src.scripts.run_m4_pipeline import load_config
from src.scripts.run_m4_pipeline import load_yearly_series
from src.scripts.run_m4_pipeline import package_versions

ENSEMBLE_MODELS = ("grnn_avg", "grnn_robust")
SINGLE_MODEL = "single_grnn"
MODEL_LABELS = {
    SINGLE_MODEL: "GRNN",
    "grnn_avg": "GRNN-Mean",
    "grnn_robust": "GRNN-Median",
}
RESULT_COLUMNS = (
    "L", "M", "model", "diff", "series_id", "domain", "smape", "mase",
)
SCOPES = {
    # (L range, M range) for the ensemble grid
    "extended": (range(1, 11), range(2, 9)),
    "core": (range(3, 11), range(3, 9)),
}


def _cell_feasible(train_length: int, lag_order: int, n_models: int, difference: bool) -> bool:
    """Exact replication of the constraint chain in the harness."""
    working = train_length - 1 if difference else train_length
    if working <= lag_order:
        return False
    n_windows = working - lag_order
    n_validation = math.ceil(0.2 * n_windows)
    n_fit = n_windows - n_validation
    if n_fit < 1 or n_validation < 1:
        return False
    return n_fit >= n_models


def build_grid(
    train_lengths: np.ndarray,
    scope: str,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Cells where EVERY series survives under both differencing variants.

    Returns (ensemble cells as (L, M), single-GRNN L values).
    """
    lag_range, m_range = SCOPES[scope]

    def full_coverage(lag_order: int, n_models: int) -> bool:
        return all(
            _cell_feasible(int(length), lag_order, n_models, True)
            and _cell_feasible(int(length), lag_order, n_models, False)
            for length in train_lengths
        )

    ensemble_cells = [
        (lag_order, n_models)
        for lag_order in lag_range
        for n_models in m_range
        if full_coverage(lag_order, n_models)
    ]
    single_lags = [
        lag_order for lag_order in lag_range if full_coverage(lag_order, 1)
    ]
    return ensemble_cells, single_lags


def _cell_config(base: dict[str, Any], lag_order: int, n_models: int) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["preparation"]["lag_order"] = lag_order
    for key in ("ensemble_avg", "ensemble_robust", "ensemble_bagging"):
        config[key]["M_values"] = [n_models]
    return config


def _naive2_reference(series_list: list[M4Series]) -> tuple[float, float]:
    """Naive2 sMAPE and MASE on this exact sample — the OWA denominator."""
    smape_values, mase_values = [], []
    for series in series_list:
        forecast = naive2_forecast_yearly(series.train, len(series.test))
        smape_values.append(smape_metric(series.test, forecast))
        mase_values.append(
            mase_metric(series.train, series.test, forecast, series.m)
        )
    return float(np.mean(smape_values)), float(np.mean(mase_values))


def _completed_cells(path: Path) -> set[tuple[int, int, str, str]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["L", "M", "model", "diff"])
    return {
        (int(row.L), int(row.M), str(row.model), str(row.diff))
        for row in frame.drop_duplicates().itertuples()
    }


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_cell(
    series_list: list[M4Series],
    base_config: dict[str, Any],
    lag_order: int,
    n_models: int,
    model_key: str,
    difference: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Scores one (L, M, model, diff) cell over every series."""
    config = _cell_config(base_config, lag_order, n_models)
    diff_label = "on" if difference else "off"
    rows, skipped = [], 0
    for series in series_list:
        try:
            result = evaluate_grnn_series(
                series=series,
                model_key=model_key,
                config=config,
                difference=difference,
            )
            if result is None:
                skipped += 1
                continue
        except INFEASIBLE_ERRORS:
            skipped += 1
            continue
        rows.append({
            "L": lag_order,
            "M": n_models,
            "model": model_key,
            "diff": diff_label,
            "series_id": series.series_id,
            "domain": series.category,
            "smape": result["smape"],
            "mase": result["mase"],
        })
    return rows, skipped


def aggregate(results_path: Path, naive2: tuple[float, float]) -> pd.DataFrame:
    """Per-cell aggregate sMAPE, MASE and OWA."""
    naive2_smape, naive2_mase = naive2
    frame = pd.read_csv(results_path)
    grouped = frame.groupby(["L", "M", "model", "diff"], sort=True).agg(
        n=("series_id", "size"),
        smape=("smape", "mean"),
        mase=("mase", "mean"),
    ).reset_index()
    grouped["owa"] = [
        owa_metric(row.smape, row.mase, naive2_smape, naive2_mase)
        for row in grouped.itertuples()
    ]
    grouped["method"] = grouped["model"].map(MODEL_LABELS)
    return grouped


def check_m2_identity(results_path: Path) -> dict[str, Any]:
    """Median-of-two equals mean-of-two: a free correctness test.

    At M = 2 the interleaved median ensemble is mathematically identical to the
    mean ensemble. Any disagreement beyond floating-point noise indicates a bug
    in `decompose_into_subsequences` or in the aggregation, not a modelling
    difference.
    """
    frame = pd.read_csv(results_path)
    subset = frame[frame["M"] == 2]
    if subset.empty:
        return {"status": "not_applicable", "detail": "no M=2 cells in results"}

    wide = subset.pivot_table(
        index=["L", "diff", "series_id"], columns="model", values="mase"
    )
    if not {"grnn_avg", "grnn_robust"}.issubset(wide.columns):
        return {"status": "not_applicable", "detail": "both M=2 models required"}

    paired = wide[["grnn_avg", "grnn_robust"]].dropna()
    deviation = (paired["grnn_avg"] - paired["grnn_robust"]).abs()
    worst = float(deviation.max()) if len(deviation) else float("nan")
    return {
        "status": "pass" if worst <= 1e-9 else "FAIL",
        "n_pairs": int(len(paired)),
        "max_abs_mase_deviation": worst,
        "note": (
            "median of two == mean of two; a non-zero deviation is a bug in "
            "the decomposition or the aggregation"
        ),
    }


def write_tables(aggregated: pd.DataFrame, tag: str) -> dict[str, str]:
    """Grid tables: rows = L, columns = M, one table per metric and diff."""
    GENERATED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for metric, label, decimals in (
        ("owa", "OWA", 4), ("smape", "sMAPE", 3), ("mase", "MASE", 3),
    ):
        for diff in sorted(aggregated["diff"].unique()):
            subset = aggregated[aggregated["diff"] == diff]
            lags = sorted(subset["L"].unique())
            m_values = sorted(subset["M"].unique())

            lines = [
                f"# L x M sensitivity grid — {label}, differencing {diff}",
                "",
                "Every cell is computed on all 23,000 series; a dash means the "
                "configuration is infeasible for at least one series, so the "
                "cell would not be comparable. The ceiling is set by the single "
                "shortest series (n = 13) via "
                "`M_max = (12 - L) - ceil(0.2 * (12 - L))`.",
                "",
                "Cell format: GRNN-Mean / GRNN-Median. Column M=1 is the single "
                "GRNN, which has no ensemble and therefore no aggregation.",
                "",
                "| L | " + " | ".join(f"M={m}" for m in m_values) + " |",
                "|---" * (len(m_values) + 1) + "|",
            ]
            for lag_order in lags:
                cells = []
                for n_models in m_values:
                    rows = subset[(subset["L"] == lag_order) & (subset["M"] == n_models)]
                    if rows.empty:
                        cells.append("—")
                        continue
                    by_model = rows.set_index("model")[metric]
                    if n_models == 1:
                        value = by_model.get(SINGLE_MODEL, float("nan"))
                        cells.append(
                            "—" if pd.isna(value) else f"{value:.{decimals}f}"
                        )
                        continue
                    parts = []
                    for model_key in ENSEMBLE_MODELS:
                        value = by_model.get(model_key, float("nan"))
                        parts.append(
                            "—" if pd.isna(value) else f"{value:.{decimals}f}"
                        )
                    cells.append(" / ".join(parts))
                lines.append(f"| {lag_order} | " + " | ".join(cells) + " |")
            lines.append("")

            stem = f"table_lag_m_{metric}_{diff}_{tag}"
            path = GENERATED_TABLES_DIR / f"{stem}.md"
            path.write_text("\n".join(lines), encoding="utf-8")
            written[stem] = str(path)

    csv_path = M4_FINAL_DIR / f"m4_lag_m_ablation_{tag}.csv"
    aggregated.to_csv(csv_path, index=False, float_format="%.12g")
    written["aggregate_csv"] = str(csv_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="ablation_extended")
    parser.add_argument("--scope", default="extended", choices=sorted(SCOPES))
    parser.add_argument(
        "--differencing", default="both", choices=["both", "on", "off"]
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="smoke run on the first N series (0 = all 23,000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    M4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    M4_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_config()
    series_list = load_yearly_series()
    if args.limit:
        series_list = series_list[: args.limit]
        print(f"[smoke] limited to {len(series_list)} series")

    train_lengths = np.array([len(series.train) for series in series_list])
    ensemble_cells, single_lags = build_grid(train_lengths, args.scope)
    diff_flags = {
        "both": [False, True], "on": [True], "off": [False],
    }[args.differencing]

    total_cells = (
        len(ensemble_cells) * len(ENSEMBLE_MODELS) + len(single_lags)
    ) * len(diff_flags)
    print(
        f"scope={args.scope}: {len(ensemble_cells)} ensemble (L,M) cells, "
        f"{len(single_lags)} single-GRNN L values, "
        f"{len(diff_flags)} differencing variant(s) "
        f"=> {total_cells} cells x {len(series_list)} series"
    )

    results_path = M4_RESULTS_DIR / f"lag_m_ablation_{args.tag}.csv"
    completed = _completed_cells(results_path)
    if completed:
        print(f"resuming: {len(completed)} cells already on disk")

    naive2 = _naive2_reference(series_list)
    print(f"Naive2 reference on this sample: sMAPE {naive2[0]:.4f}, MASE {naive2[1]:.4f}")

    work: list[tuple[int, int, str]] = [
        (lag_order, 1, SINGLE_MODEL) for lag_order in single_lags
    ] + [
        (lag_order, n_models, model_key)
        for lag_order, n_models in ensemble_cells
        for model_key in ENSEMBLE_MODELS
    ]

    done = 0
    skipped_total = 0
    for lag_order, n_models, model_key in work:
        for difference in diff_flags:
            diff_label = "on" if difference else "off"
            key = (lag_order, n_models, model_key, diff_label)
            done += 1
            if key in completed:
                continue
            cell_started = time.perf_counter()
            rows, skipped = run_cell(
                series_list=series_list,
                base_config=base_config,
                lag_order=lag_order,
                n_models=n_models,
                model_key=model_key,
                difference=difference,
            )
            skipped_total += skipped
            if skipped:
                # Should be impossible: build_grid only admits full-coverage
                # cells. Loud, because a silently shrinking sample would make
                # cells incomparable — the exact defect this grid avoids.
                print(
                    f"  WARNING {key}: {skipped} series skipped despite "
                    "full-coverage screening"
                )
            _append_rows(results_path, rows)
            elapsed = time.perf_counter() - cell_started
            total_elapsed = time.perf_counter() - started
            eta = total_elapsed / done * (total_cells - done)
            print(
                f"  [{done}/{total_cells}] L={lag_order} M={n_models} "
                f"{MODEL_LABELS[model_key]} diff={diff_label}: "
                f"{len(rows)} rows in {elapsed:.1f}s · eta {eta/60:.0f} min"
            )

    aggregated = aggregate(results_path, naive2)
    written = write_tables(aggregated, args.tag)
    m2 = check_m2_identity(results_path)
    print(f"\nM=2 median/mean identity check: {m2}")

    manifest = {
        "tag": args.tag,
        "scope": args.scope,
        "seed": args.seed,
        "started_at_utc": datetime.fromtimestamp(
            time.time() - (time.perf_counter() - started), tz=timezone.utc
        ).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_series": len(series_list),
        "ensemble_cells": [list(cell) for cell in ensemble_cells],
        "single_lags": list(single_lags),
        "differencing": args.differencing,
        "models": [SINGLE_MODEL, *ENSEMBLE_MODELS],
        "excluded_models": {
            "grnn_bagging": (
                "M counts bootstrap replicates, not interleaved "
                "sub-sequences; not the same axis"
            ),
        },
        "naive2_reference": {"smape": naive2[0], "mase": naive2[1]},
        "m2_identity_check": m2,
        "series_skipped_total": skipped_total,
        "feature_pipeline": feature_pipeline_manifest(base_config),
        "config": base_config,
        "code_state_sha256": code_state_fingerprint(),
        "package_versions": package_versions(),
        "outputs": written,
        "wall_clock_s": time.perf_counter() - started,
    }
    manifest_path = M4_RESULTS_DIR / f"manifest_lag_m_{args.tag}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"manifest: {manifest_path}")
    print(f"total: {manifest['wall_clock_s']/3600:.2f} h")


if __name__ == "__main__":
    main()
