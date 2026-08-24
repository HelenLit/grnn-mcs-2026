"""Staged M4 Yearly domain-wise comparison pipeline.

Implemented stages:
    sync -> validate -> anchors -> grnn -> table -> strat -> plots

`strat` is pure post-processing over the stored `grnn_results_<tag>.csv`
(per-step forecasts F1..F6) — it never re-runs the GRNN family.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import inspect
import json
import subprocess
import sys
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import yaml
from scipy import stats

from src.data.m4 import M4Series
from src.data.m4 import ao_screen
from src.data.preparation import FeaturePipeline
from src.data.m4 import load_m4_csv
from src.data.m4 import load_submission_forecasts
from src.evaluation.m4_harness import GRNN_FAMILY_MODEL_KEYS
from src.evaluation.m4_harness import INFEASIBLE_ERRORS
from src.evaluation.m4_harness import evaluate_grnn_series
from src.evaluation.m4_harness import score_forecasts
from src.evaluation.m4_metrics import mase
from src.evaluation.m4_metrics import owa
from src.evaluation.m4_metrics import smape
from src.evaluation.m4_stratify import ANOMALY_BIN_LABELS
from src.evaluation.m4_stratify import HORIZON
from src.evaluation.m4_stratify import LENGTH_BIN_LABELS
from src.evaluation.m4_stratify import MASE_STEP_COLUMNS
from src.evaluation.m4_stratify import SMAPE_STEP_COLUMNS
from src.evaluation.m4_stratify import STEP_COLUMNS
from src.evaluation.m4_stratify import Stratification
from src.evaluation.m4_stratify import aggregate_by_group
from src.evaluation.m4_stratify import anomaly_bin
from src.evaluation.m4_stratify import length_bin
from src.evaluation.m4_stratify import merge_small_groups
from src.evaluation.m4_stratify import series_masep
from src.evaluation.m4_stratify import to_wide
from src.evaluation.m4_stratify import verify_group_identity
from src.evaluation.m4_stratify import write_markdown
from src.evaluation.metrics import compute_metrics
from src.models.sigma_gradient import select_sigma_gradient
from src.evaluation.plotting import M4_METHOD_STYLE
from src.evaluation.plotting import apply_m4_style
from src.evaluation.plotting import save_m4_figure
from src.scripts.download_m4_forecasts import DEFAULT_DEST
from src.scripts.download_m4_forecasts import REQUIRED_SUBMISSIONS
from src.scripts.download_m4_forecasts import sync_forecasts

CONFIG_PATH = Path("src/config/config.yaml")
M4_RAW_DIR = Path("data/raw/m4")
M4_RESULTS_DIR = Path("reports/results/m4")
M4_FINAL_DIR = Path("reports/final")
M4_PLOTS_DIR = Path("plots/m4")
PUBLISHED_ANCHORS_PATH = Path("data/anchors/m4_yearly_published_anchors.csv")
DEFAULT_STAGES = ("sync", "validate", "anchors")
SUPPORTED_STAGES = DEFAULT_STAGES + ("grnn", "table", "strat", "plots")
DOMAIN_COUNTS = {
    "Micro": 6538,
    "Finance": 6519,
    "Macro": 3903,
    "Industry": 3716,
    "Other": 1236,
    "Demographic": 1088,
}
STRICT_ANCHOR_METHODS = ("118", "245", "Com", "ARIMA", "Naive2", "MLP")
ANCHOR_TOLERANCE = 0.005
ARIMA_OWA_BAND = (0.75, 1.10)
BAGGING_RUNS = 5
RESULT_KEY_COLUMNS = ("series_id", "model", "diff")
GRNN_RESULT_COLUMNS = (
    "series_id", "domain", "model", "diff", "n_train", "sigma", "M",
    "config_hash", "runtime_ms", "seed", "bagging_seed_count",
    "smape", "mase", "MAE", "RMSE", "MedAE", "MAPE", "R2",
    "smape_std", "mase_std", "F1", "F2", "F3", "F4", "F5", "F6",
    "skipped_reason",
)
SKIP_COLUMNS = ("series_id", "domain", "model", "diff", "skipped_reason")
GRNN_TABLE_LABELS = {
    "single_grnn": "GRNN",
    "grnn_avg": "GRNN-Mean",
    "grnn_robust": "GRNN-Median",
    "grnn_bagging": "GRNN-Bag",
}
OFFICIAL_TABLE_LABELS = {
    "118": "118-Smyl",
    "245": "245-FFORMA",
    "Com": "Com",
    "ARIMA": "ARIMA",
    "MLP": "MLP",
    "Naive2": "Naive2",
}
TABLE_METHOD_COLUMNS = (
    "GRNN", "GRNN-Mean", "GRNN-Median", "GRNN-Bag",
    "118-Smyl", "245-FFORMA", "Com", "ARIMA", "MLP", "Naive2",
)
TABLE_BASE_COLUMNS = ("Domain", "n", "Med. len", "Diff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="main", help="run tag for output files")
    parser.add_argument("--seed", type=int, default=42, help="global random seed")
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help=f"comma-separated stages, supported: {', '.join(SUPPORTED_STAGES)}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reserved for later resumable stages; accepted for CLI stability",
    )
    parser.add_argument(
        "--n-per-domain",
        type=int,
        default=None,
        help="stratified sample size per M4 domain for the grnn stage",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the grnn stage on all 23,000 Yearly series",
    )
    parser.add_argument(
        "--differencing",
        choices=["both", "off", "on"],
        default="both",
        help="differencing variants for the grnn stage",
    )
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_state_fingerprint() -> str:
    digest = hashlib.sha256()
    files = sorted(Path("src").glob("**/*.py")) + [CONFIG_PATH]
    for path in files:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def package_versions() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return [f"pip freeze unavailable: {proc.stderr.strip()}"]
    return sorted(line for line in proc.stdout.splitlines() if line.strip())


def input_files() -> list[Path]:
    files = [
        M4_RAW_DIR / "Train" / "Yearly-train.csv",
        M4_RAW_DIR / "Test" / "Yearly-test.csv",
        M4_RAW_DIR / "M4-info.csv",
        PUBLISHED_ANCHORS_PATH,
    ]
    for submission_id in REQUIRED_SUBMISSIONS:
        files.append(DEFAULT_DEST / "rar" / f"submission-{submission_id}.rar")
        files.append(DEFAULT_DEST / "csv" / f"submission-{submission_id}.csv")
    return files


def input_sha256s() -> dict[str, str]:
    return {
        str(path): sha256_path(path)
        for path in input_files()
        if path.exists()
    }


def load_yearly_series() -> list[M4Series]:
    return load_m4_csv(
        train_path=M4_RAW_DIR / "Train" / "Yearly-train.csv",
        test_path=M4_RAW_DIR / "Test" / "Yearly-test.csv",
        info_path=M4_RAW_DIR / "M4-info.csv",
    )


def stage_validate() -> dict[str, Any]:
    series_list = load_yearly_series()
    assert len(series_list) == 23_000, f"Expected 23,000 series, got {len(series_list)}"

    counts = pd.Series([series.category for series in series_list]).value_counts().to_dict()
    assert counts == DOMAIN_COUNTS, f"Domain counts differ: {counts}"
    assert min(len(series.train) for series in series_list) == 13
    assert {len(series.test) for series in series_list} == {6}
    assert {series.m for series in series_list} == {1}

    return {
        "n_series": len(series_list),
        "domain_counts": counts,
        "min_train_length": min(len(series.train) for series in series_list),
        "horizon": 6,
        "input_sha256": input_sha256s(),
    }


def _submission_csv_path(submission_id: str) -> Path:
    return DEFAULT_DEST / "csv" / f"submission-{submission_id}.csv"


def _load_published_anchors() -> dict[str, float]:
    anchors = pd.read_csv(PUBLISHED_ANCHORS_PATH)
    return {
        str(row["method"]): float(row["owa_yearly"])
        for _, row in anchors.iterrows()
    }


def _aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    naive2 = scores[scores["source"] == "Naive2"]
    rows = []
    for domain in ["All", *DOMAIN_COUNTS.keys()]:
        domain_scores = scores if domain == "All" else scores[scores["domain"] == domain]
        domain_naive2 = naive2 if domain == "All" else naive2[naive2["domain"] == domain]
        naive2_smape = float(domain_naive2["smape"].mean())
        naive2_mase = float(domain_naive2["mase"].mean())

        for source, source_scores in domain_scores.groupby("source", sort=False):
            mean_smape = float(source_scores["smape"].mean())
            mean_mase = float(source_scores["mase"].mean())
            rows.append({
                "domain": domain,
                "source": source,
                "n": int(len(source_scores)),
                "smape": mean_smape,
                "mase": mean_mase,
                "owa": owa(mean_smape, mean_mase, naive2_smape, naive2_mase),
            })
    return pd.DataFrame(rows)


def _apply_anchor_gate(anchors: pd.DataFrame) -> pd.DataFrame:
    published = _load_published_anchors()
    rows = []
    failures = []
    for _, row in anchors.iterrows():
        source = str(row["source"])
        domain = str(row["domain"])
        gate_tier = "none"
        published_owa = float("nan")
        delta_owa = float("nan")
        gate_status = "not_checked"

        if domain == "All" and source in STRICT_ANCHOR_METHODS:
            gate_tier = "strict"
            published_owa = published[source]
            delta_owa = abs(float(row["owa"]) - published_owa)
            gate_status = "pass" if delta_owa <= ANCHOR_TOLERANCE else "fail"
            if gate_status == "fail":
                failures.append(
                    f"{source}: OWA {row['owa']:.4f}, published {published_owa:.4f}, "
                    f"delta {delta_owa:.4f}"
                )
        elif domain == "All" and source == "ARIMA":
            gate_tier = "plausibility"
            low, high = ARIMA_OWA_BAND
            gate_status = "pass" if low <= float(row["owa"]) <= high else "fail"
            if gate_status == "fail":
                failures.append(
                    f"ARIMA: OWA {row['owa']:.4f} outside plausibility band "
                    f"[{low:.2f}, {high:.2f}]"
                )
        elif source == "Naive2":
            gate_tier = "identity"
            delta_owa = abs(float(row["owa"]) - 1.0)
            gate_status = "pass" if delta_owa <= 1e-12 else "fail"
            if gate_status == "fail":
                failures.append(f"Naive2/{domain}: OWA identity failed ({row['owa']:.16f})")

        output = row.to_dict()
        output.update({
            "gate_tier": gate_tier,
            "published_owa": published_owa,
            "delta_owa": delta_owa,
            "gate_status": gate_status,
        })
        rows.append(output)

    gated = pd.DataFrame(rows)
    if failures:
        raise AssertionError("Anchor gate failed:\n" + "\n".join(failures))
    return gated


def stage_anchors(tag: str) -> dict[str, Any]:
    M4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series_list = load_yearly_series()
    scored_frames = []

    for submission_id in REQUIRED_SUBMISSIONS:
        forecasts = load_submission_forecasts(_submission_csv_path(submission_id), horizon=6)
        scored_frames.append(score_forecasts(forecasts, series_list, submission_id))

    scores = pd.concat(scored_frames, ignore_index=True)
    scores_path = M4_RESULTS_DIR / f"submission_scores_full_{tag}.csv"
    scores.to_csv(scores_path, index=False)

    anchors = _aggregate_scores(scores)
    anchors = _apply_anchor_gate(anchors)
    anchors_path = M4_RESULTS_DIR / f"anchors_full_{tag}.csv"
    anchors.to_csv(anchors_path, index=False)

    all_rows = anchors[anchors["domain"] == "All"].copy()
    return {
        "scores_path": str(scores_path),
        "anchors_path": str(anchors_path),
        "n_per_series_scores": len(scores),
        "all_domain_owa": {
            str(row["source"]): float(row["owa"])
            for _, row in all_rows.iterrows()
        },
    }


def _sample_ids_path(tag: str) -> Path:
    return M4_RESULTS_DIR / f"sample_ids_{tag}.txt"


def _sample_series_by_domain(
    series_list: list[M4Series],
    tag: str,
    seed: int,
    n_per_domain: int | None,
    full: bool,
    force: bool,
) -> list[M4Series]:
    M4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = _sample_ids_path(tag)
    series_by_id = {series.series_id: series for series in series_list}

    if sample_path.exists() and not force:
        ids = [line.strip() for line in sample_path.read_text(encoding="utf-8").splitlines()]
        ids = [series_id for series_id in ids if series_id]
        return [series_by_id[series_id] for series_id in ids]

    if full:
        sampled = series_list
    else:
        if n_per_domain is None:
            raise ValueError("--n-per-domain is required for grnn unless --full is set")
        rng = np.random.default_rng(seed)
        chosen_ids: set[str] = set()
        for domain in DOMAIN_COUNTS:
            domain_series = [series for series in series_list if series.category == domain]
            if n_per_domain > len(domain_series):
                raise ValueError(
                    f"--n-per-domain {n_per_domain} exceeds {domain} count "
                    f"{len(domain_series)}"
                )
            chosen = rng.choice(len(domain_series), size=n_per_domain, replace=False)
            chosen_ids.update(domain_series[index].series_id for index in chosen)
        sampled = [series for series in series_list if series.series_id in chosen_ids]

    sample_path.write_text(
        "\n".join(series.series_id for series in sampled) + "\n",
        encoding="utf-8",
    )
    return sampled


def _diff_flags(differencing: str) -> list[bool]:
    if differencing == "both":
        return [False, True]
    return [differencing == "on"]


def _existing_result_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=list(RESULT_KEY_COLUMNS))
    return {
        tuple(str(row[column]) for column in RESULT_KEY_COLUMNS)
        for _, row in frame.iterrows()
    }


def _append_csv_row(path: Path, columns: tuple[str, ...], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(columns))
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def _bagging_seeds(seed: int) -> list[int]:
    return [seed + 1000 * index for index in range(BAGGING_RUNS)]


def _aggregate_bagging_rows(
    rows: list[dict[str, Any]], seed: int, series: M4Series
) -> dict[str, Any]:
    """Collapses per-seed bagging rows into one row for the seed-averaged forecast.

    `grnn_bagging` denotes the seed-averaged forecast (that is what a bagged
    ensemble *is*), so metrics are recomputed from the averaged F1..F6 rather
    than averaged across the per-seed metrics — sMAPE/MAE etc. are nonlinear
    in the forecast, so `metric(mean forecast) != mean(metric per seed)`.
    Per-seed spread is kept separately in smape_std/mase_std.
    """
    first = rows[0].copy()
    forecast_columns = tuple(f"F{index}" for index in range(1, 7))
    avg_forecast = np.array(
        [np.mean([row[column] for row in rows]) for column in forecast_columns],
        dtype=np.float64,
    )
    for column, value in zip(forecast_columns, avg_forecast):
        first[column] = float(value)
    first["smape"] = smape(series.test, avg_forecast)
    first["mase"] = mase(series.train, series.test, avg_forecast, series.m)
    first.update(compute_metrics(series.test, avg_forecast))
    first["runtime_ms"] = float(np.sum([row["runtime_ms"] for row in rows]))
    first["seed"] = seed
    first["bagging_seed_count"] = len(rows)
    first["smape_std"] = float(np.std([row["smape"] for row in rows], ddof=0))
    first["mase_std"] = float(np.std([row["mase"] for row in rows], ddof=0))
    return first


def _score_sample_naive2(series_list: list[M4Series]) -> pd.DataFrame:
    forecasts = load_submission_forecasts(_submission_csv_path("Naive2"), horizon=6)
    return score_forecasts(forecasts, series_list, "Naive2")


def _write_grnn_summary(
    results_path: Path,
    series_list: list[M4Series],
    tag: str,
) -> tuple[Path, dict[str, float]]:
    results = pd.read_csv(results_path)
    sample_ids = {series.series_id for series in series_list}
    results = results[results["series_id"].isin(sample_ids)]
    naive2 = _score_sample_naive2(series_list)
    rows = []

    for domain in ["All", *DOMAIN_COUNTS.keys()]:
        domain_naive2 = naive2 if domain == "All" else naive2[naive2["domain"] == domain]
        naive2_smape = float(domain_naive2["smape"].mean())
        naive2_mase = float(domain_naive2["mase"].mean())
        domain_results = results if domain == "All" else results[results["domain"] == domain]

        for (model, diff), group in domain_results.groupby(["model", "diff"], sort=True):
            mean_smape = float(group["smape"].mean())
            mean_mase = float(group["mase"].mean())
            rows.append({
                "domain": domain,
                "model": model,
                "diff": diff,
                "n": int(len(group)),
                "smape": mean_smape,
                "mase": mean_mase,
                "owa": owa(mean_smape, mean_mase, naive2_smape, naive2_mase),
                "naive2_smape": naive2_smape,
                "naive2_mase": naive2_mase,
            })

    summary = pd.DataFrame(rows)
    summary_path = M4_RESULTS_DIR / f"grnn_summary_{tag}.csv"
    summary.to_csv(summary_path, index=False)
    all_owa = {
        f"{row['model']}:{row['diff']}": float(row["owa"])
        for _, row in summary[summary["domain"] == "All"].iterrows()
    }
    return summary_path, all_owa


def stage_grnn(
    tag: str,
    seed: int,
    n_per_domain: int | None,
    full: bool,
    differencing: str,
    force: bool,
    config_hash: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    M4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series_list = load_yearly_series()
    sampled = _sample_series_by_domain(
        series_list=series_list,
        tag=tag,
        seed=seed,
        n_per_domain=n_per_domain,
        full=full,
        force=force,
    )

    results_path = M4_RESULTS_DIR / f"grnn_results_{tag}.csv"
    skipped_path = M4_RESULTS_DIR / f"grnn_skipped_{tag}.csv"
    if force:
        results_path.unlink(missing_ok=True)
        skipped_path.unlink(missing_ok=True)

    completed = _existing_result_keys(results_path)
    scored = 0
    skipped = 0
    total_cells = len(sampled) * len(_diff_flags(differencing)) * len(GRNN_FAMILY_MODEL_KEYS)
    print(
        f"grnn stage target: {len(sampled)} series x "
        f"{len(_diff_flags(differencing))} diff variant(s) x "
        f"{len(GRNN_FAMILY_MODEL_KEYS)} models = {total_cells} cells"
    )

    for series in sampled:
        for difference in _diff_flags(differencing):
            diff_label = "on" if difference else "off"
            for model_key in GRNN_FAMILY_MODEL_KEYS:
                key = (series.series_id, model_key, diff_label)
                if key in completed:
                    continue

                try:
                    if model_key == "grnn_bagging":
                        seed_rows = []
                        selected_config = None
                        for bagging_seed in _bagging_seeds(seed):
                            row = evaluate_grnn_series(
                                series=series,
                                model_key=model_key,
                                config=config,
                                difference=difference,
                                bagging_seed=bagging_seed,
                                selected_config=selected_config,
                            )
                            if row is None:
                                raise ValueError("infeasible (length/M)")
                            selected_config = (float(row["sigma"]), float(row["M"]))
                            seed_rows.append(row)
                        result = _aggregate_bagging_rows(seed_rows, seed, series)
                    else:
                        result = evaluate_grnn_series(
                            series=series,
                            model_key=model_key,
                            config=config,
                            difference=difference,
                        )
                        if result is None:
                            raise ValueError("infeasible (length/M)")
                        result["bagging_seed_count"] = 0
                        result["smape_std"] = ""
                        result["mase_std"] = ""

                    result["config_hash"] = config_hash
                    result["skipped_reason"] = ""
                    _append_csv_row(results_path, GRNN_RESULT_COLUMNS, result)
                    completed.add(key)
                    scored += 1
                    if (len(completed) % 100) == 0:
                        print(f"  completed {len(completed)} / {total_cells} cells")
                except INFEASIBLE_ERRORS as error:
                    _append_csv_row(
                        skipped_path,
                        SKIP_COLUMNS,
                        {
                            "series_id": series.series_id,
                            "domain": series.category,
                            "model": model_key,
                            "diff": diff_label,
                            "skipped_reason": repr(error),
                        },
                    )
                    skipped += 1

    summary_path, all_owa = _write_grnn_summary(results_path, sampled, tag)
    return {
        "results_path": str(results_path),
        "skipped_path": str(skipped_path),
        "summary_path": str(summary_path),
        "sample_ids_path": str(_sample_ids_path(tag)),
        "n_series": len(sampled),
        "n_scored_new": scored,
        "n_skipped_new": skipped,
        "all_domain_owa": all_owa,
    }


def _read_sample_ids(tag: str) -> list[str]:
    sample_path = _sample_ids_path(tag)
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Missing {sample_path}; run the grnn stage for tag '{tag}' first."
        )
    ids = [line.strip() for line in sample_path.read_text(encoding="utf-8").splitlines()]
    ids = [series_id for series_id in ids if series_id]
    if not ids:
        raise ValueError(f"{sample_path} is empty.")
    return ids


def _load_sampled_series(tag: str) -> list[M4Series]:
    sample_ids = _read_sample_ids(tag)
    series_by_id = {series.series_id: series for series in load_yearly_series()}
    missing = [series_id for series_id in sample_ids if series_id not in series_by_id]
    if missing:
        raise ValueError(f"Sample IDs missing from M4 Yearly data: {missing[:5]}")
    return [series_by_id[series_id] for series_id in sample_ids]


def _score_official_submissions(series_list: list[M4Series]) -> pd.DataFrame:
    frames = []
    for submission_id in REQUIRED_SUBMISSIONS:
        forecasts = load_submission_forecasts(_submission_csv_path(submission_id), horizon=6)
        frame = score_forecasts(forecasts, series_list, submission_id)
        frame["method"] = frame["source"].map(OFFICIAL_TABLE_LABELS)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _sample_metadata(series_list: list[M4Series]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "series_id": series.series_id,
            "domain": series.category,
            "train_length": len(series.train),
        }
        for series in series_list
    ])


def _display_domain(domain: str, n_series: int) -> str:
    if domain != "All":
        return domain
    if n_series == sum(DOMAIN_COUNTS.values()):
        return "All (full 23k)"
    return "All (sample)"


def _domain_order() -> list[str]:
    return [*DOMAIN_COUNTS.keys(), "All"]


def _metric_by_method(
    frame: pd.DataFrame,
    domain: str,
    diff: str,
    metric: str,
) -> dict[str, float]:
    subset = frame if domain == "All" else frame[frame["domain"] == domain]
    subset = subset[subset["diff"] == diff]
    values = {}
    for method, group in subset.groupby("method", sort=False):
        values[str(method)] = float(group[metric].mean())
    return values


def _naive2_denominators(
    official_scores: pd.DataFrame,
    domain: str,
) -> tuple[float, float]:
    subset = official_scores[official_scores["source"] == "Naive2"]
    if domain != "All":
        subset = subset[subset["domain"] == domain]
    return float(subset["smape"].mean()), float(subset["mase"].mean())


def _build_metric_table(
    combined_scores: pd.DataFrame,
    official_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    metric: str,
    diff_labels: list[str],
    full_n_series: int,
) -> pd.DataFrame:
    rows = []
    for domain in _domain_order():
        domain_meta = metadata if domain == "All" else metadata[metadata["domain"] == domain]
        n = int(len(domain_meta))
        median_length = float(domain_meta["train_length"].median())
        for diff in diff_labels:
            row = {
                "Domain": _display_domain(domain, full_n_series),
                "n": n,
                "Med. len": int(median_length) if median_length.is_integer() else median_length,
                "Diff": diff,
            }
            if metric == "owa":
                naive2_smape, naive2_mase = _naive2_denominators(official_scores, domain)
                for method in TABLE_METHOD_COLUMNS:
                    if method == "Naive2":
                        row[method] = 1.0
                        continue
                    method_subset = combined_scores
                    if domain != "All":
                        method_subset = method_subset[method_subset["domain"] == domain]
                    method_subset = method_subset[
                        (method_subset["diff"] == diff)
                        & (method_subset["method"] == method)
                    ]
                    if method_subset.empty:
                        row[method] = float("nan")
                    else:
                        mean_smape = float(method_subset["smape"].mean())
                        mean_mase = float(method_subset["mase"].mean())
                        row[method] = owa(mean_smape, mean_mase, naive2_smape, naive2_mase)
            else:
                metric_values = _metric_by_method(combined_scores, domain, diff, metric)
                for method in TABLE_METHOD_COLUMNS:
                    row[method] = metric_values.get(method, float("nan"))
            rows.append(row)
    return pd.DataFrame(rows, columns=[*TABLE_BASE_COLUMNS, *TABLE_METHOD_COLUMNS])


def _holm_adjust(p_values: list[float]) -> list[float]:
    indexed = [
        (index, 1.0 if pd.isna(p_value) else float(p_value))
        for index, p_value in enumerate(p_values)
    ]
    ordered = sorted(indexed, key=lambda item: item[1])
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    m = len(ordered)
    for rank, (original_index, p_value) in enumerate(ordered):
        holm_value = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, holm_value)
        adjusted[original_index] = running_max
    return adjusted


def _wilcoxon_significance(
    combined_scores: pd.DataFrame,
    diff_labels: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    target_method = "GRNN-Median"
    for domain in _domain_order():
        domain_scores = combined_scores if domain == "All" else combined_scores[
            combined_scores["domain"] == domain
        ]
        for diff in diff_labels:
            diff_scores = domain_scores[domain_scores["diff"] == diff]
            wide = diff_scores.pivot_table(
                index="series_id",
                columns="method",
                values="mase",
                aggfunc="mean",
            )
            if target_method not in wide.columns:
                raise ValueError(f"Missing {target_method} scores for {domain}/{diff}.")

            pending = []
            for competitor in TABLE_METHOD_COLUMNS:
                if competitor == target_method:
                    continue
                if competitor not in wide.columns:
                    continue
                paired = wide[[target_method, competitor]].dropna()
                n_pairs = int(len(paired))
                if n_pairs < 2:
                    statistic = float("nan")
                    p_value = float("nan")
                    median_diff = float("nan")
                else:
                    diffs = paired[target_method] - paired[competitor]
                    median_diff = float(diffs.median())
                    if np.allclose(diffs.to_numpy(), 0.0):
                        statistic = float("nan")
                        p_value = 1.0
                    else:
                        statistic_raw, p_value_raw = stats.wilcoxon(
                            paired[target_method],
                            paired[competitor],
                            zero_method="wilcox",
                            alternative="two-sided",
                        )
                        statistic = float(statistic_raw)
                        p_value = float(p_value_raw)
                pending.append({
                    "domain": domain,
                    "diff": diff,
                    "target": target_method,
                    "competitor": competitor,
                    "n_pairs": n_pairs,
                    "statistic": statistic,
                    "p_value": p_value,
                    "median_diff_mase": median_diff,
                })

            adjusted = _holm_adjust([row["p_value"] for row in pending])
            for row, p_adjusted in zip(pending, adjusted):
                row["p_holm"] = p_adjusted
                row["significant_holm"] = bool(p_adjusted <= alpha)
                median_diff = row["median_diff_mase"]
                if pd.isna(median_diff) or not row["significant_holm"]:
                    row["direction"] = "indistinguishable"
                elif median_diff < 0:
                    row["direction"] = "target_better"
                else:
                    row["direction"] = "target_worse"
                rows.append(row)
    return pd.DataFrame(rows)


def _significance_lookup(significance: pd.DataFrame) -> dict[tuple[str, str, str], bool]:
    lookup = {}
    for _, row in significance.iterrows():
        domain = _display_domain(str(row["domain"]), sum(DOMAIN_COUNTS.values()))
        key = (domain, str(row["diff"]), str(row["competitor"]))
        lookup[key] = bool(row["significant_holm"])
    return lookup


def _format_metric(value: Any, decimals: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}"


def _write_markdown_table(
    table: pd.DataFrame,
    path: Path,
    title: str,
    significance: pd.DataFrame | None = None,
) -> None:
    significance_marks = _significance_lookup(significance) if significance is not None else {}
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        cells = []
        for column in table.columns:
            value = row[column]
            if column in TABLE_METHOD_COLUMNS:
                text = _format_metric(value)
                if significance_marks.get((str(row["Domain"]), str(row["Diff"]), column), False):
                    text = f"**{text}**"
                cells.append(text)
            elif column == "Med. len":
                cells.append(str(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "Notes:",
        "- Headline values are OWA unless the filename states sMAPE or MASE.",
        "- GRNN columns were executed by this pipeline; the other columns are official M4 submitted forecasts scored by the same metric code.",
        "- Submission columns repeat across `off`/`on` rows because differencing is only a GRNN experimental axis.",
        "- Bold cells differ from `GRNN-Median` in the same domain/diff row by paired Wilcoxon signed-rank on per-series MASE after Holm correction at alpha=0.05.",
        "- `GRNN-Median` uses the robust GRNN implementation key `grnn_robust`; M is selected per series from {3, 5, 7} by within-train validation RMSE.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_significance_markdown(significance: pd.DataFrame, path: Path) -> None:
    official = {"118-Smyl", "245-FFORMA", "Com", "ARIMA", "MLP", "Naive2"}
    rows = significance[
        (significance["diff"] == "on")
        & (significance["competitor"].isin(official))
    ].copy()
    lines = [
        "# M4 GRNN-Median Significance Summary",
        "",
        "Paired Wilcoxon signed-rank tests on per-series MASE, Holm-corrected within each domain/diff family.",
        "",
        "| Domain | Competitor | n | p_holm | Direction |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in rows.sort_values(["domain", "competitor"]).iterrows():
        lines.append(
            "| "
            + " | ".join([
                str(row["domain"]),
                str(row["competitor"]),
                str(int(row["n_pairs"])),
                _format_metric(row["p_holm"], decimals=6),
                str(row["direction"]),
            ])
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _verify_metrics_match_forecasts(
    grnn: pd.DataFrame, series_list: list[M4Series]
) -> None:
    """Guards against a bookkeeping bug where a stored metric column and the
    stored F1..F6 forecast disagree about which estimator a row describes.
    """
    series_by_id = {series.series_id: series for series in series_list}
    forecast_columns = [f"F{index}" for index in range(1, 7)]
    mismatches = []
    for row in grnn.itertuples(index=False):
        series = series_by_id[row.series_id]
        forecast = np.array(
            [getattr(row, column) for column in forecast_columns], dtype=np.float64
        )
        expected_smape = smape(series.test, forecast)
        expected_mase = mase(series.train, series.test, forecast, series.m)
        if not np.isclose(row.smape, expected_smape, rtol=1e-6, atol=1e-6):
            mismatches.append(
                f"{row.series_id}/{row.model}/{row.diff}: stored smape={row.smape} "
                f"!= metric(F1..F6)={expected_smape}"
            )
        if not np.isclose(row.mase, expected_mase, rtol=1e-6, atol=1e-6):
            mismatches.append(
                f"{row.series_id}/{row.model}/{row.diff}: stored mase={row.mase} "
                f"!= metric(F1..F6)={expected_mase}"
            )
    if mismatches:
        preview = "\n".join(mismatches[:10])
        raise ValueError(
            f"{len(mismatches)} row(s) with stored metrics disagreeing with "
            f"metric(F1..F6) — the row's metric column and its forecast "
            f"describe different estimators:\n{preview}"
        )


def stage_table(tag: str) -> dict[str, Any]:
    M4_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    results_path = M4_RESULTS_DIR / f"grnn_results_{tag}.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing {results_path}; run `--tag {tag} --stages grnn` first."
        )

    series_list = _load_sampled_series(tag)
    sample_ids = {series.series_id for series in series_list}
    metadata = _sample_metadata(series_list)

    grnn = pd.read_csv(results_path)
    grnn = grnn[grnn["series_id"].isin(sample_ids)].copy()
    grnn["method"] = grnn["model"].map(GRNN_TABLE_LABELS)
    if grnn["method"].isna().any():
        missing = sorted(grnn.loc[grnn["method"].isna(), "model"].unique())
        raise ValueError(f"Unmapped GRNN model keys in {results_path}: {missing}")
    _verify_metrics_match_forecasts(grnn, series_list)
    diff_labels = [label for label in ["off", "on"] if label in set(grnn["diff"])]

    official = _score_official_submissions(series_list)
    official_repeated = []
    for diff in diff_labels:
        repeated = official.copy()
        repeated["diff"] = diff
        official_repeated.append(repeated)
    official_long = pd.concat(official_repeated, ignore_index=True)

    combined = pd.concat([
        grnn[["series_id", "domain", "method", "diff", "smape", "mase"]],
        official_long[["series_id", "domain", "method", "diff", "smape", "mase"]],
    ], ignore_index=True)

    owa_table = _build_metric_table(
        combined_scores=combined,
        official_scores=official,
        metadata=metadata,
        metric="owa",
        diff_labels=diff_labels,
        full_n_series=len(sample_ids),
    )
    smape_table = _build_metric_table(
        combined_scores=combined,
        official_scores=official,
        metadata=metadata,
        metric="smape",
        diff_labels=diff_labels,
        full_n_series=len(sample_ids),
    )
    mase_table = _build_metric_table(
        combined_scores=combined,
        official_scores=official,
        metadata=metadata,
        metric="mase",
        diff_labels=diff_labels,
        full_n_series=len(sample_ids),
    )
    significance = _wilcoxon_significance(combined, diff_labels=diff_labels)

    # Tag-scoped paths are the primary output of this stage. The UNTAGGED
    # `m4_domain_comparison*` files are the canonical copies of the headline
    # tables; they are only ever refreshed by a `main` run, so an
    # exploratory/ablation tag can never silently overwrite the headline
    # numbers.
    table_csv = M4_FINAL_DIR / f"m4_domain_comparison_{tag}.csv"
    table_md = M4_FINAL_DIR / f"m4_domain_comparison_{tag}.md"
    smape_csv = M4_FINAL_DIR / f"m4_domain_comparison_smape_{tag}.csv"
    smape_md = M4_FINAL_DIR / f"m4_domain_comparison_smape_{tag}.md"
    mase_csv = M4_FINAL_DIR / f"m4_domain_comparison_mase_{tag}.csv"
    mase_md = M4_FINAL_DIR / f"m4_domain_comparison_mase_{tag}.md"
    significance_csv = M4_FINAL_DIR / f"m4_significance_{tag}.csv"
    significance_md = M4_FINAL_DIR / f"m4_significance_{tag}.md"

    owa_table.to_csv(table_csv, index=False, float_format="%.12g")
    smape_table.to_csv(smape_csv, index=False, float_format="%.12g")
    mase_table.to_csv(mase_csv, index=False, float_format="%.12g")
    significance.to_csv(significance_csv, index=False, float_format="%.12g")
    _write_markdown_table(owa_table, table_md, "M4 Yearly Domain Comparison", significance)
    _write_markdown_table(smape_table, smape_md, "M4 Yearly Domain Comparison: sMAPE")
    _write_markdown_table(mase_table, mase_md, "M4 Yearly Domain Comparison: MASE")
    _write_significance_markdown(significance, significance_md)

    canonical_paths: dict[str, str] = {}
    if tag == "main":
        canonical = {
            M4_FINAL_DIR / "m4_domain_comparison.csv": table_csv,
            M4_FINAL_DIR / "m4_domain_comparison.md": table_md,
            M4_FINAL_DIR / "m4_domain_comparison_smape.csv": smape_csv,
            M4_FINAL_DIR / "m4_domain_comparison_smape.md": smape_md,
            M4_FINAL_DIR / "m4_domain_comparison_mase.csv": mase_csv,
            M4_FINAL_DIR / "m4_domain_comparison_mase.md": mase_md,
        }
        for canonical_path, source_path in canonical.items():
            canonical_path.write_bytes(source_path.read_bytes())
            canonical_paths[canonical_path.name] = str(canonical_path)

    return {
        "table_csv": str(table_csv),
        "table_md": str(table_md),
        "smape_csv": str(smape_csv),
        "smape_md": str(smape_md),
        "mase_csv": str(mase_csv),
        "mase_md": str(mase_md),
        "significance_csv": str(significance_csv),
        "significance_md": str(significance_md),
        "canonical_paths_updated": canonical_paths,
        "n_series": len(sample_ids),
        "n_table_rows": len(owa_table),
        "n_significance_tests": len(significance),
    }


def _load_grnn_forecasts_for_plots(tag: str, diff: str = "on") -> pd.DataFrame:
    results_path = M4_RESULTS_DIR / f"grnn_results_{tag}.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing {results_path}; run the grnn stage for tag '{tag}' first."
        )
    columns = [
        "series_id", "domain", "model", "diff", "mase",
        "F1", "F2", "F3", "F4", "F5", "F6",
    ]
    frame = pd.read_csv(results_path, usecols=columns)
    frame = frame[frame["diff"] == diff].copy()
    frame["method"] = frame["model"].map(GRNN_TABLE_LABELS)
    if frame["method"].isna().any():
        missing = sorted(frame.loc[frame["method"].isna(), "model"].unique())
        raise ValueError(f"Unmapped GRNN model keys for plots: {missing}")
    return frame


def _official_forecasts_for_plots(series_list: list[M4Series]) -> pd.DataFrame:
    rows = []
    series_by_id = {series.series_id: series for series in series_list}
    for submission_id in REQUIRED_SUBMISSIONS:
        method = OFFICIAL_TABLE_LABELS[submission_id]
        forecasts = load_submission_forecasts(_submission_csv_path(submission_id), horizon=6)
        scores = score_forecasts(forecasts, series_list, submission_id).set_index("series_id")
        for series_id, series in series_by_id.items():
            forecast = forecasts[series_id]
            score = scores.loc[series_id]
            row = {
                "series_id": series_id,
                "domain": series.category,
                "method": method,
                "mase": float(score["mase"]),
            }
            row.update({f"F{index + 1}": float(value) for index, value in enumerate(forecast)})
            rows.append(row)
    return pd.DataFrame(rows)


def _combined_plot_scores(
    grnn_on: pd.DataFrame,
    official: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["series_id", "domain", "method", "mase"]
    return pd.concat([grnn_on[columns], official[columns]], ignore_index=True)


def _select_exemplars(plot_scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    per_series = (
        plot_scores
        .groupby(["domain", "series_id"], as_index=False)["mase"]
        .mean()
        .rename(columns={"mase": "mean_mase"})
    )
    percentiles = [("p25_easy", 0.25), ("p50_median", 0.50), ("p75_hard", 0.75), ("p95_hard", 0.95)]
    selected_rows = []
    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAIN_COUNTS:
        domain_scores = (
            per_series[per_series["domain"] == domain]
            .sort_values(["mean_mase", "series_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        if domain_scores.empty:
            raise ValueError(f"No plot scores available for domain {domain}.")
        manifest_rows[domain] = []
        for label, quantile in percentiles:
            index = int(round((len(domain_scores) - 1) * quantile))
            row = domain_scores.iloc[index].to_dict()
            row["percentile_label"] = label
            row["percentile"] = quantile
            selected_rows.append(row)
            manifest_rows[domain].append({
                "series_id": str(row["series_id"]),
                "percentile_label": label,
                "percentile": quantile,
                "mean_mase": float(row["mean_mase"]),
            })
    return pd.DataFrame(selected_rows), manifest_rows


def _forecast_map(frame: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    forecast_columns = [f"F{index}" for index in range(1, 7)]
    return {
        (str(row["series_id"]), str(row["method"])): row[forecast_columns].to_numpy(dtype=float)
        for _, row in frame.iterrows()
    }


def _plot_prediction_exemplars(
    tag: str,
    series_list: list[M4Series],
    exemplars: pd.DataFrame,
    grnn_on: pd.DataFrame,
    official: pd.DataFrame,
) -> list[str]:
    import matplotlib.pyplot as plt

    series_by_id = {series.series_id: series for series in series_list}
    forecasts = _forecast_map(pd.concat([
        grnn_on[["series_id", "method", "F1", "F2", "F3", "F4", "F5", "F6"]],
        official[["series_id", "method", "F1", "F2", "F3", "F4", "F5", "F6"]],
    ], ignore_index=True))
    paths = []
    methods = list(TABLE_METHOD_COLUMNS)
    for domain in DOMAIN_COUNTS:
        selected = exemplars[exemplars["domain"] == domain].reset_index(drop=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
        axes_flat = axes.ravel()
        for axis, (_, exemplar) in zip(axes_flat, selected.iterrows()):
            series = series_by_id[str(exemplar["series_id"])]
            tail = series.train[-min(len(series.train), 24):]
            train_start = len(series.train) - len(tail)
            train_x = np.arange(train_start + 1, len(series.train) + 1)
            horizon_x = np.arange(len(series.train) + 1, len(series.train) + len(series.test) + 1)

            axis.plot(train_x, tail, color="#1F1F1F", linewidth=1.6, label="Train")
            axis.plot(horizon_x, series.test, color="#000000", linewidth=2.0, label="Actual")
            axis.axvspan(horizon_x[0] - 0.5, horizon_x[-1] + 0.5, color="#D9EAF7", alpha=0.35)

            for method in methods:
                forecast = forecasts.get((series.series_id, method))
                if forecast is None:
                    continue
                style = M4_METHOD_STYLE[method]
                axis.plot(horizon_x, forecast, label=method, **style)

            title = (
                f"{series.series_id} · {exemplar['percentile_label']} · "
                f"mean MASE {float(exemplar['mean_mase']):.2f}"
            )
            axis.set_title(title)
            axis.set_xlabel("Time index")
            axis.set_ylabel("Value")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
        fig.suptitle(f"M4 Yearly {domain}: deterministic exemplar forecasts", y=0.98)
        fig.subplots_adjust(bottom=0.16, top=0.90, hspace=0.34, wspace=0.20)
        path = M4_PLOTS_DIR / f"predictions_{domain}_{tag}.png"
        save_m4_figure(fig, path)
        paths.append(str(path))
    return paths


def _plot_owa_by_domain(tag: str, table: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    on_rows = table[table["Diff"] == "on"].copy()
    domains = on_rows["Domain"].tolist()
    methods = list(TABLE_METHOD_COLUMNS)
    x = np.arange(len(domains), dtype=float)
    width = 0.075

    fig, axis = plt.subplots(figsize=(13, 5))
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2.0) * width
        values = on_rows[method].to_numpy(dtype=float)
        color = M4_METHOD_STYLE[method]["color"]
        axis.bar(x + offset, values, width=width, label=method, color=color)
    axis.axhline(1.0, color="#4D4D4D", linewidth=1.0, linestyle=":")
    axis.set_xticks(x)
    axis.set_xticklabels(domains, rotation=25, ha="right")
    axis.set_ylabel("OWA (lower is better)")
    axis.set_title("M4 Yearly OWA by domain, differencing on for GRNN")
    axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.subplots_adjust(bottom=0.32)
    path = M4_PLOTS_DIR / f"owa_by_domain_{tag}.png"
    save_m4_figure(fig, path)
    return str(path)


def _plot_diff_effect(tag: str, table: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    domains = table[table["Diff"] == "on"]["Domain"].tolist()
    grnn_methods = ["GRNN", "GRNN-Mean", "GRNN-Median", "GRNN-Bag"]
    x = np.arange(len(domains), dtype=float)
    width = 0.18

    fig, axis = plt.subplots(figsize=(11, 4.8))
    for index, method in enumerate(grnn_methods):
        on_values = table[table["Diff"] == "on"][method].to_numpy(dtype=float)
        off_values = table[table["Diff"] == "off"][method].to_numpy(dtype=float)
        improvement = off_values - on_values
        offset = (index - (len(grnn_methods) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            improvement,
            width=width,
            label=method,
            color=M4_METHOD_STYLE[method]["color"],
        )
    axis.axhline(0.0, color="#4D4D4D", linewidth=1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(domains, rotation=25, ha="right")
    axis.set_ylabel("OWA(off) - OWA(on)")
    axis.set_title("Effect of differencing on GRNN family OWA")
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.subplots_adjust(bottom=0.30)
    path = M4_PLOTS_DIR / f"diff_effect_{tag}.png"
    save_m4_figure(fig, path)
    return str(path)


def stage_plots(tag: str) -> dict[str, Any]:
    table_path = M4_FINAL_DIR / f"m4_domain_comparison_{tag}.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Missing {table_path}; run `--tag {tag} --stages table` first."
        )

    apply_m4_style()
    M4_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    series_list = _load_sampled_series(tag)
    sample_ids = {series.series_id for series in series_list}

    grnn_on = _load_grnn_forecasts_for_plots(tag, diff="on")
    grnn_on = grnn_on[grnn_on["series_id"].isin(sample_ids)].copy()
    official = _official_forecasts_for_plots(series_list)
    plot_scores = _combined_plot_scores(grnn_on, official)
    exemplars, exemplar_manifest = _select_exemplars(plot_scores)

    exemplar_path = M4_RESULTS_DIR / f"plot_exemplars_{tag}.csv"
    exemplars.to_csv(exemplar_path, index=False, float_format="%.12g")

    table = pd.read_csv(table_path)
    prediction_paths = _plot_prediction_exemplars(
        tag=tag,
        series_list=series_list,
        exemplars=exemplars,
        grnn_on=grnn_on,
        official=official,
    )
    owa_path = _plot_owa_by_domain(tag, table)
    diff_path = _plot_diff_effect(tag, table)

    return {
        "exemplars_path": str(exemplar_path),
        "exemplars": exemplar_manifest,
        "prediction_paths": prediction_paths,
        "owa_by_domain_path": owa_path,
        "diff_effect_path": diff_path,
        "n_prediction_figures": len(prediction_paths),
    }


def feature_pipeline_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """States the feature pipeline in the manifest, including what is ABSENT.

    A manifest that merely omits the feature-expansion field is ambiguous: a
    reader cannot tell "not used" from "not recorded", so the negative marker
    is stated explicitly.

    The marker is derived from the code rather than typed, and the assertion
    below is the reason: `FeaturePipeline` carries exactly one field, the
    fitted scaler. If a future change adds an expansion step, this raises
    instead of silently writing a now-false "feature_expansion: none" into
    every manifest. The manifest thereby records both the feature-expansion
    and sigma-search settings unambiguously.
    """
    stages = [field.name for field in dataclasses.fields(FeaturePipeline)]
    if stages != ["scaler"]:
        raise AssertionError(
            "FeaturePipeline does not consist of a scaler alone "
            f"(fields: {stages}). The manifest's 'feature_expansion: none' "
            "marker would be false — update feature_pipeline_manifest() "
            "before running."
        )

    preparation = config["preparation"]
    evaluation = config["evaluation"]
    grnn = config["grnn"]
    sigma_search = evaluation["sigma_search"]
    signature = inspect.signature(select_sigma_gradient)
    n_starts = signature.parameters["n_starts"].default

    return {
        "differencing": "both variants run (off and on); "
                        f"global default {preparation['difference']}",
        "lag_order": preparation["lag_order"],
        "lag_type": "raw lags of the (optionally differenced) series",
        "scaler": preparation["scaler"],
        "transform_stages": stages,
        "feature_expansion": "none",
        "feature_expansion_evidence":
            f"dataclasses.fields(FeaturePipeline) == {stages}",
        "sigma_search": sigma_search,
        "sigma_bounds": [grnn["sigma_min"], grnn["sigma_max"]],
        "sigma_step": (
            grnn["sigma_step"] if sigma_search == "grid"
            else "ignored in gradient mode"
        ),
        "sigma_gradient_n_starts": (
            n_starts if sigma_search == "gradient" else "not applicable"
        ),
        "validation_fragment": (
            "chronologically last ceil("
            f"{evaluation['validation_size']} * n) lagged samples inside train"
        ),
    }


def append_manifest(manifest: dict[str, Any], tag: str) -> Path:
    M4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = M4_RESULTS_DIR / f"manifest_{tag}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as manifest_file:
            existing = json.load(manifest_file)
        entries = existing if isinstance(existing, list) else [existing]
    else:
        entries = []
    entries.append(manifest)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path



# ---------------------------------------------------------------------------
# stage: strat — stratified re-aggregation (horizon, length, anomaly)
#
# Everything here is post-processing over the stored `grnn_results_<tag>.csv`
# plus a re-score of the official submission files. The GRNN stage is NOT
# re-run: per-step forecasts are already stored as columns F1..F6.
# ---------------------------------------------------------------------------

GENERATED_TABLES_DIR = Path("reports/tables")

STRATIFICATIONS = (
    Stratification(
        name="length",
        column="length_bin",
        order=LENGTH_BIN_LABELS,
        title="Train length",
    ),
    Stratification(
        name="anomaly",
        column="anomaly_bin",
        order=ANOMALY_BIN_LABELS,
        title="AO count",
    ),
)

STRAT_METRICS = (
    ("smape", "sMAPE", 3),
    ("mase", "MASE", 3),
    ("owa_within", "OWA (within-group Naive2)", 4),
    ("owa_global", "OWA (global Naive2)", 4),
)


def _actuals_and_masep(series_list: list[M4Series]) -> pd.DataFrame:
    """Per-series test values, MASE denominator, length and anomaly count."""
    config = load_config()
    z_threshold = config["realdata"]["m4_yearly"]["ao_z_threshold"]
    rows = []
    for series in series_list:
        ao_count = ao_screen(series.train, z_threshold)
        row = {
            "series_id": series.series_id,
            "domain": series.category,
            "train_length": len(series.train),
            "ao_count": int(ao_count),
            "masep": series_masep(series.train, series.m),
        }
        for index, value in enumerate(series.test, start=1):
            row[f"A{index}"] = float(value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["length_bin"] = frame["train_length"].map(length_bin)
    frame["anomaly_bin"] = frame["ao_count"].map(anomaly_bin)
    return frame


def _official_forecast_frame(series_list: list[M4Series]) -> pd.DataFrame:
    """One row per (series, official method) with columns F1..F6."""
    sample_ids = [series.series_id for series in series_list]
    frames = []
    for submission_id in REQUIRED_SUBMISSIONS:
        forecasts = load_submission_forecasts(
            _submission_csv_path(submission_id), horizon=6
        )
        matrix = np.vstack([forecasts[series_id] for series_id in sample_ids])
        frame = pd.DataFrame(matrix, columns=list(STEP_COLUMNS))
        frame.insert(0, "series_id", sample_ids)
        frame.insert(1, "method", OFFICIAL_TABLE_LABELS[submission_id])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _build_step_frame(tag: str) -> tuple[pd.DataFrame, list[str]]:
    """Per-(series, method, diff) frame carrying per-step sMAPE and MASE.

    Official submissions do not depend on differencing — that is our
    preprocessing choice, not theirs — so their rows are replicated across the
    diff labels exactly as `stage_table` does, keeping the two tables
    comparable.
    """
    results_path = M4_RESULTS_DIR / f"grnn_results_{tag}.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing {results_path}; run `--tag {tag} --stages grnn` first."
        )

    series_list = _load_sampled_series(tag)
    sample_ids = {series.series_id for series in series_list}
    meta = _actuals_and_masep(series_list)

    grnn = pd.read_csv(
        results_path,
        usecols=["series_id", "model", "diff", *STEP_COLUMNS],
    )
    grnn = grnn[grnn["series_id"].isin(sample_ids)].copy()
    grnn["method"] = grnn["model"].map(GRNN_TABLE_LABELS)
    if grnn["method"].isna().any():
        missing = sorted(grnn.loc[grnn["method"].isna(), "model"].unique())
        raise ValueError(f"Unmapped GRNN model keys in {results_path}: {missing}")
    grnn = grnn.drop(columns=["model"])

    diff_labels = [label for label in ["off", "on"] if label in set(grnn["diff"])]

    official = _official_forecast_frame(series_list)
    official_repeated = []
    for diff in diff_labels:
        repeated = official.copy()
        repeated["diff"] = diff
        official_repeated.append(repeated)

    frame = pd.concat([grnn, *official_repeated], ignore_index=True)
    frame = frame.merge(meta, on="series_id", how="left", validate="many_to_one")
    if frame["masep"].isna().any():
        raise ValueError("Some scored rows have no matching series metadata.")

    forecasts = frame[list(STEP_COLUMNS)].to_numpy(dtype=np.float64)
    actuals = frame[[f"A{index}" for index in range(1, HORIZON + 1)]].to_numpy(
        dtype=np.float64
    )
    masep = frame["masep"].to_numpy(dtype=np.float64)[:, None]

    absolute_error = np.abs(actuals - forecasts)
    smape_steps = absolute_error * 200.0 / (np.abs(actuals) + np.abs(forecasts))
    mase_steps = absolute_error / masep

    for index in range(HORIZON):
        frame[SMAPE_STEP_COLUMNS[index]] = smape_steps[:, index]
        frame[MASE_STEP_COLUMNS[index]] = mase_steps[:, index]
    frame["smape"] = smape_steps.mean(axis=1)
    frame["mase"] = mase_steps.mean(axis=1)

    frame = frame.drop(columns=[*STEP_COLUMNS, *[f"A{i}" for i in range(1, HORIZON + 1)]])
    return frame, diff_labels


def _horizon_long_table(
    step_frame: pd.DataFrame,
    diff_labels: list[str],
) -> pd.DataFrame:
    """Per-horizon-step re-aggregation, overall and per domain.

    Reshapes the six stored steps into a `step` grouping column and runs the
    same aggregation engine every other stratification uses, so the horizon
    view is not a special case with its own arithmetic.
    """
    step_labels = tuple(f"h{index}" for index in range(1, HORIZON + 1))
    melted_parts = []
    for index in range(HORIZON):
        part = step_frame[["series_id", "domain", "method", "diff"]].copy()
        part["step"] = step_labels[index]
        part["smape"] = step_frame[SMAPE_STEP_COLUMNS[index]].to_numpy()
        part["mase"] = step_frame[MASE_STEP_COLUMNS[index]].to_numpy()
        melted_parts.append(part)
    melted = pd.concat(melted_parts, ignore_index=True)

    tables = []
    for diff in diff_labels:
        diff_rows = melted[melted["diff"] == diff]
        for domain in ["All", *DOMAIN_COUNTS.keys()]:
            subset = diff_rows if domain == "All" else diff_rows[
                diff_rows["domain"] == domain
            ]
            aggregated = aggregate_by_group(
                frame=subset,
                group_column="step",
                group_order=step_labels,
                methods=TABLE_METHOD_COLUMNS,
            )
            aggregated.insert(0, "domain", domain)
            aggregated.insert(1, "diff", diff)
            tables.append(aggregated)
    return pd.concat(tables, ignore_index=True)


def _stratified_long_table(
    step_frame: pd.DataFrame,
    stratification: Stratification,
    diff_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], list[str]]:
    """Re-aggregation over one grouping column, both differencing variants.

    Returns the merged frame as well as the table: once small groups have been
    folded together the significance tests must run on the SAME labels, or the
    merged bin silently disappears from the test output.
    """
    frame, order, merge_log = merge_small_groups(
        frame=step_frame,
        column=stratification.column,
        order=stratification.order,
    )
    tables = []
    for diff in diff_labels:
        aggregated = aggregate_by_group(
            frame=frame[frame["diff"] == diff],
            group_column=stratification.column,
            group_order=order,
            methods=TABLE_METHOD_COLUMNS,
        )
        aggregated.insert(0, "diff", diff)
        tables.append(aggregated)
    return pd.concat(tables, ignore_index=True), frame, order, merge_log


def _grouped_wilcoxon(
    step_frame: pd.DataFrame,
    group_column: str,
    group_order: tuple[str, ...],
    diff_labels: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired Holm-corrected Wilcoxon on per-series MASE inside each group.

    Only within-group comparisons are tested. A "bin A vs bin B" test would be
    unpaired — different series — so it is deliberately absent.
    """
    rows = []
    target_method = "GRNN-Median"
    for group in [*group_order, "All"]:
        group_rows = step_frame if group == "All" else step_frame[
            step_frame[group_column] == group
        ]
        for diff in diff_labels:
            diff_rows = group_rows[group_rows["diff"] == diff]
            wide = diff_rows.pivot_table(
                index="series_id", columns="method", values="mase", aggfunc="mean"
            )
            if target_method not in wide.columns:
                continue

            pending = []
            for competitor in TABLE_METHOD_COLUMNS:
                if competitor == target_method or competitor not in wide.columns:
                    continue
                paired = wide[[target_method, competitor]].dropna()
                n_pairs = int(len(paired))
                if n_pairs < 2:
                    statistic, p_value, median_diff = float("nan"), float("nan"), float("nan")
                else:
                    diffs = paired[target_method] - paired[competitor]
                    median_diff = float(diffs.median())
                    if np.allclose(diffs.to_numpy(), 0.0):
                        statistic, p_value = float("nan"), 1.0
                    else:
                        statistic_raw, p_value_raw = stats.wilcoxon(
                            paired[target_method],
                            paired[competitor],
                            zero_method="wilcox",
                            alternative="two-sided",
                        )
                        statistic = float(statistic_raw)
                        p_value = float(p_value_raw)
                pending.append({
                    "group": group,
                    "diff": diff,
                    "target": target_method,
                    "competitor": competitor,
                    "n_pairs": n_pairs,
                    "statistic": statistic,
                    "p_value": p_value,
                    "median_diff_mase": median_diff,
                })

            for row, p_adjusted in zip(pending, _holm_adjust([r["p_value"] for r in pending])):
                row["p_holm"] = p_adjusted
                row["significant_holm"] = bool(p_adjusted <= alpha)
                median_diff = row["median_diff_mase"]
                if pd.isna(median_diff) or not row["significant_holm"]:
                    row["direction"] = "indistinguishable"
                elif median_diff < 0:
                    row["direction"] = "target_better"
                else:
                    row["direction"] = "target_worse"
                rows.append(row)
    return pd.DataFrame(rows)


def _plot_stratified(
    long_table: pd.DataFrame,
    group_order: tuple[str, ...],
    metric: str,
    metric_label: str,
    x_label: str,
    path: Path,
    diff: str = "on",
) -> None:
    """One line per method across the grouping dimension."""
    import matplotlib.pyplot as plt

    apply_m4_style()
    subset = long_table[long_table["diff"] == diff]
    if "domain" in subset.columns:
        subset = subset[subset["domain"] == "All"]
    groups = [group for group in group_order if group in set(subset["group"])]
    positions = np.arange(len(groups))

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for method in TABLE_METHOD_COLUMNS:
        method_rows = subset[subset["method"] == method].set_index("group")
        values = [
            float(method_rows.loc[group, metric]) if group in method_rows.index
            else np.nan
            for group in groups
        ]
        if all(np.isnan(value) for value in values):
            continue
        axis.plot(
            positions, values, marker="o", markersize=3.5,
            label=method, **M4_METHOD_STYLE.get(method, {}),
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(groups)
    axis.set_xlabel(x_label)
    axis.set_ylabel(metric_label)
    axis.set_title(f"{metric_label} by {x_label.lower()} (differencing {diff})")
    axis.legend(ncol=2, fontsize=7, framealpha=0.9)
    save_m4_figure(fig, path)


def _plot_length_curve(
    step_frame: pd.DataFrame,
    path: Path,
    diff: str = "on",
    min_count: int = 100,
    smooth_window: int = 5,
) -> None:
    """Continuous view: accuracy relative to Naive2 against train length.

    Bins answer "which regime", but they hide monotonicity, so the continuous
    dependency on length is reported alongside the binned table.

    Plotted as the per-series **ratio** to Naive2 on the same series, not raw
    MASE. Raw MASE level is driven by how hard a series is, and difficulty is
    not constant across lengths, so a raw curve mixes the two effects and is
    unreadable. The ratio removes the difficulty confound the same way
    `owa_within` does for the binned table: 1.0 is Naive2, below is better.
    """
    import matplotlib.pyplot as plt

    apply_m4_style()
    subset = step_frame[step_frame["diff"] == diff]

    naive2 = subset[subset["method"] == "Naive2"].set_index("series_id")["mase"]
    naive2 = naive2[naive2 > 0]
    subset = subset[subset["series_id"].isin(naive2.index)].copy()
    subset["mase_ratio"] = (
        subset["mase"].to_numpy()
        / naive2.reindex(subset["series_id"]).to_numpy()
    )

    # Only lengths carrying enough series for a stable median. M4 Yearly has a
    # long, thin tail (max 835 observations, most of it single series), which
    # would otherwise dominate the axis with noise. The retained range covers
    # ~95% of the sample; the coverage is stated in the title rather than
    # trimmed silently.
    counts = subset.drop_duplicates("series_id")["train_length"].value_counts()
    keep = sorted(length for length, count in counts.items() if count >= min_count)
    coverage = counts[keep].sum() / counts.sum() if keep else 0.0

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for method in TABLE_METHOD_COLUMNS:
        method_rows = subset[subset["method"] == method]
        if method_rows.empty:
            continue
        curve = method_rows.groupby("train_length")["mase_ratio"].median().reindex(keep)
        # Centred rolling median over neighbouring lengths — the raw per-length
        # medians jump because adjacent lengths hold disjoint series.
        smoothed = curve.rolling(window=smooth_window, center=True, min_periods=1).median()
        axis.plot(keep, smoothed.to_numpy(), label=method, **M4_METHOD_STYLE.get(method, {}))
    axis.axhline(1.0, color="#4D4D4D", linewidth=0.8, linestyle=":", zorder=0)
    axis.set_xlabel("Train length (observations)")
    axis.set_ylabel("Median per-series MASE / Naive2 MASE")
    axis.set_title(
        f"Accuracy vs available history, differencing {diff}\n"
        f"lengths with n >= {min_count} series ({coverage:.0%} of the sample), "
        f"rolling median over {smooth_window} lengths"
    )
    axis.legend(ncol=2, fontsize=7, framealpha=0.9)
    save_m4_figure(fig, path)


def stage_strat(tag: str) -> dict[str, Any]:
    """Horizon, length and anomaly stratification from the stored results."""
    M4_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    M4_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    step_frame, diff_labels = _build_step_frame(tag)
    outputs: dict[str, Any] = {"diff_labels": diff_labels, "merge_log": []}
    gate_failures: list[str] = []

    # --- Per-horizon-step decomposition ------------------------------------
    step_labels = tuple(f"h{index}" for index in range(1, HORIZON + 1))
    horizon_long = _horizon_long_table(step_frame, diff_labels)
    horizon_csv = M4_FINAL_DIR / f"m4_horizon_decomposition_{tag}.csv"
    horizon_long.to_csv(horizon_csv, index=False, float_format="%.12g")
    outputs["horizon_csv"] = str(horizon_csv)

    for diff in diff_labels:
        overall = horizon_long[
            (horizon_long["domain"] == "All") & (horizon_long["diff"] == diff)
        ]
        gate_failures += [
            f"horizon/diff={diff}: {failure}"
            for failure in verify_group_identity(overall, step_labels)
        ]

    for metric, metric_label, decimals in STRAT_METRICS:
        wide_parts = []
        for diff in diff_labels:
            subset = horizon_long[
                (horizon_long["domain"] == "All") & (horizon_long["diff"] == diff)
            ]
            wide = to_wide(subset, metric, step_labels, TABLE_METHOD_COLUMNS, "Step")
            wide.insert(1, "Diff", diff)
            wide_parts.append(wide)
        table = pd.concat(wide_parts, ignore_index=True)
        base = f"table_horizon_{metric}_{tag}"
        table.to_csv(GENERATED_TABLES_DIR / f"{base}.csv", index=False, float_format="%.12g")
        write_markdown(
            table,
            GENERATED_TABLES_DIR / f"{base}.md",
            f"M4 Yearly — {metric_label} by horizon step",
            note=(
                "Rows h1..h6 are the official evaluation structure with one fewer "
                "averaging step; `All` is the horizon-averaged value already reported. "
                "`owa_within` normalises by Naive2 at the same step (Naive2 = 1 "
                "everywhere); `owa_global` uses the horizon-averaged Naive2, so the "
                "row shows how error accumulates."
            ),
            float_format=lambda value, d=decimals: f"{value:.{d}f}",
        )
        _plot_stratified(
            horizon_long, step_labels, metric, metric_label,
            "Horizon step", M4_PLOTS_DIR / f"horizon_{metric}_{tag}.png",
        )

    # --- Length and anomaly stratification ---------------------------------
    for stratification in STRATIFICATIONS:
        long_table, merged_frame, order, merge_log = _stratified_long_table(
            step_frame, stratification, diff_labels
        )
        outputs["merge_log"] += merge_log
        long_csv = M4_FINAL_DIR / f"m4_{stratification.name}_strat_{tag}.csv"
        long_table.to_csv(long_csv, index=False, float_format="%.12g")
        outputs[f"{stratification.name}_csv"] = str(long_csv)

        for diff in diff_labels:
            gate_failures += [
                f"{stratification.name}/diff={diff}: {failure}"
                for failure in verify_group_identity(
                    long_table[long_table["diff"] == diff], order
                )
            ]

        for metric, metric_label, decimals in STRAT_METRICS:
            wide_parts = []
            for diff in diff_labels:
                wide = to_wide(
                    long_table[long_table["diff"] == diff],
                    metric, order, TABLE_METHOD_COLUMNS, stratification.title,
                )
                wide.insert(1, "Diff", diff)
                wide_parts.append(wide)
            table = pd.concat(wide_parts, ignore_index=True)
            base = f"table_{stratification.name}_{metric}_{tag}"
            table.to_csv(
                GENERATED_TABLES_DIR / f"{base}.csv", index=False, float_format="%.12g"
            )
            write_markdown(
                table,
                GENERATED_TABLES_DIR / f"{base}.md",
                f"M4 Yearly — {metric_label} by {stratification.title.lower()}",
                note=(
                    "Bin edges are pre-registered, not data-driven. `owa_within` "
                    "normalises by Naive2 inside the bin (measures the method against "
                    "the difficulty of that bin); `owa_global` keeps the published "
                    "Naive2 denominator and stays comparable with the leaderboard."
                ),
                float_format=lambda value, d=decimals: f"{value:.{d}f}",
            )
            _plot_stratified(
                long_table, order, metric, metric_label, stratification.title,
                M4_PLOTS_DIR / f"{stratification.name}_{metric}_{tag}.png",
            )

        significance = _grouped_wilcoxon(
            merged_frame, stratification.column, order, diff_labels
        )
        significance_csv = (
            M4_FINAL_DIR / f"m4_significance_{stratification.name}_{tag}.csv"
        )
        significance.to_csv(significance_csv, index=False, float_format="%.12g")
        outputs[f"{stratification.name}_significance_csv"] = str(significance_csv)

    _plot_length_curve(step_frame, M4_PLOTS_DIR / f"length_curve_mase_{tag}.png")
    outputs["length_curve_png"] = str(M4_PLOTS_DIR / f"length_curve_mase_{tag}.png")

    if gate_failures:
        raise AssertionError(
            "Group-identity gate failed (groups must reconstruct the aggregate):\n"
            + "\n".join(gate_failures)
        )
    outputs["identity_gate"] = "pass"
    outputs["n_series"] = int(step_frame["series_id"].nunique())
    return outputs


def main() -> None:
    args = parse_args()
    stages = tuple(stage.strip() for stage in args.stages.split(",") if stage.strip())
    unsupported = [stage for stage in stages if stage not in SUPPORTED_STAGES]
    if unsupported:
        raise ValueError(f"Unsupported stage(s): {', '.join(unsupported)}")

    config = load_config()
    manifest = {
        "tag": args.tag,
        "seed": args.seed,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "feature_pipeline": feature_pipeline_manifest(config),
        "code_state_sha256": code_state_fingerprint(),
        "package_versions": package_versions(),
        "stages": {},
    }

    try:
        for stage in stages:
            started = time.perf_counter()
            if stage == "sync":
                result = sync_forecasts(reports_dir=M4_RESULTS_DIR)
            elif stage == "validate":
                result = stage_validate()
            elif stage == "anchors":
                result = stage_anchors(args.tag)
            elif stage == "grnn":
                result = stage_grnn(
                    tag=args.tag,
                    seed=args.seed,
                    n_per_domain=args.n_per_domain,
                    full=args.full,
                    differencing=args.differencing,
                    force=args.force,
                    config_hash=manifest["code_state_sha256"],
                    config=config,
                )
            elif stage == "table":
                result = stage_table(tag=args.tag)
            elif stage == "strat":
                result = stage_strat(tag=args.tag)
            elif stage == "plots":
                result = stage_plots(tag=args.tag)
            else:
                raise NotImplementedError(
                    f"Stage '{stage}' is not implemented"
                )

            result["wall_clock_s"] = time.perf_counter() - started
            manifest["stages"][stage] = result
            print(f"{stage}: done in {result['wall_clock_s']:.2f}s")
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path = append_manifest(manifest, args.tag)
        print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
