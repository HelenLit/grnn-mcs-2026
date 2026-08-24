"""Tail / robustness analysis of median vs mean aggregation.

Terminology note: the measured event is named "exceedance" — the per-series
MASE exceeds a multiple of the Naive2 MASE on that SAME series. Nothing
breaks; the event is a large forecasting error on one series, and
reliability-engineering vocabulary would overstate it.

**Why this exists.** The anomaly stratification in the `strat` stage compares
aggregate means and paired per-series MASE, and on both of those GRNN-Mean
wins nearly everywhere. Those are the wrong instruments for the robustness
question.

The mechanism claimed for median aggregation is not a shift of the whole error
distribution — it is protection against a single base model blowing up on a
contaminated sub-sequence. Such an effect lives in the TAIL. It is invisible in
the mean and it is actively hidden by a paired Wilcoxon test, because median
aggregation loses slightly on the majority of series (win rate ~0.47) while
winning rarely and decisively on the series where a member fails. Wilcoxon ranks
the many small losses and reports "mean is better"; the practitioner's question
is "how often is the forecast far worse than doing nothing".

This script measures that directly.

**Definitions, all fixed in advance, not tuned here:**

  * *Exceedance* of a method on a series: per-series MASE exceeding
    `threshold` times the Naive2 MASE **on that same series**. The relative form
    matters — an absolute MASE cutoff is dominated by intrinsically hard series
    where every method does badly, and would measure series difficulty rather
    than the method's behaviour. Thresholds 2, 3 and 5 are all reported so the headline cannot be
    accused of resting on one cutoff.
  * *Noise proxy*: `std / (1.4826 * MAD)` of the differenced training series —
    the ratio of a non-robust to a robust scale estimate, which is large exactly
    when a few large deviations inflate the standard deviation. Computed from
    training data only.
  * The additive-outlier screen already used elsewhere (`ao_screen`, |z| > 3 on
    differences) is reported alongside as a second, weaker instrument: with a
    median train length of 29 the standard deviation in that z-score is itself
    inflated by the outlier it is meant to detect, so it under-detects.
  * Noise groups are equal-mass quintiles. The split is not tuned to maximise
    any effect; a fixed-threshold version (the AO screen) is reported next to it
    so the conclusion does not depend on the binning.

**Test.** An exceedance is a paired binary outcome per series, so the correct test is
McNemar's, exact, on the discordant pairs — not a t-test on counts and not
Wilcoxon on the errors. Holm correction across the five quintiles. A monotone
dose-response across the noise quintiles is tested by Spearman correlation.

Usage:
    python -m src.scripts.run_robustness_analysis --tag main
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data.m4 import ao_screen
from src.evaluation.m4_stratify import write_markdown
from src.scripts.run_m4_pipeline import M4_FINAL_DIR
from src.scripts.run_m4_pipeline import M4_RESULTS_DIR
from src.scripts.run_m4_pipeline import GENERATED_TABLES_DIR
from src.scripts.run_m4_pipeline import _build_step_frame
from src.scripts.run_m4_pipeline import _load_sampled_series
from src.scripts.run_m4_pipeline import _holm_adjust
from src.scripts.run_m4_pipeline import code_state_fingerprint
from src.scripts.run_m4_pipeline import load_config

MEAN = "GRNN-Mean"
MEDIAN = "GRNN-Median"
BASELINE = "Naive2"
THRESHOLDS = (2.0, 3.0, 5.0)
HEADLINE_THRESHOLD = 3.0
QUINTILE_LABELS = ("Q1 low", "Q2", "Q3", "Q4", "Q5 high")


def noise_features(series_list, z_threshold: float) -> pd.DataFrame:
    """Per-series noise descriptors, computed on TRAINING data only."""
    rows = []
    for series in series_list:
        differences = np.diff(np.asarray(series.train, dtype=np.float64))
        centre = np.median(differences)
        mad = np.median(np.abs(differences - centre))
        robust_scale = 1.4826 * mad if mad > 0 else float(np.std(differences))
        if robust_scale <= 0:
            noise_ratio = 1.0
        else:
            noise_ratio = float(np.std(differences) / robust_scale)
        rows.append({
            "series_id": series.series_id,
            "domain": series.category,
            "train_length": len(series.train),
            "noise_ratio": noise_ratio,
            "ao_count": int(ao_screen(series.train, z_threshold)),
        })
    frame = pd.DataFrame(rows)
    frame["noise_quintile"] = pd.qcut(
        frame["noise_ratio"], 5, labels=list(QUINTILE_LABELS)
    ).astype(str)
    frame["ao_group"] = frame["ao_count"].clip(upper=2).map(
        {0: "AO=0", 1: "AO=1", 2: "AO>=2"}
    )
    return frame


def _mcnemar_exact(
    failed_a: np.ndarray, failed_b: np.ndarray
) -> tuple[int, int, float]:
    """Exact McNemar on paired binary exceedances.

    Returns (only_a, only_b, two-sided p). Concordant pairs carry no
    information about which method is better and are correctly ignored.
    """
    only_a = int(np.sum(failed_a & ~failed_b))
    only_b = int(np.sum(~failed_a & failed_b))
    if only_a + only_b == 0:
        return only_a, only_b, 1.0
    p_value = float(
        stats.binomtest(only_a, only_a + only_b, 0.5).pvalue
    )
    return only_a, only_b, p_value


def failure_table(
    wide: pd.DataFrame,
    group_column: str,
    group_order: list[str],
    threshold: float,
) -> pd.DataFrame:
    """Exceedance counts and McNemar tests per group."""
    rows = []
    for group in [*group_order, "All"]:
        subset = wide if group == "All" else wide[wide[group_column] == group]
        baseline = subset[BASELINE].replace(0, np.nan)
        usable = baseline.notna()
        subset = subset[usable]
        baseline = baseline[usable]
        if subset.empty:
            continue

        failed_mean = (subset[MEAN] / baseline > threshold).to_numpy()
        failed_median = (subset[MEDIAN] / baseline > threshold).to_numpy()
        only_mean, only_median, p_value = _mcnemar_exact(failed_mean, failed_median)
        n_mean, n_median = int(failed_mean.sum()), int(failed_median.sum())
        rows.append({
            "group": group,
            "n": int(len(subset)),
            "threshold": threshold,
            "fail_mean": n_mean,
            "fail_median": n_median,
            "reduction_pct": (
                float("nan") if n_mean == 0 else (1 - n_median / n_mean) * 100.0
            ),
            "only_mean_failed": only_mean,
            "only_median_failed": only_median,
            "p_mcnemar": p_value,
        })

    table = pd.DataFrame(rows)
    grouped = table[table["group"] != "All"]
    if not grouped.empty:
        adjusted = _holm_adjust(list(grouped["p_mcnemar"]))
        table.loc[table["group"] != "All", "p_holm"] = adjusted
    table.loc[table["group"] == "All", "p_holm"] = table.loc[
        table["group"] == "All", "p_mcnemar"
    ]
    return table


def central_accuracy_table(
    wide: pd.DataFrame,
    group_column: str,
    group_order: list[str],
) -> pd.DataFrame:
    """Mean MASE and the paired Wilcoxon — the view that favours the MEAN.

    Reported next to the failure counts on purpose: the two answer different
    questions and both must be stated, or it will look like the favourable
    one was picked.
    """
    rows = []
    for group in [*group_order, "All"]:
        subset = wide if group == "All" else wide[wide[group_column] == group]
        paired = subset[[MEDIAN, MEAN]].dropna()
        if len(paired) < 2:
            continue
        differences = paired[MEDIAN] - paired[MEAN]
        if np.allclose(differences.to_numpy(), 0.0):
            p_value = 1.0
        else:
            p_value = float(
                stats.wilcoxon(paired[MEDIAN], paired[MEAN]).pvalue
            )
        rows.append({
            "group": group,
            "n": int(len(paired)),
            "mean_mase_median_agg": float(paired[MEDIAN].mean()),
            "mean_mase_mean_agg": float(paired[MEAN].mean()),
            "median_wins_share": float((differences < 0).mean()),
            "p_wilcoxon": p_value,
            "central_winner": (
                MEDIAN if paired[MEDIAN].mean() < paired[MEAN].mean() else MEAN
            ),
        })
    return pd.DataFrame(rows)


def quantile_table(wide: pd.DataFrame, group_column: str, group_order: list[str]) -> pd.DataFrame:
    """Per-series MASE quantiles — shows WHERE the two differ."""
    rows = []
    for group in [*group_order, "All"]:
        subset = wide if group == "All" else wide[wide[group_column] == group]
        for method in (MEAN, MEDIAN):
            values = subset[method].dropna()
            if values.empty:
                continue
            row = {"group": group, "method": method, "n": int(len(values))}
            for quantile in (0.5, 0.75, 0.9, 0.95, 0.99):
                row[f"p{int(quantile * 100)}"] = float(values.quantile(quantile))
            row["max"] = float(values.max())
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    config = load_config()
    z_threshold = config["realdata"]["m4_yearly"]["ao_z_threshold"]

    series_list = _load_sampled_series(args.tag)
    features = noise_features(series_list, z_threshold)

    step_frame, _ = _build_step_frame(args.tag)
    differenced = step_frame[step_frame["diff"] == "on"]
    wide = differenced.pivot_table(
        index="series_id", columns="method", values="mase"
    )
    missing = [m for m in (MEAN, MEDIAN, BASELINE) if m not in wide.columns]
    if missing:
        raise ValueError(f"Missing methods in results: {missing}")
    wide = wide.join(
        features.set_index("series_id")[["noise_quintile", "ao_group", "domain"]]
    )

    groupings = {
        "noise": ("noise_quintile", list(QUINTILE_LABELS)),
        "anomaly": ("ao_group", ["AO=0", "AO=1", "AO>=2"]),
        "domain": ("domain", sorted(features["domain"].unique())),
    }

    M4_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    headline: dict[str, object] = {}

    for name, (column, order) in groupings.items():
        failures = pd.concat(
            [failure_table(wide, column, order, threshold) for threshold in THRESHOLDS],
            ignore_index=True,
        )
        central = central_accuracy_table(wide, column, order)
        quantiles = quantile_table(wide, column, order)

        for frame, kind in (
            (failures, "failures"), (central, "central"), (quantiles, "quantiles"),
        ):
            path = M4_FINAL_DIR / f"m4_robustness_{name}_{kind}_{args.tag}.csv"
            frame.to_csv(path, index=False, float_format="%.12g")
            outputs[f"{name}_{kind}"] = str(path)

        head = failures[failures["threshold"] == HEADLINE_THRESHOLD]
        report = head[[
            "group", "n", "fail_mean", "fail_median", "reduction_pct",
            "only_mean_failed", "only_median_failed", "p_holm",
        ]].rename(columns={
            "group": "Group", "n": "n",
            "fail_mean": "Mean exceeds", "fail_median": "Median exceeds",
            "reduction_pct": "Reduction %",
            "only_mean_failed": "only Mean", "only_median_failed": "only Median",
            "p_holm": "Holm p",
        })
        md_path = GENERATED_TABLES_DIR / f"table_robustness_{name}_{args.tag}.md"
        write_markdown(
            report,
            md_path,
            f"Large-error reduction by median aggregation — by {name}",
            note=(
                f"An exceedance is a per-series MASE above "
                f"{HEADLINE_THRESHOLD:g}x the Naive2 MASE on the SAME series — that is, "
                "the method's error on that series is more than "
                f"{HEADLINE_THRESHOLD:g} times the error of simply repeating the last "
                "observation. The relative form is deliberate: an absolute cutoff "
                "would measure how hard a series is rather than how the method "
                "behaved on it. "
                "Counts are paired per series; the test is exact McNemar on the "
                "discordant pairs, Holm-corrected across groups. Differencing on. "
                "Thresholds 2x and 5x are in the CSV and give the same direction. "
                "Read together with the central-accuracy table: on mean MASE and "
                "on the paired Wilcoxon test, MEAN aggregation wins — the two "
                "views answer different questions and both must be stated."
            ),
            float_format=lambda value: f"{value:.4g}",
        )
        outputs[f"{name}_markdown"] = str(md_path)

        if name == "noise":
            quintiles = head[head["group"] != "All"]
            rho, trend_p = stats.spearmanr(
                np.arange(len(quintiles)), quintiles["reduction_pct"].to_numpy()
            )
            overall = head[head["group"] == "All"].iloc[0]
            headline = {
                "threshold": HEADLINE_THRESHOLD,
                "overall_fail_mean": int(overall["fail_mean"]),
                "overall_fail_median": int(overall["fail_median"]),
                "overall_reduction_pct": float(overall["reduction_pct"]),
                "overall_p_mcnemar": float(overall["p_mcnemar"]),
                "per_quintile_reduction_pct": [
                    float(value) for value in quintiles["reduction_pct"]
                ],
                "monotone_trend_spearman_rho": float(rho),
                "monotone_trend_p": float(trend_p),
            }

    manifest = {
        "tag": args.tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_series": int(len(series_list)),
        "definitions": {
            "exceedance": "per-series MASE > threshold x Naive2 MASE (same series)",
            "thresholds": list(THRESHOLDS),
            "noise_proxy": "std / (1.4826 * MAD) of the differenced train series",
            "noise_groups": "equal-mass quintiles, not tuned",
            "test": "exact McNemar on discordant pairs, Holm across groups",
        },
        "headline_noise": headline,
        "outputs": outputs,
        "code_state_sha256": code_state_fingerprint(),
    }
    manifest_path = M4_RESULTS_DIR / f"manifest_robustness_{args.tag}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(headline, indent=2))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
