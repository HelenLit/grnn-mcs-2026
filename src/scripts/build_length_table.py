"""Accuracy versus training-series length.

A re-aggregation of already-stored per-series forecasts — nothing is
re-trained. All 23,000 M4 Yearly series are grouped by the length of their
training part and the main experiment's metrics (sMAPE and OWA) are
recomputed inside each group for GRNN-Median, GRNN-Mean, ARIMA, Naive2, MLP.

**Group boundaries come from the data, not from round numbers**: the
empirical quartiles of the length distribution are 20, 29 and 40, so the
groups are 13-20 / 21-29 / 30-40 / 41+. They are not exactly 25% each because those
three lengths are themselves mass points (length 20 alone holds 8.1% of the
sample, 29 holds 12.8%, 40 holds 12.9%), which makes an exact quartile split
impossible; the realised shares are 30.6 / 29.2 / 22.9 / 17.3%, and the
smallest group still holds 3,979 series.

OWA inside a group uses that group's own Naive2 means as the denominator —
the same convention the main experiment applies per domain — so Naive2 is
1.000 in every row and each group is judged against its own difficulty.

Usage:
    python -m src.scripts.build_length_table [tag]   # default: main
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.scripts.run_m4_pipeline import _build_step_frame
from src.scripts.run_m4_pipeline import _holm_adjust

OUT_DIR = Path("reports/tables")
FINAL_DIR = Path("reports/final")

# Empirical quartiles of the training-length distribution (20 / 29 / 40).
BINS = [(13, 20), (21, 29), (30, 40), (41, 10**9)]
LABELS = ["13–20", "21–29", "30–40", "41+"]
METHODS = ["GRNN-Median", "GRNN-Mean", "ARIMA", "Naive2", "MLP"]


def _label(length: int) -> str:
    for (low, high), name in zip(BINS, LABELS):
        if low <= length <= high:
            return name
    raise ValueError(f"length {length} outside all bins")


def _group_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """sMAPE and OWA per group; OWA against the group's own Naive2."""
    rows = []
    for name in [*LABELS, "Усі"]:
        subset = frame if name == "Усі" else frame[frame["bin"] == name]
        naive2 = subset[subset["method"] == "Naive2"]
        naive2_smape = float(naive2["smape"].mean())
        naive2_mase = float(naive2["mase"].mean())
        row: dict[str, object] = {
            "Довжина": name,
            "n": subset["series_id"].nunique(),
        }
        for method in METHODS:
            scores = subset[subset["method"] == method]
            mean_smape = float(scores["smape"].mean())
            mean_mase = float(scores["mase"].mean())
            row[f"{method} sMAPE"] = mean_smape
            row[f"{method} OWA"] = 0.5 * (
                mean_smape / naive2_smape + mean_mase / naive2_mase
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _significance(frame: pd.DataFrame) -> pd.DataFrame:
    """Within-group paired Wilcoxon on per-series MASE, Holm per group."""
    rows = []
    for name in LABELS:
        subset = frame[frame["bin"] == name]
        wide = subset.pivot_table(
            index="series_id", columns="method", values="mase"
        )
        pending = []
        for competitor in ["GRNN-Mean", "ARIMA", "Naive2", "MLP"]:
            paired = wide[["GRNN-Median", competitor]].dropna()
            diffs = paired["GRNN-Median"] - paired[competitor]
            if len(paired) < 10 or np.allclose(diffs, 0):
                continue
            _, p_value = stats.wilcoxon(paired["GRNN-Median"], paired[competitor])
            pending.append({
                "group": name, "competitor": competitor, "n_pairs": len(paired),
                "median_diff_mase": float(diffs.median()), "p_value": p_value,
            })
        for row, p_holm in zip(pending, _holm_adjust([r["p_value"] for r in pending])):
            row["p_holm"] = p_holm
            if p_holm > 0.05:
                row["direction"] = "indistinguishable"
            else:
                row["direction"] = (
                    "target_better" if row["median_diff_mase"] < 0 else "target_worse"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    frame, _ = _build_step_frame(tag)
    frame = frame[frame["diff"] == "on"].copy()
    frame["bin"] = frame["train_length"].map(_label)

    table = _group_rows(frame)
    significance = _significance(frame)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"table_length_groups_{tag}.csv", index=False,
                 float_format="%.6g")
    significance.to_csv(
        FINAL_DIR / f"m4_significance_length_groups_{tag}.csv", index=False,
        float_format="%.6g",
    )

    lines = [
        "# Залежність точності від довжини навчального ряду",
        "",
        f"Тег `{tag}`, диференціювання увімкнене; повторна агрегація збережених "
        "прогнозів, без перенавчання моделей. Межі груп — емпіричні квартилі "
        "розподілу довжин (20, 29, 40); точний поділ на чверті неможливий, бо ці "
        "довжини є масовими точками (20 — 8.1 % вибірки, 29 — 12.8 %, 40 — 12.9 %). "
        "OWA обчислено відносно Naive2 тієї самої групи (та сама конвенція, що для "
        "предметних категорій в основному експерименті), тому Naive2 = 1.000 у "
        "кожному рядку.",
        "",
        "| Довжина | n | " + " | ".join(
            f"{m} sMAPE / OWA" for m in METHODS
        ) + " |",
        "|---|---|" + "---|" * len(METHODS),
    ]
    for _, row in table.iterrows():
        cells = " | ".join(
            f"{row[f'{m} sMAPE']:.2f} / {row[f'{m} OWA']:.3f}" for m in METHODS
        )
        lines.append(f"| {row['Довжина']} | {int(row['n'])} | {cells} |")
    (OUT_DIR / f"table_length_groups_{tag}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(significance.to_string(index=False))
    print(f"\nwrote {OUT_DIR}/table_length_groups_{tag}.{{csv,md}} and "
          f"{FINAL_DIR}/m4_significance_length_groups_{tag}.csv")


if __name__ == "__main__":
    main()
