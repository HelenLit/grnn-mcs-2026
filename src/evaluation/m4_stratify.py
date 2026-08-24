"""Generic stratified re-aggregation of M4 Yearly scores.

One grouping engine for every stratification the analysis needs — domain,
horizon step, train length and anomaly level all do the same thing: group the
per-series scores and re-aggregate sMAPE / MASE / OWA inside each group.

Two facts make this cheap:

  1. Per-step forecasts are already frozen on disk. `grnn_results_<tag>.csv`
     stores `F1..F6` per (series, model, diff), so the horizon decomposition is
     post-processing — the GRNN stage is never re-run for it.
  2. The official evaluation code itself keeps the horizon dimension
     (`Total_smape[method, step, series]`) and collapses it only in the final
     `mean()`. A per-step table is the official structure with one fewer
     averaging step, not an invention.

**Two OWA columns, always.** OWA has no per-series definition, so a stratified
OWA needs a choice of denominator and the choice changes what the number means:

  * `owa_within` — Naive2's means taken inside the same group. Answers "how
    well does the method do relative to the difficulty of this group". Naive2
    is 1.0 in every group by construction.
  * `owa_global` — Naive2's means over the whole sample, i.e. the published
    denominator. Stays comparable with the M4 leaderboard, and because the
    denominator is then constant, the mean of `owa_global` over the groups
    reproduces the aggregate OWA exactly. That identity is the correctness
    gate (`verify_group_identity`).

Reporting only one invites the obvious reviewer question, so both are emitted
side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

HORIZON = 6
STEP_COLUMNS = tuple(f"F{index}" for index in range(1, HORIZON + 1))
SMAPE_STEP_COLUMNS = tuple(f"smape_h{index}" for index in range(1, HORIZON + 1))
MASE_STEP_COLUMNS = tuple(f"mase_h{index}" for index in range(1, HORIZON + 1))

# Pre-registered, NOT data-driven. Fixed edges chosen before looking at any
# result so the boundaries cannot be accused of post-hoc selection
# (reproducibility contract). M4 Yearly train length runs from 13 up, with
# a median of 29, so these split the mass rather than isolating a tail.
LENGTH_BIN_EDGES = (13, 20, 26, 32, 41)
LENGTH_BIN_LABELS = ("13-19", "20-25", "26-31", "32-40", "41+")

# Anomaly level from the additive-outlier screen (`ao_screen`, |z| > 3.0 on the
# differenced train series). Bins are counts, not rates, because the screen
# returns a count and most series have zero.
ANOMALY_BIN_LABELS = ("0", "1", "2", "3+")

MIN_GROUP_SIZE = 200


@dataclass(frozen=True)
class Stratification:
    """One grouping dimension.

    Fields:
        name: Short slug used in output filenames.
        column: Column of the per-series frame to group by.
        order: Group labels in report order.
        title: Human-readable axis name for tables and plots.
    """

    name: str
    column: str
    order: tuple[str, ...]
    title: str


def series_masep(train: np.ndarray, m: int) -> float:
    """MASE scaling denominator of one series, per the official R code.

    Mirrors `m4_metrics.mase`'s denominator exactly so per-step MASE and the
    already-reported per-series MASE cannot drift apart.
    """
    train = np.asarray(train, dtype=np.float64)
    if len(train) <= m:
        raise ValueError(f"Train length {len(train)} too short for m={m}")
    return float(np.mean(np.abs(train[m:] - train[:-m])))


def per_step_scores(
    forecasts: np.ndarray,
    actual: np.ndarray,
    masep: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-horizon-step sMAPE and MASE, before any averaging.

    Elementwise forms of the official definitions; their means over the six
    steps are the per-series sMAPE and MASE already reported.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    absolute_error = np.abs(actual - forecasts)
    smape_steps = absolute_error * 200.0 / (np.abs(actual) + np.abs(forecasts))
    mase_steps = absolute_error / masep
    return smape_steps, mase_steps


def length_bin(train_length: int) -> str:
    """Maps a train length onto a pre-registered length bin label."""
    for index in range(len(LENGTH_BIN_EDGES) - 1, 0, -1):
        if train_length >= LENGTH_BIN_EDGES[index]:
            return LENGTH_BIN_LABELS[index]
    return LENGTH_BIN_LABELS[0]


def anomaly_bin(ao_count: int) -> str:
    """Maps an additive-outlier count onto an anomaly bin label."""
    return ANOMALY_BIN_LABELS[min(int(ao_count), 3)]


def merge_small_groups(
    frame: pd.DataFrame,
    column: str,
    order: tuple[str, ...],
    min_size: int = MIN_GROUP_SIZE,
) -> tuple[pd.DataFrame, tuple[str, ...], list[str]]:
    """Folds under-populated groups into their neighbour, and says so.

    A group with fewer than `min_size` series gives an unstable mean; dropping
    it silently would understate coverage, so it is merged into the adjacent
    group and the merge is returned as a log line for the manifest.

    Returns:
        (frame with the column rewritten, new group order, merge log lines)
    """
    counts = frame.drop_duplicates("series_id")[column].value_counts()
    remaining = [label for label in order if counts.get(label, 0) > 0]
    log: list[str] = []
    mapping: dict[str, str] = {label: label for label in remaining}

    changed = True
    while changed and len(remaining) > 1:
        changed = False
        for position, label in enumerate(remaining):
            size = sum(
                counts.get(source, 0)
                for source, target in mapping.items()
                if target == label
            )
            if size >= min_size:
                continue
            neighbour = remaining[position - 1] if position > 0 else remaining[1]
            merged_label = (
                f"{neighbour}+{label}" if position > 0 else f"{label}+{neighbour}"
            )
            for source, target in list(mapping.items()):
                if target in (label, neighbour):
                    mapping[source] = merged_label
            remaining = [
                merged_label if item in (label, neighbour) else item
                for item in remaining
            ]
            remaining = list(dict.fromkeys(remaining))
            log.append(
                f"{column}: group '{label}' had n={size} < {min_size}; "
                f"merged with '{neighbour}' into '{merged_label}'"
            )
            changed = True
            break

    if log:
        frame = frame.assign(**{column: frame[column].map(mapping)})
    return frame, tuple(remaining), log


def _owa(
    mean_smape: float,
    mean_mase: float,
    naive2_smape: float,
    naive2_mase: float,
) -> float:
    return 0.5 * (mean_smape / naive2_smape + mean_mase / naive2_mase)


def aggregate_by_group(
    frame: pd.DataFrame,
    group_column: str,
    group_order: tuple[str, ...],
    methods: tuple[str, ...],
    smape_column: str = "smape",
    mase_column: str = "mase",
    naive2_label: str = "Naive2",
    include_all: bool = True,
) -> pd.DataFrame:
    """Re-aggregates sMAPE / MASE / OWA inside each group.

    `frame` is long: one row per (series_id, method, group), carrying the two
    metric columns. Both OWA conventions are emitted — see the module
    docstring for why.

    Returns:
        Long frame with columns group, method, n, smape, mase, owa_within,
        owa_global.
    """
    naive2_rows = frame[frame["method"] == naive2_label]
    if naive2_rows.empty:
        raise ValueError(f"No {naive2_label} rows — OWA has no denominator.")
    global_naive2_smape = float(naive2_rows[smape_column].mean())
    global_naive2_mase = float(naive2_rows[mase_column].mean())

    groups = [*group_order, "All"] if include_all else list(group_order)
    rows = []
    for group in groups:
        subset = frame if group == "All" else frame[frame[group_column] == group]
        if subset.empty:
            continue
        group_naive2 = subset[subset["method"] == naive2_label]
        naive2_smape = float(group_naive2[smape_column].mean())
        naive2_mase = float(group_naive2[mase_column].mean())

        for method in methods:
            method_rows = subset[subset["method"] == method]
            if method_rows.empty:
                continue
            mean_smape = float(method_rows[smape_column].mean())
            mean_mase = float(method_rows[mase_column].mean())
            rows.append({
                "group": group,
                "method": method,
                "n": int(len(method_rows)),
                "smape": mean_smape,
                "mase": mean_mase,
                "owa_within": _owa(
                    mean_smape, mean_mase, naive2_smape, naive2_mase
                ),
                "owa_global": _owa(
                    mean_smape, mean_mase, global_naive2_smape, global_naive2_mase
                ),
            })
    return pd.DataFrame(rows)


def verify_group_identity(
    long_table: pd.DataFrame,
    group_order: tuple[str, ...],
    tolerance: float = 1e-9,
) -> list[str]:
    """Checks that the groups reconstruct the aggregate — the free gate.

    With a constant (global) Naive2 denominator, OWA is linear in the two
    metric means, so the n-weighted mean of a method's per-group values must
    reproduce its "All" value. Any deviation means the groups do not partition
    the sample, or a method is missing from some group. Verified for sMAPE,
    MASE and `owa_global`; `owa_within` is deliberately excluded because its
    denominator varies by group and the identity does not hold for it.

    Returns:
        List of failure descriptions; empty when the identity holds.
    """
    failures = []
    grouped = long_table[long_table["group"] != "All"]
    aggregate = long_table[long_table["group"] == "All"].set_index("method")

    for metric in ("smape", "mase", "owa_global"):
        for method, rows in grouped.groupby("method", sort=False):
            if method not in aggregate.index:
                continue
            weights = rows["n"].to_numpy(dtype=np.float64)
            if weights.sum() == 0:
                continue
            reconstructed = float(
                np.average(rows[metric].to_numpy(dtype=np.float64), weights=weights)
            )
            expected = float(aggregate.loc[method, metric])
            deviation = abs(reconstructed - expected)
            if deviation > tolerance:
                failures.append(
                    f"{metric}/{method}: groups reconstruct {reconstructed:.12g}, "
                    f"aggregate is {expected:.12g}, deviation {deviation:.3g}"
                )
    return failures


def to_wide(
    long_table: pd.DataFrame,
    metric: str,
    group_order: tuple[str, ...],
    methods: tuple[str, ...],
    group_title: str,
) -> pd.DataFrame:
    """Pivots the long aggregate into a report table: groups as rows."""
    ordered_groups = [
        group for group in [*group_order, "All"]
        if group in set(long_table["group"])
    ]
    rows = []
    for group in ordered_groups:
        subset = long_table[long_table["group"] == group].set_index("method")
        row = {
            group_title: group,
            "n": int(subset["n"].max()) if not subset.empty else 0,
        }
        for method in methods:
            row[method] = (
                float(subset.loc[method, metric])
                if method in subset.index else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=[group_title, "n", *methods])


def write_markdown(
    table: pd.DataFrame,
    path,
    title: str,
    note: str = "",
    float_format: Callable[[float], str] = lambda value: f"{value:.4f}",
) -> None:
    """Writes a report table as GitHub-flavoured Markdown."""
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    header = list(table.columns)
    lines.append("| " + " | ".join(str(column) for column in header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for _, row in table.iterrows():
        cells = []
        for column in header:
            value = row[column]
            if isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append("—" if np.isnan(value) else float_format(float(value)))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
