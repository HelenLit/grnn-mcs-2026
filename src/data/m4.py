"""Loaders for the M4 Competition Yearly subset.

The CSV loader (`load_m4_csv`) is the active path for the domain-wise M4
comparison pipeline. The older `.tsf` loader is kept for provenance of
earlier pilot runs.

File format (Monash Time Series Forecasting Archive convention,
github.com/rakshitha123/TSForecasting):
- header lines start with `#` (comments) or `@` (attributes such as
  `@frequency yearly`, `@horizon 6`, `@equallength false`), ending with `@data`;
- each data line holds one FULL series: `T<n>:<start_timestamp>:v1,v2,...`
  where the LAST `horizon` values are the official M4 test set.

Contract: the official train/test split is reproduced by cutting off the final
`horizon` values — series are NEVER re-split anywhere downstream.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class M4Series:
    """One M4 series with its official train/test split.

    Fields:
        series_id: Official M4 identifier, e.g. "Y1" or legacy "T1".
        train: Training values.
        test: The official test horizon.
        category: M4 domain, e.g. "Micro".
        m: MASE seasonal period (1 for Yearly).
    """
    series_id: str
    train: np.ndarray
    test: np.ndarray
    category: str = ""
    m: int = 1


def _row_values_without_padding(row: pd.Series) -> np.ndarray:
    """Return numeric row values after the ID column, dropping right padding."""
    values = pd.to_numeric(row.iloc[1:], errors="coerce").dropna().to_numpy(
        dtype=np.float64
    )
    return values


def load_m4_csv(
    train_path: str | Path,
    test_path: str | Path,
    info_path: str | Path,
) -> list[M4Series]:
    """Load official M4 Train/Test CSVs and join domain metadata.

    Args:
        train_path: Path to `Train/Yearly-train.csv`.
        test_path: Path to `Test/Yearly-test.csv`.
        info_path: Path to `M4-info.csv`.

    Returns:
        Yearly M4 series in the same order as the train CSV.

    Raises:
        AssertionError: If the official split or metadata is inconsistent.
    """
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    info_frame = pd.read_csv(info_path).set_index("M4id")

    assert len(train_frame) == len(test_frame), "Train/Test row counts differ"
    assert train_frame.iloc[:, 0].is_unique, "Train IDs are not unique"
    assert test_frame.iloc[:, 0].is_unique, "Test IDs are not unique"

    test_by_id = test_frame.set_index(test_frame.columns[0], drop=False)
    series_list: list[M4Series] = []
    for _, train_row in train_frame.iterrows():
        series_id = str(train_row.iloc[0])
        assert series_id in test_by_id.index, f"Missing test row for {series_id}"
        assert series_id in info_frame.index, f"Missing M4-info row for {series_id}"

        info_row = info_frame.loc[series_id]
        train = _row_values_without_padding(train_row)
        test = _row_values_without_padding(test_by_id.loc[series_id])
        horizon = int(info_row["Horizon"])

        assert len(test) == horizon, (
            f"{series_id}: test length {len(test)} != horizon {horizon}"
        )
        assert len(train) > 0, f"{series_id}: empty train series"
        assert not np.isnan(train).any(), f"{series_id}: train contains NaN"
        assert not np.isnan(test).any(), f"{series_id}: test contains NaN"

        series_list.append(
            M4Series(
                series_id=series_id,
                train=train,
                test=test,
                category=str(info_row["category"]),
                m=int(info_row["Frequency"]),
            )
        )

    return series_list


def load_submission_forecasts(
    filepath: str | Path,
    horizon: int,
    id_prefix: str = "Y",
    min_count: int = 23_000,
) -> dict[str, np.ndarray]:
    """Load official M4 submission point forecasts for one frequency prefix.

    Submission files are wide CSVs with an ID column and horizon columns named
    either `F1..` or `X1..` depending on the archive. Only the first `horizon`
    forecast columns are used.

    Args:
        filepath: Path to `submission-<id>.csv`.
        horizon: Number of forecast columns to take (6 for Yearly).
        id_prefix: Series ID prefix to keep (`"Y"` for Yearly).
        min_count: Minimum expected number of filtered rows.

    Returns:
        Mapping from series ID to forecast array.

    Raises:
        AssertionError: If required rows/columns are missing or padded NA appears
            inside the requested horizon.
    """
    frame = pd.read_csv(filepath)
    id_column = frame.columns[0]
    forecast_columns = list(frame.columns[1:1 + horizon])
    assert len(forecast_columns) == horizon, (
        f"{filepath}: fewer than {horizon} forecast columns"
    )

    frame[id_column] = frame[id_column].astype(str)
    yearly = frame[frame[id_column].str.startswith(id_prefix)].copy()
    assert len(yearly) >= min_count, (
        f"{filepath}: expected at least {min_count} {id_prefix} rows, got {len(yearly)}"
    )
    assert yearly[id_column].is_unique, f"{filepath}: duplicate series IDs"

    values = yearly[forecast_columns].apply(pd.to_numeric, errors="coerce")
    assert not values.isna().any().any(), (
        f"{filepath}: NA found inside first {horizon} forecast columns"
    )

    return {
        str(series_id): np.asarray(row, dtype=np.float64)
        for series_id, row in zip(yearly[id_column], values.to_numpy())
    }


def parse_tsf_horizon(filepath: str) -> int:
    """Reads the `@horizon` attribute from a `.tsf` file header.

    Args:
        filepath: Path to the `.tsf` file.

    Returns:
        The forecast horizon declared in the file header.

    Raises:
        ValueError: If no `@horizon` attribute is present before `@data`.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as tsf_file:
        for line in tsf_file:
            line = line.strip()
            if line.lower().startswith("@horizon"):
                return int(line.split()[1])
            if line.lower() == "@data":
                break
    raise ValueError(f"No @horizon attribute found in {filepath}")


def load_m4_yearly(filepath: str, horizon: int) -> list[M4Series]:
    """Parses the M4-Yearly `.tsf` file and applies the official split.

    Each data line contains the full series; the final `horizon` values are
    the official M4 test set (Monash archive convention).

    Args:
        filepath: Path to `m4_yearly_dataset.tsf`.
        horizon: Official forecast horizon (6 for M4-Yearly). Validated
            against the file's `@horizon` attribute.

    Returns:
        List of M4Series in file order (T1, T2, ...).

    Raises:
        ValueError: If `horizon` contradicts the file header, a line is
            malformed, or a series is too short to contain the horizon.
    """
    declared_horizon = parse_tsf_horizon(filepath)
    if declared_horizon != horizon:
        raise ValueError(
            f"Configured horizon ({horizon}) does not match @horizon "
            f"({declared_horizon}) in {filepath}"
        )

    series_list: list[M4Series] = []
    in_data = False
    with open(filepath, "r", encoding="utf-8", errors="replace") as tsf_file:
        for line in tsf_file:
            line = line.strip()
            if not in_data:
                in_data = line.lower() == "@data"
                continue
            if not line or line.startswith("#"):
                continue

            parts = line.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed .tsf data line (expected 3 ':'-separated "
                    f"fields, got {len(parts)}): {line[:80]}"
                )
            series_id = parts[0]
            values = np.array(
                [float(value) for value in parts[2].split(",")],
                dtype=np.float64,
            )
            if len(values) <= horizon:
                raise ValueError(
                    f"Series {series_id} has {len(values)} values — too short "
                    f"to split off a horizon of {horizon}"
                )
            series_list.append(
                M4Series(
                    series_id=series_id,
                    train=values[:-horizon],
                    test=values[-horizon:],
                )
            )

    if not series_list:
        raise ValueError(f"No data lines found in {filepath}")
    return series_list


def sample_series(
    series_list: list[M4Series],
    n_sample: int,
    random_seed: int,
    min_train_length: int = 0,
) -> list[M4Series]:
    """Draws a reproducible random sample of series.

    Sampling is without replacement, seeded, and preserves file order in the
    output so that repeated runs produce byte-identical ID lists.

    Args:
        series_list: Full list of parsed series.
        n_sample: Number of series to draw. If it meets or exceeds the number
            of eligible series, all eligible series are returned.
        random_seed: Seed for `np.random.default_rng`.
        min_train_length: Only series with at least this many training
            observations are eligible (0 = no filter).

    Returns:
        The sampled series, in original file order.
    """
    eligible = [
        (index, series)
        for index, series in enumerate(series_list)
        if len(series.train) >= min_train_length
    ]
    if n_sample >= len(eligible):
        return [series for _, series in eligible]

    rng = np.random.default_rng(random_seed)
    chosen = rng.choice(len(eligible), size=n_sample, replace=False)
    return [eligible[index][1] for index in sorted(chosen)]


def ao_screen(train: np.ndarray, z_threshold: float) -> int:
    """Counts additive-outlier candidates via a z-score screen on differences.

    The first difference of the training series is z-scored; observations
    whose |z| exceeds `z_threshold` are flagged. This is a simple AO screen —
    a screening heuristic, not a formal Chen–Liu detection.

    Args:
        train: Training values of one series.
        z_threshold: Flagging threshold on |z| of the differenced series.

    Returns:
        Number of flagged (potentially anomalous) differences.
    """
    if len(train) < 3:
        return 0
    diffs = np.diff(train)
    std = diffs.std()
    if std == 0.0:
        return 0
    z_scores = (diffs - diffs.mean()) / std
    return int(np.sum(np.abs(z_scores) > z_threshold))
