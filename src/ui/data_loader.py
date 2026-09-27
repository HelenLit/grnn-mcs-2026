"""Data ingestion layer for M4 benchmark dataset and custom CSV uploads.

Handles:
1. M4 local loading, sampling, domain filtering, and repair/download fallback.
2. CSV upload parser with 3-layout auto-detection (tidy, row-wise, column-wise, ambiguous).
3. Parse-time validation (numeric check with cell location, internal NaNs rejection,
   minimum length floor >= 3, constant and zero-value flagging).
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Literal
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import streamlit as st

from src.data.m4 import M4Series, load_m4_csv, sample_series
import src.ui.strings_ua as ua

# M4 dataset repository default base URL
M4_DEFAULT_URL_BASE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"

# Standard paths for M4 files
M4_DIR = Path("data/raw/m4")
M4_TRAIN_PATH = M4_DIR / "Train" / "Yearly-train.csv"
M4_TEST_PATH = M4_DIR / "Test" / "Yearly-test.csv"
M4_INFO_PATH = M4_DIR / "M4-info.csv"

# Example file path
EXAMPLE_CSV_PATH = Path("data/synthetic/sinusoid_noise/sinusoid_noise.csv")


@dataclass
class SeriesData:
    """Standard container for a parsed time series in the UI layer.

    Fields:
        series_id: Identifier string (e.g., "Y1", "sinusoid_noise", "series_1").
        values: 1D numpy array of historical training/observation values.
        test: 1D numpy array of ground-truth test values (for M4 official test,
              or holdout in evaluation mode). None in forecast mode.
        category: Domain or category label (e.g., "Micro", "Finance", "").
        m: Seasonality period for MASE scaling (1 for Yearly/non-seasonal).
        is_constant: True if all differences are zero (MASE denominator = 0).
        has_zeros: True if any values equal 0.0 (sMAPE potential instability).
        length: Number of points in `values`.
    """
    series_id: str
    values: np.ndarray
    test: np.ndarray | None = None
    category: str = ""
    m: int = 1
    is_constant: bool = False
    has_zeros: bool = False
    length: int = 0

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        self.length = len(self.values)
        if self.test is not None:
            self.test = np.asarray(self.test, dtype=np.float64)

        # Check constancy
        if self.length < 2:
            self.is_constant = True
        else:
            self.is_constant = bool(np.all(self.values == self.values[0]))

        # Check zeros
        self.has_zeros = bool(np.any(self.values == 0.0))


class CSVParseError(Exception):
    """Custom exception raised when CSV data violates parse-time validation rules."""
    pass


class AmbiguousLayoutException(Exception):
    """Raised when CSV structure is ambiguous and requires user confirmation."""
    def __init__(self, message: str, sample_shape: tuple[int, int]):
        super().__init__(message)
        self.sample_shape = sample_shape


# ==============================================================================
# M4 Local Loader & Download Helper
# ==============================================================================

def check_m4_local_files() -> tuple[bool, list[str]]:
    """Check whether local M4 Yearly files are present and non-empty.

    Returns:
        tuple (is_available, missing_file_names)
    """
    missing: list[str] = []
    if not (M4_TRAIN_PATH.exists() and M4_TRAIN_PATH.stat().st_size > 0):
        missing.append("Train/Yearly-train.csv")
    if not (M4_TEST_PATH.exists() and M4_TEST_PATH.stat().st_size > 0):
        missing.append("Test/Yearly-test.csv")
    if not (M4_INFO_PATH.exists() and M4_INFO_PATH.stat().st_size > 0):
        missing.append("M4-info.csv")
    return (len(missing) == 0, missing)


def validate_m4_url(url_base: str, timeout: float = 5.0) -> bool:
    """Validate that the provided M4 dataset URL base is reachable.

    Args:
        url_base: Base URL (e.g. GitHub raw URL).
        timeout: Request timeout in seconds.

    Returns:
        True if URL is accessible, False otherwise.
    """
    cleaned_base = url_base.rstrip("/")
    test_target = f"{cleaned_base}/M4-info.csv"
    try:
        req = urllib.request.Request(
            test_target,
            headers={"User-Agent": "GRNN-UI-Validator/1.0"},
            method="HEAD",
)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        # Some servers don't support HEAD; try GET with small range
        try:
            req = urllib.request.Request(
                test_target,
                headers={"User-Agent": "GRNN-UI-Validator/1.0", "Range": "bytes=0-100"},
)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status in (200, 206)
        except Exception:
            return False


def download_m4_files(url_base: str) -> None:
    """Download required M4 Yearly files into data/raw/m4/.

    Args:
        url_base: Base URL where M4 files are hosted.

    Raises:
        RuntimeError: If download fails for any file.
    """
    cleaned_base = url_base.rstrip("/")
    files_to_download = [
        ("Train/Yearly-train.csv", M4_TRAIN_PATH),
        ("Test/Yearly-test.csv", M4_TEST_PATH),
        ("M4-info.csv", M4_INFO_PATH),
    ]

    for rel_url, target_path in files_to_download:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        full_url = f"{cleaned_base}/{rel_url}"
        try:
            req = urllib.request.Request(
                full_url,
                headers={"User-Agent": "GRNN-UI-Downloader/1.0"},
)
            with urllib.request.urlopen(req, timeout=30.0) as response, open(target_path, "wb") as out_file:
                chunk_size = 64 * 1024
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
        except Exception as e:
            if target_path.exists():
                target_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed downloading {rel_url} from {full_url}: {e}") from e


M4_DOMAINS: tuple[str,...] = (
    "All",
    "Demographic",
    "Finance",
    "Industry",
    "Macro",
    "Micro",
    "Other",
)

M4_SCOPES: tuple[int,...] = (100, 1000, 23000)


@st.cache_data(show_spinner="Завантаження вбудованих даних M4 Yearly...")
def load_m4_data_cached() -> list[M4Series]:
    """Load all 23,000 M4 Yearly series using cached load_m4_csv."""
    return load_m4_csv(
        train_path=M4_TRAIN_PATH,
        test_path=M4_TEST_PATH,
        info_path=M4_INFO_PATH,
)


def get_m4_domain_counts(
    series_list: list[M4Series] | None = None,
) -> dict[str, int]:
    """Calculate exact series count per domain in M4 Yearly."""
    if series_list is None:
        series_list = load_m4_data_cached()
    counts = {"All": len(series_list)}
    for s in series_list:
        cat = s.category or "Other"
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def get_m4_sample(
    scope: int = 100,
    domain: str = "All",
    seed: int = 42,
    series_list: list[M4Series] | None = None,
) -> list[SeriesData]:
    """Filter and sample M4 series by scope, domain, and seed.

    Args:
        scope: Target number of series (100, 1000, 23000).
        domain: Domain filter ("All", "Micro", "Finance", etc.).
        seed: Random seed for deterministic sampling.
        series_list: Optional preloaded M4Series list (loads cached if None).

    Returns:
        List of SeriesData ready for modeling/EDA.
    """
    if series_list is None:
        series_list = load_m4_data_cached()

    # Filter by domain if specified
    if domain and domain != "All":
        filtered = [s for s in series_list if s.category == domain]
    else:
        filtered = series_list

    # Sample without replacement
    sampled_m4 = sample_series(filtered, n_sample=scope, random_seed=seed)

    # Convert to SeriesData
    return [
        SeriesData(
            series_id=s.series_id,
            values=s.train,
            test=s.test,
            category=s.category,
            m=s.m,
)
        for s in sampled_m4
    ]


def render_m4_selector_controls(
    domain_counts: dict[str, int] | None = None,
) -> tuple[int, str, int]:
    """Render M4 Scope, Domain filter, Seed input, and 23k runtime warning.

    Returns:
        tuple (scope: int, domain: str, seed: int).
    """
    if domain_counts is None:
        try:
            domain_counts = get_m4_domain_counts()
        except Exception:
            domain_counts = {}

    if "m4_scope" not in st.session_state:
        st.session_state["m4_scope"] = 100
    if "m4_domain" not in st.session_state:
        st.session_state["m4_domain"] = "All"
    if "m4_seed" not in st.session_state:
        st.session_state["m4_seed"] = 42

    # 1. Scope selector (100 / 1 000 / 23 000)
    scope = st.selectbox(
        ua.M4_SCOPE_LABEL,
        options=list(M4_SCOPES),
        format_func=lambda n: {
            100: ua.M4_SCOPE_100,
            1000: ua.M4_SCOPE_1000,
            23000: ua.M4_SCOPE_ALL,
        }.get(n, f"{n} рядів"),
        key="m4_scope",
)

    # 2. Runtime Warning for 23,000 series
    if scope == 23000:
        st.warning(f"⚠️ **{ua.M4_SCOPE_ALL_WARNING}**")

    # 3. Domain filter
    def _format_domain(d: str) -> str:
        count = domain_counts.get(d) if domain_counts else None
        if d == "All":
            label = ua.M4_DOMAIN_ALL
        else:
            label = d
        if count is not None:
            return f"{label} ({count:,} рядів)"
        return label

    domain = st.selectbox(
        ua.M4_DOMAIN_LABEL,
        options=list(M4_DOMAINS),
        format_func=_format_domain,
        key="m4_domain",
)

    # 4. Deterministic Seed
    seed = st.number_input(
        ua.M4_SEED_LABEL,
        min_value=0,
        max_value=2_147_483_647,
        step=1,
        key="m4_seed",
        help="Фіксоване випадкове зерно для детермінованого та відтворюваного семплювання рядів.",
)

    return int(scope), str(domain), int(seed)




# ==============================================================================
# CSV Upload Parsing & 3-Layout Auto-Detection
# ==============================================================================

def detect_csv_layout(
    content: str | bytes,
) -> Literal["tidy", "row_wise", "column_wise", "ambiguous"]:
    """Auto-detect CSV layout among the three accepted forms.

    Layouts:
      (a) "tidy": Single series with header (t,y) or single column y, one point per row.
      (b) "row_wise": Multiple series, one series per row (M4-style, ragged lengths).
      (c) "column_wise": Multiple series, one series per column with header names.
      (d) "ambiguous": Square-ish or uncertain layout requiring user confirmation.

    Returns:
        Layout identifier: "tidy", "row_wise", "column_wise", or "ambiguous".
    """
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="replace")
    else:
        text = str(content)

    # Read the raw lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise CSVParseError("Файл порожній.")

    # Try reading as dataframe with header
    try:
        df_head = pd.read_csv(io.StringIO(text), nrows=5)
    except Exception as e:
        raise CSVParseError(f"Помилка читання CSV: {e}") from e

    cols_lower = [str(c).strip().lower() for c in df_head.columns]

    # Layout (a) detection by explicit headers
    if cols_lower == ["t", "y"] or cols_lower == ["y"]:
        return "tidy"
    if len(df_head.columns) == 2 and ("t" in cols_lower and "y" in cols_lower):
        return "tidy"

    # Read complete raw matrix without header for shape inspection
    try:
        raw_df = pd.read_csv(io.StringIO(text), header=None, dtype=str)
    except Exception as e:
        raise CSVParseError(f"Помилка структури CSV: {e}") from e

    n_rows, n_cols = raw_df.shape

    # Single column -> tidy single series
    if n_cols == 1:
        return "tidy"

    # Single row of data -> row_wise single series
    if n_rows == 1:
        return "row_wise"

    # Two columns: check if first column is sequential indices (0,1,2... or timestamps)
    if n_cols == 2:
        first_col = pd.to_numeric(raw_df.iloc[1:, 0], errors="coerce")
        second_col = pd.to_numeric(raw_df.iloc[1:, 1], errors="coerce")
        if first_col.notna().all() and second_col.notna().all():
            diffs = np.diff(first_col.to_numpy())
            if len(diffs) > 0 and np.all(diffs == diffs[0]) and diffs[0] > 0:
                return "tidy"

    # Check for string IDs in first column vs string headers in first row
    first_col_is_str = False
    first_col_sample = raw_df.iloc[:, 0].dropna()
    numeric_first_col = pd.to_numeric(first_col_sample, errors="coerce")
    if numeric_first_col.isna().sum() > len(first_col_sample) * 0.5:
        first_col_is_str = True

    first_row_is_str = False
    first_row_sample = raw_df.iloc[0,:].dropna()
    numeric_first_row = pd.to_numeric(first_row_sample, errors="coerce")
    if numeric_first_row.isna().sum() > len(first_row_sample) * 0.5:
        first_row_is_str = True

    # Check raggedness pattern (trailing blanks in rows vs trailing blanks in columns)
    row_trailing_nans = 0
    for _, row in raw_df.iterrows():
        non_nas = row.dropna()
        if len(non_nas) < n_cols:
            row_trailing_nans += 1

    col_trailing_nans = 0
    for _, col in raw_df.items():
        non_nas = col.dropna()
        if len(non_nas) < n_rows:
            col_trailing_nans += 1

    # Heuristics based on shape and structure
    if first_col_is_str and not first_row_is_str:
        return "row_wise"
    if first_row_is_str and not first_col_is_str:
        return "column_wise"

    if row_trailing_nans > 0 and col_trailing_nans == 0:
        return "row_wise"
    if col_trailing_nans > 0 and row_trailing_nans == 0:
        return "column_wise"

    # Aspect ratio check
    aspect_ratio = n_rows / n_cols
    if aspect_ratio >= 3.0:
        return "column_wise"
    elif aspect_ratio <= 0.33:
        return "row_wise"

    # If square-ish (0.33 < aspect_ratio < 3.0) and both/neither have string headers, layout is ambiguous
    return "ambiguous"


def _parse_tidy(text: str) -> tuple[list[SeriesData], list[str]]:
    """Parse tidy single-series layout (a)."""
    df = pd.read_csv(io.StringIO(text))
    cols_lower = [str(c).strip().lower() for c in df.columns]

    if "y" in cols_lower:
        y_col_idx = cols_lower.index("y")
        target_series = df.iloc[:, y_col_idx]
        series_name = df.columns[y_col_idx]
    elif len(df.columns) == 1:
        target_series = df.iloc[:, 0]
        series_name = df.columns[0]
    elif len(df.columns) == 2:
        # Default to second column as target y
        target_series = df.iloc[:, 1]
        series_name = df.columns[1]
    else:
        target_series = df.iloc[:, -1]
        series_name = df.columns[-1]

    # Validate numeric values and check for non-numeric cells
    values_list: list[float] = []
    for row_idx, raw_val in enumerate(target_series):
        if pd.isna(raw_val) or str(raw_val).strip() == "":
            raise CSVParseError(
                ua.ERR_INTERNAL_NAN.format(series_id=str(series_name))
)
        try:
            val_float = float(raw_val)
            values_list.append(val_float)
        except (ValueError, TypeError):
            # Row index is 2-indexed in CSV (1 for header + 1 for 1-based row)
            raise CSVParseError(
                ua.ERR_NON_NUMERIC.format(row=row_idx + 2, col=1, val=str(raw_val))
)

    values_arr = np.array(values_list, dtype=np.float64)
    if len(values_arr) < 3:
        raise CSVParseError(
            ua.ERR_SERIES_TOO_SHORT.format(series_id=str(series_name), count=len(values_arr))
)

    series_data = SeriesData(
        series_id=str(series_name),
        values=values_arr,
)
    warnings: list[str] = []
    if series_data.has_zeros:
        warnings.append(ua.WARN_ZERO_VALUES.format(series_id=series_data.series_id))
    if series_data.is_constant:
        warnings.append(ua.ERR_CONSTANT_SERIES.format(series_id=series_data.series_id))

    return [series_data], warnings


def _parse_row_wise(text: str) -> tuple[list[SeriesData], list[str]]:
    """Parse row-wise M4-style layout (b)."""
    raw_df = pd.read_csv(io.StringIO(text), header=None, dtype=str)
    series_list: list[SeriesData] = []
    warnings: list[str] = []

    # Check if first row is a header row (all strings e.g. V1, V2, V3...)
    first_row_non_numeric = pd.to_numeric(raw_df.iloc[0,:].dropna(), errors="coerce").isna().all()
    start_row = 1 if (first_row_non_numeric and len(raw_df) > 1) else 0

    for row_idx in range(start_row, len(raw_df)):
        row = raw_df.iloc[row_idx]

        # Determine series_id from first column or row index
        first_cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        first_cell_numeric = True
        try:
            float(first_cell)
        except ValueError:
            first_cell_numeric = False

        if not first_cell_numeric and first_cell:
            series_id = first_cell
            val_cells = row.iloc[1:].tolist()
            col_offset = 2
        else:
            series_id = f"series_{row_idx + 1}"
            val_cells = row.tolist()
            col_offset = 1

        # Strip trailing empty/NaN cells (ragged padding)
        while val_cells and (pd.isna(val_cells[-1]) or str(val_cells[-1]).strip() == ""):
            val_cells.pop()

        if not val_cells:
            continue

        # Parse and validate values
        values: list[float] = []
        for col_idx, cell in enumerate(val_cells):
            if pd.isna(cell) or str(cell).strip() == "":
                # Internal NaN detected (non-trailing empty cell)
                raise CSVParseError(
                    ua.ERR_INTERNAL_NAN.format(series_id=series_id)
)
            try:
                values.append(float(cell))
            except (ValueError, TypeError):
                raise CSVParseError(
                    ua.ERR_NON_NUMERIC.format(
                        row=row_idx + 1,
                        col=col_idx + col_offset,
                        val=str(cell),
)
)

        values_arr = np.array(values, dtype=np.float64)
        if len(values_arr) < 3:
            raise CSVParseError(
                ua.ERR_SERIES_TOO_SHORT.format(series_id=series_id, count=len(values_arr))
)

        item = SeriesData(series_id=series_id, values=values_arr)
        if item.has_zeros:
            warnings.append(ua.WARN_ZERO_VALUES.format(series_id=item.series_id))
        if item.is_constant:
            warnings.append(ua.ERR_CONSTANT_SERIES.format(series_id=item.series_id))
        series_list.append(item)

    if not series_list:
        raise CSVParseError("Не знайдено валідних часових рядів у файлі.")

    return series_list, warnings


def _parse_column_wise(text: str) -> tuple[list[SeriesData], list[str]]:
    """Parse column-wise layout (c)."""
    # Read with header first
    df = pd.read_csv(io.StringIO(text), dtype=str)
    series_list: list[SeriesData] = []
    warnings: list[str] = []

    for col_idx, col_name in enumerate(df.columns):
        series_id = str(col_name).strip()
        col_cells = df[col_name].tolist()

        # Strip trailing NaNs from bottom
        while col_cells and (pd.isna(col_cells[-1]) or str(col_cells[-1]).strip() == ""):
            col_cells.pop()

        if not col_cells:
            continue

        values: list[float] = []
        for row_idx, cell in enumerate(col_cells):
            if pd.isna(cell) or str(cell).strip() == "":
                # Internal NaN
                raise CSVParseError(
                    ua.ERR_INTERNAL_NAN.format(series_id=series_id)
)
            try:
                values.append(float(cell))
            except (ValueError, TypeError):
                raise CSVParseError(
                    ua.ERR_NON_NUMERIC.format(
                        row=row_idx + 2,  # 1 for header + 1 for 1-based index
                        col=col_idx + 1,
                        val=str(cell),
)
)

        values_arr = np.array(values, dtype=np.float64)
        if len(values_arr) < 3:
            raise CSVParseError(
                ua.ERR_SERIES_TOO_SHORT.format(series_id=series_id, count=len(values_arr))
)

        item = SeriesData(series_id=series_id, values=values_arr)
        if item.has_zeros:
            warnings.append(ua.WARN_ZERO_VALUES.format(series_id=item.series_id))
        if item.is_constant:
            warnings.append(ua.ERR_CONSTANT_SERIES.format(series_id=item.series_id))
        series_list.append(item)

    if not series_list:
        raise CSVParseError("Не знайдено валідних часових рядів у файлі.")

    return series_list, warnings


def parse_csv_content(
    content: str | bytes,
    layout_override: Literal["tidy", "row_wise", "column_wise"] | None = None,
) -> tuple[list[SeriesData], list[str], Literal["tidy", "row_wise", "column_wise"]]:
    """Parse raw CSV content with layout auto-detection and validation.

    Args:
        content: String or bytes containing CSV data.
        layout_override: Explicit layout choice if auto-detection was ambiguous.

    Returns:
        tuple of (parsed_series_list, warning_messages, applied_layout).

    Raises:
        AmbiguousLayoutException: If layout is ambiguous and no override provided.
        CSVParseError: If parse-time validation rules are violated.
    """
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="replace")
    else:
        text = str(content)

    if not text.strip():
        raise CSVParseError("Файл порожній.")

    if layout_override:
        layout = layout_override
    else:
        detected = detect_csv_layout(text)
        if detected == "ambiguous":
            # Inspect sample shape to attach to exception
            raw_df = pd.read_csv(io.StringIO(text), header=None, nrows=10)
            raise AmbiguousLayoutException(
                ua.AMBIGUOUS_LAYOUT_MSG,
                sample_shape=raw_df.shape,
)
        layout = detected

    if layout == "tidy":
        series_list, warnings = _parse_tidy(text)
    elif layout == "row_wise":
        series_list, warnings = _parse_row_wise(text)
    elif layout == "column_wise":
        series_list, warnings = _parse_column_wise(text)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    return series_list, warnings, layout


def get_example_csv_bytes() -> bytes:
    """Retrieve reference fixture sinusoid_noise.csv bytes for example download button."""
    if EXAMPLE_CSV_PATH.exists():
        return EXAMPLE_CSV_PATH.read_bytes()

    # Fallback generation if file was moved
    t = np.arange(100)
    y = np.sin(t * 0.1) + 0.1 * np.random.default_rng(42).normal(size=100)
    df = pd.DataFrame({"t": t, "y": y})
    return df.to_csv(index=False).encode("utf-8")
