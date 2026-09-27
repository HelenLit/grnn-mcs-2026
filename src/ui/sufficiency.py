"""Data-Sufficiency Validation Engine for the GRNN Forecasting UI.

Implements the data-sufficiency rules derived from the pipeline constraints:
    required(L, M, h, mode) = L + M + 1        # Forecast mode, and all M4 runs
    required(L, M, h, mode) = L + M + 1 + h    # Evaluation mode on uploaded data

Guarantees mathematically that any dataset passing this validation will satisfy
all internal assertions of `preparation.py` and model decompositions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import streamlit as st

import src.ui.strings_ua as ua
from src.ui.data_loader import SeriesData


@dataclass(frozen=True)
class SufficiencyResult:
    """Outcome of a dataset sufficiency check against modeling parameters.

    Fields:
        is_sufficient: True if every series in the dataset meets the minimum length.
        required_len: Minimum required observation points N.
        actual_min_len: Length K of the shortest series in the dataset.
        shortest_series_id: Identifier of the shortest series.
        deficit: Point shortfall (required_len - actual_min_len) if insufficient, else 0.
        insufficient_series_ids: List of IDs for all series that are too short.
        insufficient_count: Total count of series falling below required_len.
        message: Formatted live status string.
    """
    is_sufficient: bool
    required_len: int
    actual_min_len: int
    shortest_series_id: str
    deficit: int
    insufficient_series_ids: list[str]
    insufficient_count: int
    message: str


def required_length(
    lags: int,
    models: int,
    horizon: int = 6,
    mode: str = "evaluation",
    is_m4: bool = False,
) -> int:
    """Calculate the exact minimum series length required for a given configuration.

    Derivation:
    - Differencing consumes 1 point: series of length N -> (N - 1) diffs.
    - Sliding window of lag order L consumes L points: (N - 1) - L training pairs.
    - Interleaved decomposition requires at least M pairs (1 per model, M=1 for single):
          (N - 1) - L >= M  =>  N >= L + M + 1.
    - In Evaluation mode on custom upload: holdout consumes h points, so:
          N >= L + M + 1 + h.
    - For M4 runs: official test set is separate, so train length needs L + M + 1.

    Args:
        lags: Lag order L (>= 1).
        models: Number of ensemble models M (>= 1).
        horizon: Forecast horizon h (>= 1).
        mode: Operational mode ("evaluation" or "forecast").
        is_m4: True if active dataset is M4 Yearly benchmark.

    Returns:
        Integer minimum required series length.
    """
    if lags < 1:
        raise ValueError(f"Lags L must be >= 1, got {lags}")
    if models < 1:
        raise ValueError(f"Models M must be >= 1, got {models}")
    if horizon < 1:
        raise ValueError(f"Horizon h must be >= 1, got {horizon}")

    base_required = lags + models + 1

    if is_m4 or mode.lower() == "forecast":
        return base_required
    return base_required + horizon


def check_dataset_sufficiency(
    series_list: Sequence[SeriesData],
    lags: int,
    models: int,
    horizon: int = 6,
    mode: str = "evaluation",
    is_m4: bool = False,
) -> SufficiencyResult:
    """Validate data sufficiency across an entire dataset.

    If ANY series is shorter than required_length, the whole dataset
    is declared insufficient and cannot be run.

    Args:
        series_list: Sequence of SeriesData objects.
        lags: Lag order L.
        models: Number of models M.
        horizon: Forecast horizon h.
        mode: Operational mode ("evaluation" or "forecast").
        is_m4: True if M4 Yearly dataset.

    Returns:
        SufficiencyResult containing detailed diagnostic info.
    """
    req_len = required_length(
        lags=lags,
        models=models,
        horizon=horizon,
        mode=mode,
        is_m4=is_m4,
)

    if not series_list:
        return SufficiencyResult(
            is_sufficient=False,
            required_len=req_len,
            actual_min_len=0,
            shortest_series_id="—",
            deficit=req_len,
            insufficient_series_ids=[],
            insufficient_count=0,
            message=ua.SUFFICIENCY_LIVE_TEMPLATE.format(
                required=req_len,
                actual=0,
                series_id="—",
),
)

    # Find shortest series and any failing series
    min_series = min(series_list, key=lambda s: s.length)
    actual_min_len = min_series.length
    shortest_id = min_series.series_id

    insufficient_ids = [s.series_id for s in series_list if s.length < req_len]
    insufficient_count = len(insufficient_ids)
    is_sufficient = (insufficient_count == 0)
    deficit = max(0, req_len - actual_min_len)

    message = ua.SUFFICIENCY_LIVE_TEMPLATE.format(
        required=req_len,
        actual=actual_min_len,
        series_id=shortest_id,
)

    return SufficiencyResult(
        is_sufficient=is_sufficient,
        required_len=req_len,
        actual_min_len=actual_min_len,
        shortest_series_id=shortest_id,
        deficit=deficit,
        insufficient_series_ids=insufficient_ids,
        insufficient_count=insufficient_count,
        message=message,
)


@st.dialog(ua.DIALOG_INSUFFICIENCY_TITLE)
def show_insufficiency_dialog(
    result: SufficiencyResult,
    lags: int,
    models: int,
    horizon: int,
    mode: str,
    is_m4: bool,
) -> None:
    """Render the modal explanation dialog when data is insufficient."""
    resolution_hint = ua.RESOLUTION_M4_HINT if is_m4 else ua.RESOLUTION_UPLOAD_HINT
    mode_name = ua.MODE_EVALUATION if mode == "evaluation" else ua.MODE_FORECAST

    body = ua.DIALOG_INSUFFICIENCY_BODY.format(
        L=lags,
        M=models,
        h=horizon,
        mode=mode_name,
        required=result.required_len,
        series_id=result.shortest_series_id,
        actual=result.actual_min_len,
        deficit=result.deficit,
        resolution_hint=resolution_hint,
)

    if result.insufficient_count > 1:
        body += f"\n\n*(Загалом недостатню довжину мають {result.insufficient_count} рядів у поточному наборі даних)*."

    st.markdown(body)

    if st.button(ua.DIALOG_OK_BTN, type="primary", use_container_width=True):
        st.rerun()
