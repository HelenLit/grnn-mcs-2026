"""Parameter Panel Management and Optimizer Registry Binding.

Implements parameter controls, defaults, validation bounds, and registry bindings
per the frozen baseline configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
import streamlit as st

from src.models.sigma_optimizers import ALL_METHODS
import src.ui.strings_ua as ua


@dataclass
class UIParameters:
    """Hyperparameters and configuration settings for the GRNN forecasting run.

    Fields:
        lags: Lag order L (1..12, default 3). Maps to `lag_order`.
        models: Number of models M (1..20, default 6). M=1 is single GRNN.
        aggregation: Aggregation method ("median" or "mean", default "median").
        optimizer: Sigma optimizer key ("brent", "gradient", "coarse2fine", "silverman", "grid").
        horizon: Forecast horizon h (>= 1, default 6; locked to 6 for M4).
        mode: Operational mode ("evaluation" or "forecast", default "evaluation").
        difference: Differencing flag (always True).
    """
    lags: int = 3
    models: int = 6
    aggregation: Literal["median", "mean"] = "median"
    optimizer: str = "brent"
    horizon: int = 6
    mode: Literal["evaluation", "forecast"] = "evaluation"
    difference: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration into a dictionary for run manifests."""
        return asdict(self)


# Frozen baseline default parameters
DEFAULT_PARAMETERS = UIParameters(
    lags=3,
    models=6,
    aggregation="median",
    optimizer="brent",
    horizon=6,
    mode="evaluation",
    difference=True,
)


def validate_parameters(params: UIParameters, is_m4: bool = False) -> list[str]:
    """Validate parameter ranges and compatibility.

    Args:
        params: UIParameters instance.
        is_m4: Whether active dataset is M4 Yearly.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    if not (1 <= params.lags <= 12):
        errors.append(f"Кількість лагів L повинна бути в межах 1…12, отримано: {params.lags}")
    if not (1 <= params.models <= 20):
        errors.append(f"Кількість моделей M повинна бути в межах 1…20, отримано: {params.models}")
    if params.aggregation not in ("median", "mean"):
        errors.append(f"Невідомий метод агрегування: '{params.aggregation}' (дозволені: 'median', 'mean')")
    if params.optimizer not in ALL_METHODS:
        errors.append(f"Невідомий оптимізатор: '{params.optimizer}' (доступні: {list(ALL_METHODS.keys())})")
    if params.horizon < 1:
        errors.append(f"Горизонт прогнозування h повинен бути >= 1, отримано: {params.horizon}")
    if is_m4 and params.horizon != 6:
        errors.append(f"Для M4 Yearly горизонт прогнозування зафіксовано h=6, отримано: {params.horizon}")
    if is_m4 and params.mode != "evaluation":
        errors.append(f"Для M4 Yearly режим роботи зафіксовано як 'evaluation', отримано: '{params.mode}'")
    if not params.difference:
        errors.append("Диференціювання (first differencing) повинно бути завжди увімкнено.")

    return errors


def reset_parameters_to_defaults() -> None:
    """Reset Streamlit session state parameter keys to baseline defaults."""
    st.session_state["lags"] = DEFAULT_PARAMETERS.lags
    st.session_state["models"] = DEFAULT_PARAMETERS.models
    st.session_state["aggregation"] = DEFAULT_PARAMETERS.aggregation
    st.session_state["optimizer"] = DEFAULT_PARAMETERS.optimizer
    st.session_state["horizon"] = DEFAULT_PARAMETERS.horizon
    st.session_state["mode"] = DEFAULT_PARAMETERS.mode


def render_parameter_controls(is_m4: bool = False) -> UIParameters:
    """Render parameter inputs, tooltips, selectboxes, and reset button.

    Args:
        is_m4: True if active dataset is M4 Yearly benchmark.

    Returns:
        Active UIParameters instance constructed from user inputs.
    """
    st.subheader(ua.PARAMS_HEADER)

    if "lags" not in st.session_state:
        st.session_state["lags"] = DEFAULT_PARAMETERS.lags
    if "models" not in st.session_state:
        st.session_state["models"] = DEFAULT_PARAMETERS.models
    if "aggregation" not in st.session_state:
        st.session_state["aggregation"] = DEFAULT_PARAMETERS.aggregation
    if "optimizer" not in st.session_state:
        st.session_state["optimizer"] = DEFAULT_PARAMETERS.optimizer
    if "horizon" not in st.session_state:
        st.session_state["horizon"] = DEFAULT_PARAMETERS.horizon

    if is_m4:
        st.session_state["horizon"] = 6

    # Lags (L)
    lags = st.number_input(
        ua.PARAM_LAGS_LABEL,
        min_value=1,
        max_value=12,
        key="lags",
        help=ua.PARAM_LAGS_HELP,
)

    # Models (M)
    models = st.number_input(
        ua.PARAM_MODELS_LABEL,
        min_value=1,
        max_value=20,
        key="models",
        help=ua.PARAM_MODELS_HELP,
)

    # Aggregation
    aggregation = st.radio(
        ua.PARAM_AGG_LABEL,
        options=["median", "mean"],
        format_func=lambda a: ua.PARAM_AGG_MEDIAN if a == "median" else ua.PARAM_AGG_MEAN,
        key="aggregation",
        help=ua.PARAM_AGG_HELP,
)

    # Sigma Optimizer
    optimizer_keys = list(ALL_METHODS.keys())
    optimizer = st.selectbox(
        ua.PARAM_OPTIMIZER_LABEL,
        options=optimizer_keys,
        format_func=lambda k: ua.OPTIMIZER_LABELS.get(k, k),
        key="optimizer",
        help=ua.OPTIMIZER_TOOLTIPS.get(st.session_state.get("optimizer", "brent"), ua.PARAM_OPTIMIZER_HELP),
)

    # Tooltip / note for silverman
    if optimizer == "silverman":
        st.caption(f"ℹ️ {ua.OPTIMIZER_TOOLTIPS['silverman']}")

    # Horizon (h)
    horizon = st.number_input(
        ua.PARAM_HORIZON_LABEL,
        min_value=1,
        max_value=100,
        disabled=is_m4,
        key="horizon",
        help=ua.PARAM_HORIZON_M4_LOCKED_NOTE if is_m4 else ua.PARAM_HORIZON_HELP,
)


    # Reset to defaults button
    st.button(
        ua.RESET_DEFAULTS_BUTTON,
        on_click=reset_parameters_to_defaults,
        help=ua.RESET_DEFAULTS_HELP,
        use_container_width=True,
)

    # Differencing note
    st.caption(f"💡 {ua.DIFFERENCING_NOTE}")

    mode = "evaluation" if is_m4 else st.session_state.get("mode", "evaluation")

    return UIParameters(
        lags=int(lags),
        models=int(models),
        aggregation=aggregation,
        optimizer=optimizer,
        horizon=int(horizon),
        mode=mode,
        difference=True,
)
