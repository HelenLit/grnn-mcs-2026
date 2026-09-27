"""GRNN Time Series Forecasting Streamlit Application.

Entry point for the local interactive demo of GRNN ensemble forecasting models
on M4 Yearly benchmark data and custom uploaded time series.

Launch with:
    streamlit run src/ui/app.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src.*' imports always resolve
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

import src.ui.strings_ua as ua

import src.ui.theme as theme
from src.models.sigma_optimizers import ALL_METHODS
from src.ui.data_loader import (
    M4_DEFAULT_URL_BASE,
    SeriesData,
    CSVParseError,
    AmbiguousLayoutException,
    check_m4_local_files,
    validate_m4_url,
    download_m4_files,
    get_m4_sample,
    get_m4_domain_counts,
    render_m4_selector_controls,
    parse_csv_content,
    get_example_csv_bytes,
)

from src.ui.sufficiency import (
    required_length,
    check_dataset_sufficiency,
    show_insufficiency_dialog,
)
from src.ui.parameters import (
    UIParameters,
    render_parameter_controls,
    reset_parameters_to_defaults,
)
from src.ui.run_engine import (
    RunResults,
    SeriesForecastResult,
    execute_run,
)
from src.ui.eda import (
    compute_dataset_eda,
    render_eda_tab,
)
from src.ui.results_view import (
    render_results_tab,
    render_download_tab,
)






# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title=ua.APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply unified design system styling
theme.inject_custom_css()


def _init_session_state() -> None:
    """Initialize Streamlit session state defaults if not already present."""
    if "mode" not in st.session_state:
        st.session_state["mode"] = "evaluation"
    if "lags" not in st.session_state:
        st.session_state["lags"] = 3
    if "models" not in st.session_state:
        st.session_state["models"] = 6
    if "aggregation" not in st.session_state:
        st.session_state["aggregation"] = "median"
    if "optimizer" not in st.session_state:
        st.session_state["optimizer"] = "brent"
    if "horizon" not in st.session_state:
        st.session_state["horizon"] = 6
    if "active_dataset" not in st.session_state:
        st.session_state["active_dataset"] = "m4"
    if "m4_scope" not in st.session_state:
        st.session_state["m4_scope"] = 100
    if "m4_domain" not in st.session_state:
        st.session_state["m4_domain"] = "All"
    if "m4_seed" not in st.session_state:
        st.session_state["m4_seed"] = 42
    if "m4_download_url" not in st.session_state:
        st.session_state["m4_download_url"] = M4_DEFAULT_URL_BASE
    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None
    if "layout_override" not in st.session_state:
        st.session_state["layout_override"] = None
    if "active_series_list" not in st.session_state:
        st.session_state["active_series_list"] = []
    if "data_warnings" not in st.session_state:
        st.session_state["data_warnings"] = []
    if "data_error" not in st.session_state:
        st.session_state["data_error"] = None
    if "detected_layout" not in st.session_state:
        st.session_state["detected_layout"] = None
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "eda_data" not in st.session_state:
        st.session_state["eda_data"] = None


def reset_to_defaults() -> None:
    """Reset modeling hyperparameters to frozen baseline defaults."""
    st.session_state["lags"] = 3
    st.session_state["models"] = 6
    st.session_state["aggregation"] = "median"
    st.session_state["optimizer"] = "brent"
    st.session_state["horizon"] = 6


def render_header() -> None:
    """Render the application header with thesis context badges."""
    st.markdown(
        f'<span class="badge-pill">{ua.BADGE_THESIS}</span> '
        f'<span class="badge-pill" style="background: linear-gradient(135deg, {theme.EMERALD_GREEN}, {theme.DODGER_BLUE});">{ua.BADGE_BENCHMARK}</span>',
        unsafe_allow_html=True,
)
    st.title(ua.APP_HEADER)
    st.markdown(
        f'<p class="muted-text" style="font-size: 1.05rem; margin-top: -0.5rem; margin-bottom: 1.5rem;">{ua.APP_SUBTITLE}</p>',
        unsafe_allow_html=True,
)


def _load_active_data() -> tuple[list[SeriesData], list[str], str | None]:
    """Load and return active series based on session state.

    Returns:
        tuple (series_list, warnings, error_message)
    """
    data_source = st.session_state.get("active_dataset", "m4")
    warnings: list[str] = []

    if data_source == "m4":
        m4_available, missing = check_m4_local_files()
        if not m4_available:
            return [], [], f"Відсутні локальні файли M4: {', '.join(missing)}"
        try:
            series = get_m4_sample(
                scope=st.session_state.get("m4_scope", 100),
                domain=st.session_state.get("m4_domain", "All"),
                seed=int(st.session_state.get("m4_seed", 42)),
)
            # Check for any zero or constant warnings in M4 sample
            for s in series:
                if s.is_constant:
                    warnings.append(ua.ERR_CONSTANT_SERIES.format(series_id=s.series_id))
                if s.has_zeros:
                    warnings.append(ua.WARN_ZERO_VALUES.format(series_id=s.series_id))
            return series, warnings, None
        except Exception as e:
            return [], [], f"Помилка завантаження даних M4: {e}"

    elif data_source == "upload":
        uploaded_file = st.session_state.get("csv_uploader_file")
        if uploaded_file is None:
            return [], [], "Оберіть CSV файл для завантаження."

        try:
            content = uploaded_file.getvalue()
            layout_override = st.session_state.get("layout_override")
            series, parse_warnings, applied_layout = parse_csv_content(
                content,
                layout_override=layout_override,
)
            st.session_state["detected_layout"] = applied_layout
            return series, parse_warnings, None
        except AmbiguousLayoutException as amb_err:
            st.session_state["is_ambiguous_layout"] = True
            return [], [], str(amb_err)
        except CSVParseError as parse_err:
            return [], [], str(parse_err)
        except Exception as e:
            return [], [], f"Непередбачена помилка парсингу: {e}"

    return [], [], "Невідоме джерело даних"


def render_m4_download_panel() -> None:
    """Render the repair panel for downloading missing M4 data files."""
    st.warning(ua.M4_MISSING_TITLE, icon="⚠️")
    st.write(ua.M4_MISSING_MSG)

    url_input = st.text_input(
        ua.M4_URL_LABEL,
        value=st.session_state["m4_download_url"],
        key="input_m4_url",
)

    col1, col2 = st.columns([2, 3])
    with col1:
        if st.button(ua.M4_DOWNLOAD_BTN, type="primary", use_container_width=True):
            with st.spinner("Перевірка URL та завантаження даних..."):
                if validate_m4_url(url_input):
                    try:
                        download_m4_files(url_input)
                        st.success(ua.M4_DOWNLOAD_SUCCESS)
                        st.rerun()
                    except Exception as e:
                        st.error(f"{ua.M4_DOWNLOAD_ERROR} ({e})")
                else:
                    st.error(ua.M4_DOWNLOAD_ERROR)


def render_sidebar() -> None:
    """Render sidebar controls: mode, data source, parameters, sufficiency, actions."""
    with st.sidebar:
        st.subheader(ua.MODE_LABEL)
        is_m4 = (st.session_state.get("active_dataset", "m4") == "m4")

        if is_m4:
            # Locked to Evaluation for M4
            st.session_state["mode"] = "evaluation"
            st.radio(
                ua.MODE_LABEL,
                options=["evaluation"],
                format_func=lambda _: ua.MODE_EVALUATION,
                index=0,
                disabled=True,
                label_visibility="collapsed",
)
            st.caption(ua.MODE_M4_LOCKED_NOTE)
        else:
            mode_idx = 0 if st.session_state["mode"] == "evaluation" else 1
            selected_mode = st.radio(
                ua.MODE_LABEL,
                options=["evaluation", "forecast"],
                format_func=lambda m: ua.MODE_EVALUATION if m == "evaluation" else ua.MODE_FORECAST,
                index=mode_idx,
                label_visibility="collapsed",
                help=ua.MODE_EVALUATION_HELP,
)
            st.session_state["mode"] = selected_mode
            st.caption(
                ua.MODE_EVALUATION_HELP
                if selected_mode == "evaluation"
                else ua.MODE_FORECAST_HELP
)

        st.divider()

        # Data source selector
        st.subheader(ua.DATA_SOURCE_LABEL)
        data_source = st.radio(
            ua.DATA_SOURCE_LABEL,
            options=["m4", "upload"],
            format_func=lambda s: ua.DATA_SOURCE_M4 if s == "m4" else ua.DATA_SOURCE_UPLOAD,
            label_visibility="collapsed",
)
        st.session_state["active_dataset"] = data_source

        if data_source == "m4":
            m4_ok, missing = check_m4_local_files()
            if not m4_ok:
                st.error(f"Файли M4 не знайдено: {', '.join(missing)}")
            else:
                render_m4_selector_controls()


        else:
            uploaded_file = st.file_uploader(
                ua.UPLOAD_LABEL,
                type=["csv"],
                help=ua.UPLOAD_FORMATS_HINT,
                key="csv_uploader_file",
)
            if uploaded_file is not None and uploaded_file.name != st.session_state.get("uploaded_file_name"):
                st.session_state["uploaded_file_name"] = uploaded_file.name
                st.session_state["layout_override"] = None

            # Ambiguous layout resolution if raised
            if st.session_state.get("is_ambiguous_layout") and uploaded_file is not None:
                st.info(ua.AMBIGUOUS_LAYOUT_TITLE)
                layout_choice = st.radio(
                    ua.AMBIGUOUS_LAYOUT_MSG,
                    options=["row_wise", "column_wise"],
                    format_func=lambda opt: ua.AMBIGUOUS_CHOICE_ROWS if opt == "row_wise" else ua.AMBIGUOUS_CHOICE_COLS,
                    key="radio_layout_choice",
)
                st.session_state["layout_override"] = layout_choice
                st.session_state["is_ambiguous_layout"] = False

            # Example file download
            st.download_button(
                label=ua.DOWNLOAD_EXAMPLE_BTN,
                data=get_example_csv_bytes(),
                file_name="sinusoid_noise_example.csv",
                mime="text/csv",
                use_container_width=True,
)

        st.divider()

        # Update active series list
        series_list, data_warnings, data_error = _load_active_data()
        st.session_state["active_series_list"] = series_list
        st.session_state["data_warnings"] = data_warnings
        st.session_state["data_error"] = data_error

        # Modeling parameters
        ui_params = render_parameter_controls(is_m4=is_m4)
        st.session_state["ui_parameters"] = ui_params

        st.divider()

        # Live Sufficiency Status
        sufficiency_result = check_dataset_sufficiency(
            series_list=series_list,
            lags=ui_params.lags,
            models=ui_params.models,
            horizon=ui_params.horizon,
            mode=ui_params.mode,
            is_m4=is_m4,
)
        st.session_state["sufficiency_result"] = sufficiency_result

        is_sufficient = sufficiency_result.is_sufficient and (data_error is None) and (len(series_list) > 0)
        banner_class = "ok" if is_sufficient else "insufficient"
        icon = "✅" if is_sufficient else "⚠️"

        st.markdown(
            f'<div class="sufficiency-banner {banner_class}">'
            f'<span>{icon} {sufficiency_result.message}</span>'
            f'</div>',
            unsafe_allow_html=True,
)

        if not sufficiency_result.is_sufficient and len(series_list) > 0:
            if st.button("🔍 Пояснення дефіциту довжини", use_container_width=True):
                show_insufficiency_dialog(
                    result=sufficiency_result,
                    lags=ui_params.lags,
                    models=ui_params.models,
                    horizon=ui_params.horizon,
                    mode=ui_params.mode,
                    is_m4=is_m4,
)

        # Action Buttons
        has_data = len(series_list) > 0 and (data_error is None)
        col_run, col_eda = st.columns([3, 2])
        with col_run:
            if st.button(
                ua.BTN_RUN,
                type="primary",
                disabled=not is_sufficient,
                use_container_width=True,
                key="btn_run",
):
                st.session_state["execute_run_requested"] = True
                st.rerun()
        with col_eda:
            if st.button(
                ua.BTN_EDA,
                disabled=not has_data,
                use_container_width=True,
                key="btn_eda",
):
                st.session_state["execute_eda_requested"] = True



def render_main_content() -> None:
    """Render the primary content area with tabbed views for Data, EDA, Results, and Downloads."""
    series_list = st.session_state.get("active_series_list", [])
    data_error = st.session_state.get("data_error")
    data_warnings = st.session_state.get("data_warnings", [])
    data_source = st.session_state.get("active_dataset", "m4")
    ui_params: UIParameters | None = st.session_state.get("ui_parameters")
    is_m4 = (data_source == "m4")

    # Handle Run Execution Trigger
    if st.session_state.get("execute_run_requested"):
        st.session_state["execute_run_requested"] = False
        if series_list and ui_params:
            run_container = st.container()
            with run_container:
                st.markdown('<div class="ui-card">', unsafe_allow_html=True)
                st.subheader(f"⚙️ {ua.PARAMS_HEADER}: Виконання прогнозування")
                prog_bar = st.progress(0.0)
                status_box = st.empty()

                def _progress_cb(current: int, total: int, series_id: str) -> None:
                    pct = current / total
                    prog_bar.progress(pct)
                    status_box.markdown(
                        f"**{ua.STATUS_RUNNING_TEMPLATE.format(i=current, n=total, percent=pct * 100)}** — ряд `{series_id}`"
)

                try:
                    new_results = execute_run(
                        series_list=series_list,
                        params=ui_params,
                        is_m4=is_m4,
                        progress_callback=_progress_cb,
                        dataset_info={
                            "source": data_source,
                            "count": len(series_list),
                            "domain": st.session_state.get("m4_domain", "All"),
                            "seed": st.session_state.get("m4_seed", 42),
                            "uploaded_filename": st.session_state.get("uploaded_file_name"),
                        },
)
                    # Atomic replacement
                    st.session_state["results"] = new_results
                    prog_bar.progress(1.0)
                    status_box.empty()
                    st.success(
                        f"🎉 {ua.STATUS_SUCCESS} Оброблено {len(new_results.series_results)} рядів за {new_results.total_runtime_s:.2f} с.",
                        icon="✅",
)
                except InterruptedError as ie:
                    prog_bar.empty()
                    status_box.empty()
                    st.warning(f"⚠️ {ie}")
                except Exception as e:
                    prog_bar.empty()
                    status_box.empty()
                    st.error(ua.STATUS_ERROR_TEMPLATE.format(error=str(e)), icon="🚨")

                st.markdown('</div>', unsafe_allow_html=True)

    tab_overview, tab_eda, tab_results, tab_download = st.tabs([
        "📊 Огляд даних",
        "🔍 EDA Аналіз",
        "📈 Результати та Графіки",
        "💾 Експорт (ZIP)",
    ])

    with tab_overview:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.subheader("Активний набір даних")

        # Missing M4 repair panel check
        if data_source == "m4":
            m4_ok, _ = check_m4_local_files()
            if not m4_ok:
                render_m4_download_panel()
                st.markdown('</div>', unsafe_allow_html=True)
                return

        # Show errors if any
        if data_error:
            st.error(data_error, icon="🚨")
        elif not series_list:
            if data_source == "upload":
                st.info("Будь ласка, завантажте CSV файл у бічній панелі зліва.")
        else:
            # Active dataset summary
            lengths = [s.length for s in series_list]
            min_len = int(np.min(lengths))
            max_len = int(np.max(lengths))
            median_len = float(np.median(lengths))
            constant_count = sum(1 for s in series_list if s.is_constant)
            zeros_count = sum(1 for s in series_list if s.has_zeros)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Кількість рядів", len(series_list))
            col2.metric("Мін. довжина", min_len)
            col3.metric("Медіанна довжина", f"{median_len:.1f}")
            col4.metric("Макс. довжина", max_len)

            if data_source == "m4":
                st.success(
                    f"**Вбудовані дані M4 Yearly:** {len(series_list)} рядів "
                    f"(область: `{st.session_state.get('m4_domain', 'All')}`, seed: `{st.session_state.get('m4_seed', 42)}`)."
)
                if st.session_state.get("m4_scope") == 23000:
                    st.warning(f"⚠️ **{ua.M4_SCOPE_ALL_WARNING}**", icon="⚠️")
            else:

                detected = st.session_state.get("detected_layout", "auto")
                layout_desc = {
                    "tidy": "Окремий ряд (tidy t,y)",
                    "row_wise": "Порядковий (M4-стиль)",
                    "column_wise": "Поколонковий (колонки-ряди)",
                }.get(detected, detected)
                st.success(
                    f"**Завантажений CSV файл:** `{st.session_state.get('uploaded_file_name')}` "
                    f"(структура: **{layout_desc}**, рядів: {len(series_list)})."
)

            # Sufficiency warning in overview
            sufficiency_result = st.session_state.get("sufficiency_result")
            if sufficiency_result and not sufficiency_result.is_sufficient:
                st.error(
                    f"⚠️ **Недостатня довжина даних для обраної конфігурації:** "
                    f"потрібно мінімум {sufficiency_result.required_len} точок, "
                    f"найкоротший ряд '{sufficiency_result.shortest_series_id}' має {sufficiency_result.actual_min_len} точок "
                    f"(дефіцит: {sufficiency_result.deficit} точок).",
                    icon="🚫",
)

            # Warning badges for data characteristics
            if constant_count > 0:
                st.warning(
                    f"⚠️ Виявлено {constant_count} константних рядів (знаменник MASE дорівнює нулю). "
                    "В режимі Оцінки запуск таких рядів заблоковано.",
                    icon="⚠️",
)
            if zeros_count > 0:
                st.info(
                    f"ℹ️ {zeros_count} рядів містять нульові значення (можлива нестабільність sMAPE).",
                    icon="ℹ️",
)

            # Data preview table
            st.markdown("#### Попередній перегляд рядів")
            preview_rows = []
            for s in series_list[:10]:
                preview_rows.append({
                    "ID ряду": s.series_id,
                    "Довжина": s.length,
                    "Категорія": s.category or "—",
                    "Перші значення": ", ".join(f"{v:.3f}" for v in s.values[:5]) + ("..." if s.length > 5 else ""),
                    "Останні значення": ", ".join(f"{v:.3f}" for v in s.values[-3:]) if s.length >= 3 else "",
                    "Константний": "Так" if s.is_constant else "Ні",
                    "Містить 0": "Так" if s.has_zeros else "Ні",
                })
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
            if len(series_list) > 10:
                st.caption(f"Показано 10 рядів із {len(series_list)}.")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab_eda:
        render_eda_tab(series_list)


    with tab_results:
        render_results_tab(st.session_state.get("results"))

    with tab_download:
        eda_df = None
        if series_list:
            eda_df, _, _ = compute_dataset_eda(series_list)
        render_download_tab(st.session_state.get("results"), eda_df=eda_df)




def main() -> None:
    """Main application loop."""
    _init_session_state()
    render_header()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
