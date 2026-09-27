"""UI design tokens, color palette, and visual styling.

Central color palette constants for the UI theme (light theme).
All UI modules and plotting routines import colors from this module.
"""

from typing import Final
import streamlit as st

# ==============================================================================
# Structural UI Design Tokens (Light Interface)
# ==============================================================================
MAIN_BACKGROUND: Final[str] = "#fdfcfc"
SURFACE: Final[str] = "#ffffff"
PRIMARY_TEXT: Final[str] = "#1f1f1f"
MUTED_TEXT: Final[str] = "rgba(0, 0, 0, 0.55)"
TEXT_ON_PRIMARY: Final[str] = "#ffffff"

# ==============================================================================
# Vibrant Brand and Accent Tokens
# ==============================================================================
DODGER_BLUE: Final[str] = "#3A8BFF"      # Primary accent: run button, progress bar, forecast line
EMERALD_GREEN: Final[str] = "#4CC88B"    # Positive accent: actual/holdout line, success notices
GOLDENROD_YELLOW: Final[str] = "#D0C42E" # Warning accent: sample runtime warnings, notices
CORAL_RED: Final[str] = "#EE5257"        # Error accent: insufficiency highlights, cancel, errors

# ==============================================================================
# Soft Pastel Transition Tokens (Gradients & Accents)
# ==============================================================================
MUTED_BLUE: Final[str] = "#4796E3"
SOFT_PURPLE: Final[str] = "#9177C7"
WARM_ROSE: Final[str] = "#CA6673"

# ==============================================================================
# Plot Role Palette
# ==============================================================================
PLOT_COLORS: Final[dict[str, str]] = {
    "history": PRIMARY_TEXT,             # Charcoal (#1f1f1f) for historical data
    "forecast": DODGER_BLUE,             # Dodger Blue (#3A8BFF) for model forecasts
    "actual": EMERALD_GREEN,             # Emerald Green (#4CC88B) for actual holdout points
    "boundary": "rgba(0, 0, 0, 0.35)",   # Muted marker for train/test split boundary
}

# ==============================================================================
# Custom CSS Injection
# ==============================================================================
CUSTOM_CSS: Final[str] = f"""
<style>
/* Main canvas background */
.stApp {{
    background-color: {MAIN_BACKGROUND};
    color: {PRIMARY_TEXT};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}

/* Surface cards & containers */
div[data-testid="stVerticalBlock"] > div.css-card,
div.ui-card {{
    background-color: {SURFACE};
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 8px;
    padding: 1.25rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    margin-bottom: 1rem;
}}

/* Typography polish */
h1, h2, h3, h4, h5, h6 {{
    color: {PRIMARY_TEXT};
    font-weight: 600;
    letter-spacing: -0.01em;
}}

/* Muted text utility */
.muted-text {{
    color: {MUTED_TEXT};
    font-size: 0.875rem;
}}

/* Live sufficiency banner */
.sufficiency-banner {{
    padding: 0.6rem 0.9rem;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 0.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.sufficiency-banner.ok {{
    background-color: rgba(76, 200, 139, 0.12);
    border: 1px solid rgba(76, 200, 139, 0.4);
    color: #1a7a49;
}}
.sufficiency-banner.insufficient {{
    background-color: rgba(238, 82, 87, 0.12);
    border: 1px solid rgba(238, 82, 87, 0.4);
    color: #a82024;
}}

/* Header pill badge */
.badge-pill {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, {DODGER_BLUE}, {MUTED_BLUE});
    color: {TEXT_ON_PRIMARY};
    margin-bottom: 0.5rem;
}}

/* Buttons polish */
button[kind="primary"] {{
    background-color: {DODGER_BLUE} !important;
    color: {TEXT_ON_PRIMARY} !important;
    border: none !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    transition: opacity 0.2s ease, transform 0.1s ease;
}}
button[kind="primary"]:hover {{
    opacity: 0.92;
    transform: translateY(-1px);
}}
button[kind="secondary"] {{
    border-radius: 6px !important;
    font-weight: 500 !important;
}}

/* Metric cards in Streamlit */
[data-testid="stMetricValue"] {{
    color: {PRIMARY_TEXT};
    font-weight: 700;
}}
[data-testid="stMetricLabel"] {{
    color: {MUTED_TEXT};
    font-size: 0.825rem;
}}

/* Progress bar color */
.stProgress > div > div > div > div {{
    background-color: {DODGER_BLUE} !important;
}}
</style>
"""


def inject_custom_css() -> None:
    """Inject the unified design token CSS into the active Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
