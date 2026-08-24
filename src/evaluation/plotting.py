"""Shared plotting style for M4 comparison figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M4_METHOD_STYLE = {
    "GRNN": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.6},
    "GRNN-Mean": {"color": "#009E73", "linestyle": "-", "linewidth": 1.6},
    "GRNN-Median": {"color": "#D55E00", "linestyle": "-", "linewidth": 1.8},
    "GRNN-Bag": {"color": "#CC79A7", "linestyle": "-", "linewidth": 1.6},
    "118-Smyl": {"color": "#E69F00", "linestyle": "--", "linewidth": 1.3},
    "245-FFORMA": {"color": "#56B4E9", "linestyle": "--", "linewidth": 1.3},
    "Com": {"color": "#F0E442", "linestyle": "--", "linewidth": 1.3},
    "ARIMA": {"color": "#7F7F7F", "linestyle": "--", "linewidth": 1.3},
    "MLP": {"color": "#8B4513", "linestyle": "--", "linewidth": 1.3},
    "Naive2": {"color": "#4D4D4D", "linestyle": ":", "linewidth": 1.4},
}


def apply_m4_style() -> None:
    """Set deterministic, print-readable Matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
    })


def save_m4_figure(fig: plt.Figure, path: Path) -> None:
    """Save a PNG without timestamp-like metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Date": None, "Software": "grnn_mcs"},
    )
    plt.close(fig)
