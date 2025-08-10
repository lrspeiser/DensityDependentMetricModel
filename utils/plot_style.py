# Simple plotting style helper for consistent figures across the paper
from __future__ import annotations

import matplotlib as mpl

# Preferred color palette (colorblind-friendly)
_DEFAULT_COLORS = [
    "#1f77b4",  # blue (GR)
    "#d62728",  # red (ER)
    "#2ca02c",  # green (NFW)
    "#ff7f0e",  # orange (ER extrapolation/shading)
    "#7f7f7f",  # gray (data band)
]


def apply_paper_style():
    mpl.rcParams.update({
        "figure.figsize": (10, 7),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.prop_cycle": mpl.cycler(color=_DEFAULT_COLORS),
    })
