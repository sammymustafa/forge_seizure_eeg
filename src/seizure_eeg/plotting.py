"""Shared matplotlib styling and a validated categorical palette."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, in the fixed order they must be assigned in.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e4e3de"
# Single-hue sequential ramp (blue), light -> dark.
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

#: Fixed montage -> colour map, so a montage keeps its colour in every figure
#: even when a plot shows only a subset.
MONTAGE_COLOR = {
    "full_18_bipolar": SERIES[0],
    "glasses_4_contacts": SERIES[1],
    "temporal_4_contacts": SERIES[2],
    "posterior_temporal_4_contacts": SERIES[3],
    "lower_temporal_4_contacts": SERIES[6],
    "left_temporal_2_contacts": SERIES[4],
    "right_temporal_2_contacts": SERIES[5],
}

SEIZURE_SHADE = "#f0d9d9"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.family": "DejaVu Sans",
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
