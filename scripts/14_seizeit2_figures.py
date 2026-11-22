"""Secondary figures: SeizeIT2 subtype / onset-location detection.

These are results from REAL behind-the-ear wearable hardware. They are plotted
on their own axes and never on the same scale as the CHB-MIT scalp proxies,
which are simulated wearable placements.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seizure_eeg.config import FIGURES_DIR, TABLES_DIR
from seizure_eeg.plotting import (GRID, INK, INK_2, INK_MUTED, SEQ, SERIES,
                                  despine, use_style)

DATASET_TAG = "SeizeIT2 · real behind-the-ear wearable EEG (125 patients)"

PRETTY = {
    "focal_impaired_awareness": "Focal impaired awareness",
    "focal_aware": "Focal aware",
    "focal_unknown_awareness": "Focal, awareness unknown",
    "focal_to_bilateral_tonic_clonic": "Focal → bilateral tonic-clonic",
    "focal_unspecified": "Focal, unspecified",
    "unknown_onset": "Unknown onset",
    "unclassified": "Unclassified",
    "temporal": "Temporal", "frontal": "Frontal", "central": "Central",
    "parietal": "Parietal", "occipital": "Occipital", "insular": "Insular",
    "multilobar": "Multilobar", "unknown": "Unknown / unspecified",
    "left": "Left", "right": "Right", "bilateral": "Bilateral",
    "awake": "Awake", "asleep": "Asleep",
    "ipsilateral": "Onset side = device side",
    "contralateral": "Onset side ≠ device side",
    "undetermined": "Side undetermined",
    "True": "Temporal lobe involved", "False": "Temporal lobe not involved",
}


def label_for(v, n_sz, n_pt) -> str:
    base = PRETTY.get(str(v), str(v))
    return f"{base}\n{n_sz} seizures · {n_pt} patients"


def _ci_bars(ax, tab, key, value, lo, hi, color, xlabel, title, fmt="{:.2f}",
             xmax=1.05, min_patients=3):
    """``key`` is the grouping column; it is named explicitly rather than taken
    positionally, because the tables carry ``dataset`` and ``subset`` columns."""
    tab = tab.reset_index(drop=True)
    ypos = np.arange(len(tab))[::-1]
    if key not in tab.columns:
        raise KeyError(f"{key!r} not in {list(tab.columns)}")
    for y, r in zip(ypos, tab.itertuples()):
        v = getattr(r, value)
        if not np.isfinite(v):
            continue
        thin = r.n_patients < min_patients
        ax.barh(y, v, height=0.6, color=color,
                alpha=0.38 if thin else 1.0, zorder=3)
        l, h = getattr(r, lo), getattr(r, hi)
        if np.isfinite(l) and np.isfinite(h):
            ax.plot([l, h], [y, y], color=INK_2, lw=1.5, zorder=4,
                    solid_capstyle="butt")
        ax.text(max(v, h if np.isfinite(h) else v) + xmax * 0.02, y,
                fmt.format(v) + ("  (sparse)" if thin else ""),
                va="center", ha="left", fontsize=8.5, color=INK,
                fontweight="bold", zorder=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([label_for(getattr(r, key), r.n_seizures, r.n_patients)
                        for r in tab.itertuples()], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=10)
    ax.grid(axis="y", visible=False)
    despine(ax)


def fig_subtype() -> None:
    tab = pd.read_csv(TABLES_DIR / "seizeit2_subgroup_seizure_type.csv")
    lat = tab.copy()
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                             gridspec_kw={"wspace": 0.42})
    _ci_bars(axes[0], tab, "seizure_type", "sensitivity", "sensitivity_lo",
             "sensitivity_hi", SERIES[0], "Fraction of seizures detected",
             "A   Seizure sensitivity by subtype")
    lmax = float(np.nanmax(lat.latency_hi.fillna(lat.median_latency_s))) * 1.3
    _ci_bars(axes[1], lat, "seizure_type", "median_latency_s", "latency_lo",
             "latency_hi", SERIES[2], "Seconds after annotated onset",
             "B   Detection latency by subtype", fmt="{:.0f}s",
             xmax=max(30.0, lmax))
    fig.suptitle("Detection by seizure subtype — " + DATASET_TAG + "\n"
                 "patient-disjoint 5-fold CV, threshold fitted on training "
                 "patients for ≤6 false alarms/h; bars with <3 patients are "
                 "faded and marked sparse",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.14)
    fig.savefig(FIGURES_DIR / "fig5_seizeit2_subtype.png")
    plt.close(fig)


def fig_onset() -> None:
    lobe = pd.read_csv(TABLES_DIR / "seizeit2_subgroup_onset_lobe.csv")
    hemi = pd.read_csv(TABLES_DIR / "seizeit2_subgroup_onset_hemisphere.csv")
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={"wspace": 0.42,
                                          "width_ratios": [1.25, 1]})
    _ci_bars(axes[0], lobe, "onset_lobe", "sensitivity", "sensitivity_lo",
             "sensitivity_hi", SERIES[1], "Fraction of seizures detected",
             "A   Sensitivity by annotated onset lobe")
    _ci_bars(axes[1], hemi, "onset_hemisphere", "sensitivity",
             "sensitivity_lo", "sensitivity_hi", SERIES[6],
             "Fraction of seizures detected",
             "B   Sensitivity by onset hemisphere")
    fig.suptitle("Detection by anatomical onset — " + DATASET_TAG + "\n"
                 "EXPLORATORY: behind-the-ear sensors were positioned with "
                 "reference to each patient's suspected onset zone, so "
                 "anatomy and electrode placement are confounded",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.14)
    fig.savefig(FIGURES_DIR / "fig6_seizeit2_onset_location.png")
    plt.close(fig)


def fig_proximity() -> None:
    tab = pd.read_csv(TABLES_DIR / "seizeit2_subgroup_onset_vs_device.csv")
    vig = pd.read_csv(TABLES_DIR / "seizeit2_subgroup_vigilance.csv")
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.0),
                             gridspec_kw={"wspace": 0.42})
    _ci_bars(axes[0], tab, "onset_vs_device", "sensitivity", "sensitivity_lo",
             "sensitivity_hi", SERIES[3], "Fraction of seizures detected",
             "A   Onset hemisphere versus the side the device was worn")
    _ci_bars(axes[1], vig, "vigilance", "sensitivity", "sensitivity_lo",
             "sensitivity_hi", SERIES[4], "Fraction of seizures detected",
             "B   Sensitivity by vigilance state")
    fig.suptitle("Electrode proximity and state — " + DATASET_TAG + "\n"
                 "EXPLORATORY: sensor side was itself chosen clinically, so "
                 "this is not a randomised placement comparison",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.16)
    fig.savefig(FIGURES_DIR / "fig7_seizeit2_proximity_vigilance.png")
    plt.close(fig)


def fig_annotation_inventory() -> None:
    """What the SeizeIT2 annotations contain, before any modelling."""
    ev = pd.read_csv(TABLES_DIR / "seizeit2_events_harmonised.csv")
    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                             gridspec_kw={"wspace": 0.45})
    for ax, col, color, title in [
            (axes[0], "seizure_type", SERIES[0], "A   Seizure subtype"),
            (axes[1], "onset_lobe", SERIES[1], "B   Annotated onset lobe"),
            (axes[2], "onset_hemisphere", SERIES[6], "C   Onset hemisphere")]:
        g = (ev.groupby(col)
             .agg(n_seizures=("event_index", "size"),
                  n_patients=("patient_id", "nunique"))
             .sort_values("n_seizures"))
        ypos = np.arange(len(g))
        ax.barh(ypos, g.n_seizures, height=0.62, color=color, zorder=3)
        for y, (n, p) in enumerate(zip(g.n_seizures, g.n_patients)):
            ax.text(n + ev.shape[0] * 0.012, y, f"{n}  ({p} pts)",
                    va="center", fontsize=8, color=INK, fontweight="bold")
        ax.set_yticks(ypos)
        ax.set_yticklabels([PRETTY.get(str(i), str(i)) for i in g.index],
                           fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, g.n_seizures.max() * 1.42)
        ax.set_xlabel("Annotated seizures")
        ax.set_title(title, loc="left", pad=10)
        ax.grid(axis="y", visible=False)
        despine(ax)
    fig.suptitle(f"SeizeIT2 annotation inventory — {len(ev)} seizures in "
                 f"{ev.patient_id.nunique()} patients, all focal epilepsy\n"
                 "unknown and multilobar onsets are kept as their own groups, "
                 "never folded into a single lobe",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.12)
    fig.savefig(FIGURES_DIR / "fig8_seizeit2_annotation_inventory.png")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_annotation_inventory()
    made = ["fig8_seizeit2_annotation_inventory.png"]
    if (TABLES_DIR / "seizeit2_subgroup_seizure_type.csv").exists():
        fig_subtype(); made.append("fig5_seizeit2_subtype.png")
        fig_onset(); made.append("fig6_seizeit2_onset_location.png")
        fig_proximity(); made.append("fig7_seizeit2_proximity_vigilance.png")
    else:
        print("subgroup tables missing - run scripts/13_seizeit2_benchmark.py")
    print("wrote:", ", ".join(made))


if __name__ == "__main__":
    main()
