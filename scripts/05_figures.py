"""Figures.

  explore  Fig 1: the same seizure seen through each montage (chb01_03).
  results  Figs 2-5: montage comparison, FA trade-off, placement control,
           per-patient spread. Requires scripts/06_experiments.py output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seizure_eeg.benchmark import bootstrap_ci
from seizure_eeg.config import (DATA_ROOT, FEATURES_DIR, FEATURES_PER_CHANNEL,
                                FIGURES_DIR, MONTAGE_CONTACTS, MONTAGE_LABELS,
                                MONTAGE_ORDER, MONTAGES, TABLES_DIR)
from seizure_eeg.dataset import CH_INDEX
from seizure_eeg.plotting import (GRID, INK, INK_2, INK_MUTED, MONTAGE_COLOR,
                                  SEIZURE_SHADE, SERIES, despine, use_style)
from seizure_eeg.summary import parse_summary

SHOW = ["full_18_bipolar", "glasses_4_contacts", "temporal_4_contacts",
        "posterior_temporal_4_contacts"]


# ----------------------------------------------------------------- figure 1
def fig_explore() -> None:
    import mne
    case, fname = "chb01", "chb01_03.edf"
    edf = DATA_ROOT / case / fname
    if not edf.exists():
        raise SystemExit(f"{edf} not found (raw EDF pruned?)")
    intervals = parse_summary(DATA_ROOT / case / f"{case}-summary.txt")[fname]
    onset, offset = intervals[0]

    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    traces = [("FP1-F7", "glasses_4_contacts", "Glasses-like"),
              ("F7-T7", "temporal_4_contacts", "Temporal"),
              ("T7-P7", "posterior_temporal_4_contacts", "Posterior-temporal"),
              ("FZ-CZ", "full_18_bipolar", "Midline (full scalp only)")]
    t0, t1 = onset - 40, offset + 40
    i0, i1 = int(t0 * sf), int(t1 * sf)
    picks = [raw.ch_names.index(ch) for ch, _, _ in traces]
    sig = raw.get_data(picks=picks, start=i0, stop=i1) * 1e6   # microvolts
    raw.close()
    t = np.arange(sig.shape[1]) / sf + t0

    with np.load(FEATURES_DIR / f"{case}.npz", allow_pickle=False) as z:
        X = z["X"]
    meta = pd.read_csv(FEATURES_DIR / f"{case}_meta.csv.gz")
    sel = (meta.edf_file == fname).to_numpy()
    Xf, mf = X[sel], meta[sel].reset_index(drop=True)
    fidx = FEATURES_PER_CHANNEL.index("log_line_length")

    use_style()
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.8),
                             gridspec_kw={"height_ratios": [1.5, 1],
                                          "hspace": 0.45})

    ax = axes[0]
    step = 620.0
    top = (len(traces) - 1) * step
    for k, (ch, mont, nice) in enumerate(traces):
        base = (len(traces) - 1 - k) * step
        y = np.clip(sig[k] - np.median(sig[k]), -step * 0.46, step * 0.46) + base
        ax.plot(t, y, color=MONTAGE_COLOR[mont], lw=0.5, alpha=0.95)
        ax.text(t0 + 1, base + step * 0.40, f"{ch}  ·  {nice}",
                color=MONTAGE_COLOR[mont], fontsize=9, fontweight="bold",
                ha="left", va="bottom")
    ax.axvspan(onset, offset, color=SEIZURE_SHADE, zorder=0)
    ax.text((onset + offset) / 2, top + step * 0.60,
            f"annotated seizure  ({onset}–{offset} s)",
            ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set(xlim=(t0, t1), ylim=(-step * 0.62, top + step * 0.92))
    ax.set_xlabel("Time in recording (s)", labelpad=2)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    xs = t1 + (t1 - t0) * 0.012
    ax.plot([xs, xs], [-step * 0.45, -step * 0.45 + 200], color=INK_2, lw=1.6,
            clip_on=False)
    ax.text(xs + (t1 - t0) * 0.008, -step * 0.45 + 100, "200 µV", fontsize=8.5,
            color=INK_2, ha="left", va="center", clip_on=False)
    ax.set_title("A   One seizure (chb01_03), band-passed 0.5–40 Hz, as each "
                 "electrode set would see it", loc="left", pad=16)
    despine(ax, keep=("bottom",))

    ax = axes[1]
    for mont in SHOW:
        cols = [CH_INDEX[c] for c in MONTAGES[mont]]
        v = np.nanmean(Xf[:, cols, fidx], axis=1)
        z = (v - np.nanmedian(v)) / (np.nanpercentile(v, 75)
                                     - np.nanpercentile(v, 25) + 1e-9)
        ax.plot(mf.start_s / 60, z, lw=1.3, alpha=0.9,
                color=MONTAGE_COLOR[mont], label=MONTAGE_LABELS[mont])
    ax.axvspan(onset / 60, offset / 60, color=SEIZURE_SHADE, zorder=0)
    ax.annotate("seizure", xy=(onset / 60, ax.get_ylim()[1] * 0.80),
                xytext=(onset / 60 - 8, ax.get_ylim()[1] * 0.80),
                fontsize=9, color=INK_2, ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=INK_2, lw=1))
    ax.set(xlabel="Minutes from recording start",
           ylabel="Line length\n(robust z, montage mean)")
    ax.set_title("B   Mean line length per montage across the whole hour — the "
                 "ictal excursion survives channel reduction", loc="left",
                 pad=10)
    ax.legend(loc="upper left", ncol=2)
    despine(ax)

    fig.suptitle("Same recording, same windows, different electrode sets",
                 fontsize=13.5, fontweight="bold", color=INK, x=0.012,
                 ha="left", y=0.985)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "fig1_exploratory_montages.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------- figure 2-5
def _agg(patients: pd.DataFrame, metric: str, target: float) -> pd.DataFrame:
    d = patients[patients.target_fa_per_h == target]
    rows = []
    for mont, g in d.groupby("montage"):
        vals = g.groupby("patient_id")[metric].mean().to_numpy()
        lo, hi = bootstrap_ci(vals)
        rows.append({"montage": mont, "mean": np.nanmean(vals),
                     "lo": lo, "hi": hi, "n": len(vals)})
    return pd.DataFrame(rows).set_index("montage")


def _bar_panel(ax, stats, order, title, xlabel, fmt="{:.2f}", xmax=None):
    ypos = np.arange(len(order))[::-1]
    for y, mont in zip(ypos, order):
        if mont not in stats.index:
            continue
        r = stats.loc[mont]
        c = MONTAGE_COLOR[mont]
        ax.barh(y, r["mean"], height=0.62, color=c, zorder=3)
        ax.plot([r["lo"], r["hi"]], [y, y], color=INK_2, lw=1.4, zorder=4,
                solid_capstyle="butt")
        span = xmax or stats["hi"].max()
        ax.text(max(r["mean"], r["hi"]) + span * 0.02, y,
                fmt.format(r["mean"]), va="center", ha="left", fontsize=8.5,
                color=INK, fontweight="bold", zorder=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([MONTAGE_LABELS[m] for m in order], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=10)
    ax.grid(axis="y", visible=False)
    if xmax:
        ax.set_xlim(0, xmax)
    despine(ax)


def fig_results() -> None:
    folds = pd.read_csv(TABLES_DIR / "cv_folds_main.csv")
    patients = pd.read_csv(TABLES_DIR / "cv_patients_main.csv")
    order = [m for m in MONTAGE_ORDER if m in set(folds.montage)]
    use_style()

    # ---- fig 2: headline comparison
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4),
                             gridspec_kw={"wspace": 0.10})
    au = _agg(patients, "auprc", 6.0)
    _bar_panel(axes[0], au, order, "A   AUPRC",
               "Area under precision–recall curve", "{:.3f}",
               xmax=max(0.05, au["hi"].max() * 1.30))
    se = _agg(patients, "event_sensitivity", 6.0)
    _bar_panel(axes[1], se, order, "B   Seizure sensitivity",
               "Fraction of seizures detected", "{:.2f}", xmax=1.12)
    axes[1].set_yticklabels([])
    la = _agg(patients, "median_latency_s", 6.0)
    _bar_panel(axes[2], la, order, "C   Median detection latency",
               "Seconds after annotated onset", "{:.0f}s",
               xmax=max(10, la["hi"].max() * 1.32))
    axes[2].set_yticklabels([])
    n_pat = patients.patient_id.nunique()
    fig.suptitle("Patient-disjoint performance by electrode set — "
                 f"{n_pat} patients, identical windows, thresholds fitted on "
                 "training patients for ≤6 false alarms/h\n"
                 "bars = mean over held-out patients, line = 95% bootstrap CI "
                 "over patients",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.10)
    fig.savefig(FIGURES_DIR / "fig2_montage_performance.png")
    plt.close(fig)

    # ---- fig 3: sensitivity vs allowed false-alarm rate
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    targets = sorted(patients.target_fa_per_h.unique())
    for i, mont in enumerate(order):
        xs, ys = [], []
        for tgt in targets:
            g = patients[(patients.montage == mont)
                         & (patients.target_fa_per_h == tgt)]
            xs.append(g.groupby("patient_id").false_alarms_per_hour.mean().mean())
            ys.append(g.groupby("patient_id").event_sensitivity.mean().mean())
        ax.plot(xs, ys, marker="o", color=MONTAGE_COLOR[mont],
                label=MONTAGE_LABELS[mont], lw=2, markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        if i == 0:   # label the operating points once, on the reference series
            xmax_pt = max(xs)
            for x, y, tgt in zip(xs, ys, targets):
                right = x > 0.6 * xmax_pt
                ax.annotate(f"{tgt:g} FA/h target", (x, y),
                            xytext=(-8 if right else 8, -13),
                            textcoords="offset points", fontsize=8,
                            color=INK_MUTED, ha="right" if right else "left")
    ax.set(xlabel="Achieved false alarms per hour on held-out patients",
           ylabel="Seizure sensitivity", ylim=(0.3, 1.02))
    ax.set_title("Sensitivity / false-alarm trade-off — at a matched alarm "
                 "budget the reduced sets are not behind\n"
                 "each point is one target budget; thresholds fitted on "
                 "training patients, FA/h measured on held-out patients",
                 loc="left", pad=10)
    ax.legend(loc="lower right", ncol=2)
    despine(ax)
    fig.savefig(FIGURES_DIR / "fig3_sensitivity_vs_false_alarms.png")
    plt.close(fig)

    # ---- fig 4: placement control
    ctrl_path = TABLES_DIR / "random_pair_control.csv"
    if ctrl_path.exists():
        ctrl = pd.read_csv(ctrl_path)
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.hist(ctrl.auprc, bins=28, color="#9ec5f4", edgecolor="white",
                linewidth=0.8, zorder=3)
        named = {"temporal_4_contacts": "F7-T7+F8-T8",
                 "glasses_4_contacts": "FP1-F7+FP2-F8",
                 "posterior_temporal_4_contacts": "T7-P7+T8-P8"}
        ymax = ax.get_ylim()[1]
        for i, (mont, key) in enumerate(named.items()):
            row = ctrl[ctrl.channels == key]
            if row.empty:
                continue
            v = float(row.auprc.iloc[0])
            pct = 100.0 * (ctrl.auprc < v).mean()
            ax.axvline(v, color=MONTAGE_COLOR[mont], lw=2, zorder=4)
            ax.text(v, ymax * (0.95 - 0.14 * i),
                    f" {MONTAGE_LABELS[mont].split(' (')[0]} — {pct:.0f}th pct",
                    color=MONTAGE_COLOR[mont], fontsize=8.5,
                    fontweight="semibold", va="top")
        ax.set(xlabel="AUPRC (patient-disjoint CV)",
               ylabel="Number of 2-channel electrode sets")
        ax.set_title("Placement control: every 2-channel subset of the standard "
                     "18 derivations\nwhere the wearable-plausible placements "
                     "land in that distribution", loc="left")
        despine(ax)
        fig.savefig(FIGURES_DIR / "fig4_placement_control.png")
        plt.close(fig)

    # ---- fig 5: per-patient paired spread
    d = patients[patients.target_fa_per_h == 6.0]
    piv = d.pivot_table(index="patient_id", columns="montage", values="auprc")
    cols = [m for m in order if m in piv.columns]
    piv = piv[cols].sort_values(cols[0], ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    xs = np.arange(len(piv))
    # Patients are discrete, so draw the per-patient range rather than joining
    # categories with a line that would imply continuity.
    for x, (_, row) in zip(xs, piv.iterrows()):
        v = row.dropna()
        if len(v) > 1:
            ax.plot([x, x], [v.min(), v.max()], color=GRID, lw=5,
                    solid_capstyle="round", zorder=1)
    for mont in cols:
        ax.scatter(xs, piv[mont], s=38, color=MONTAGE_COLOR[mont],
                   label=MONTAGE_LABELS[mont], zorder=3,
                   edgecolor="white", linewidth=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(piv.index, rotation=45, ha="right", fontsize=8.5)
    ax.set_xlim(-0.7, len(piv) - 0.3)
    ax.set(ylabel="AUPRC on held-out patient", xlabel="")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Per-patient AUPRC — the spread between patients dwarfs the "
                 "spread between electrode sets\n"
                 "patients ordered by full-scalp AUPRC; grey bar spans the "
                 "electrode sets for that patient", loc="left", pad=10)
    ax.legend(loc="upper right", ncol=2)
    despine(ax)
    fig.savefig(FIGURES_DIR / "fig5_per_patient.png")
    plt.close(fig)
    fig_electrode_count()
    fig_paired_loss()
    print("wrote figures to", FIGURES_DIR)


def fig_electrode_count() -> None:
    """Primary figure: performance against physical electrode count.

    Counts are drawn as ordered groups rather than on a numeric axis: several
    configurations share a contact count and would otherwise be plotted on top
    of each other, and the three counts (2, 4, 19) are far from evenly spaced.
    """
    folds = pd.read_csv(TABLES_DIR / "cv_folds_main.csv")
    patients = pd.read_csv(TABLES_DIR / "cv_patients_main.csv")
    order = [m for m in MONTAGE_ORDER if m in set(folds.montage)]
    counts = sorted({len(MONTAGE_CONTACTS[m]) for m in order})
    chans = {c: sorted({len(MONTAGES[m]) for m in order
                        if len(MONTAGE_CONTACTS[m]) == c})
             for c in counts}

    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0),
                             gridspec_kw={"wspace": 0.22})
    for ax, metric, label, title in [
            (axes[0], "auprc", "AUPRC", "A   AUPRC"),
            (axes[1], "event_sensitivity", "Seizure sensitivity",
             "B   Seizure sensitivity at ≤6 FA/h")]:
        stats = _agg(patients, metric, 6.0)
        group_means = []
        for gi, count in enumerate(counts):
            members = [m for m in order
                       if len(MONTAGE_CONTACTS[m]) == count and m in stats.index]
            width = 0.52
            offs = (np.linspace(-width / 2, width / 2, len(members))
                    if len(members) > 1 else np.array([0.0]))
            vals = []
            for off, mont in zip(offs, members):
                r = stats.loc[mont]
                x = gi + off
                ax.plot([x, x], [r["lo"], r["hi"]], color=INK_2, lw=1.5,
                        zorder=3, solid_capstyle="butt")
                ax.scatter([x], [r["mean"]], s=86, color=MONTAGE_COLOR[mont],
                           zorder=4, edgecolor="white", linewidth=1.3,
                           label=MONTAGE_LABELS[mont])
                vals.append(r["mean"])
            group_means.append(np.mean(vals) if vals else np.nan)
        ax.plot(np.arange(len(counts)), group_means, color=INK_MUTED, lw=1.5,
                ls="--", zorder=2)
        ax.set_xticks(np.arange(len(counts)))
        ax.set_xticklabels(
            [f"{c} contacts\n{'/'.join(str(k) for k in chans[c])} "
             f"bipolar channel{'s' if max(chans[c]) > 1 else ''}"
             for c in counts], fontsize=9)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(counts) - 0.4)
        ax.set(xlabel="", ylabel=label)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", pad=10)
        despine(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Performance against electrode count — CHB-MIT scalp "
                 "proxies, 23 patients, identical windows\n"
                 "each point is one configuration (mean over held-out "
                 "patients, 95% bootstrap CI); the dashed line joins the mean "
                 "at each count",
                 fontsize=11, fontweight="bold", x=0.008, ha="left", y=1.06)
    fig.savefig(FIGURES_DIR / "fig9_performance_vs_electrode_count.png")
    plt.close(fig)


def fig_paired_loss() -> None:
    """Primary figure: the paired performance-loss distribution."""
    patients = pd.read_csv(TABLES_DIR / "cv_patients_main.csv")
    d = patients[patients.target_fa_per_h == 6.0]
    piv = d.groupby(["montage", "patient_id"]).auprc.mean().unstack(0)
    if "full_18_bipolar" not in piv.columns:
        return
    order = [m for m in MONTAGE_ORDER
             if m in piv.columns and m != "full_18_bipolar"]
    use_style()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    rng = np.random.default_rng(0)
    ypos = np.arange(len(order))[::-1]
    for y, mont in zip(ypos, order):
        delta = (piv[mont] - piv["full_18_bipolar"]).dropna()
        jitter = rng.uniform(-0.14, 0.14, len(delta))
        ax.scatter(delta, y - 0.10 + jitter, s=26, color=MONTAGE_COLOR[mont],
                   alpha=0.55, zorder=3, edgecolor="white", linewidth=0.6)
        lo, hi = bootstrap_ci(delta.to_numpy())
        ax.plot([lo, hi], [y + 0.26, y + 0.26], color=INK, lw=2.4, zorder=5,
                solid_capstyle="round")
        ax.scatter([delta.mean()], [y + 0.26], s=60, color=INK, zorder=6,
                   marker="D", edgecolor="white", linewidth=1)
        sig = "" if lo <= 0 <= hi else "  ✱"
        ax.text(max(hi, 0.012) + 0.012, y + 0.26, f"{delta.mean():+.3f} "
                f"[{lo:+.3f}, {hi:+.3f}]{sig}", va="center", fontsize=8.5,
                color=INK, fontweight="bold")
    ax.axvline(0, color=INK_2, lw=1.4, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([MONTAGE_LABELS[m] for m in order], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Per-patient AUPRC difference from the full 18-channel "
                  "montage  (negative = worse)")
    ax.set_xlim(piv[order].sub(piv["full_18_bipolar"], axis=0).min().min() - 0.05,
                0.42)
    ax.grid(axis="y", visible=False)
    ax.set_title("Paired performance loss, with its uncertainty\n"
                 "one dot per held-out patient; diamond and bar are the mean "
                 "and its 95% patient bootstrap CI; ✱ marks an interval "
                 "excluding zero", loc="left", pad=10)
    despine(ax)
    fig.savefig(FIGURES_DIR / "fig10_paired_performance_loss.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["explore", "results"])
    args = ap.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    {"explore": fig_explore, "results": fig_results}[args.command]()


if __name__ == "__main__":
    main()
