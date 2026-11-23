"""Figure 0: where each electrode configuration sits on the scalp.

An introduction figure. Each panel is a top-down view of the head (nose up)
showing the standard 10-20 positions in grey, with one configuration's physical
contacts filled in and its bipolar derivations drawn as lines between them.

The point the figure has to make is that a "4-contact" wearable is **2 bipolar
channels**, not 4 signals, so every panel states both counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Polygon

from seizure_eeg.config import (FIGURES_DIR, MONTAGE_CONTACTS, MONTAGE_LABELS,
                                MONTAGES, MONTAGE_ORDER)
from seizure_eeg.plotting import GRID, INK, INK_2, INK_MUTED, MONTAGE_COLOR, use_style

#: Standard 10-20 positions, projected top-down (x right, y anterior, head r=1).
POS: dict[str, tuple[float, float]] = {
    "FP1": (-0.309, 0.951), "FP2": (0.309, 0.951),
    "F7": (-0.809, 0.588), "F3": (-0.450, 0.520), "FZ": (0.0, 0.500),
    "F4": (0.450, 0.520), "F8": (0.809, 0.588),
    "T7": (-1.000, 0.000), "C3": (-0.500, 0.000), "CZ": (0.0, 0.0),
    "C4": (0.500, 0.000), "T8": (1.000, 0.000),
    "P7": (-0.809, -0.588), "P3": (-0.450, -0.520), "PZ": (0.0, -0.500),
    "P4": (0.450, -0.520), "P8": (0.809, -0.588),
    "O1": (-0.309, -0.951), "O2": (0.309, -0.951),
    # FT9/FT10 lie BELOW the standard plane, near the mastoid; in a top-down
    # projection they are drawn just outside the outline.
    "FT9": (-1.30, 0.34), "FT10": (1.30, 0.34),
}
BELOW_PLANE = {"FT9", "FT10"}

#: Conventional capitalisation; ``str.title()`` would give "Ft9" and "T7".
DISPLAY = {"FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz"}


def shown(name: str) -> str:
    return DISPLAY.get(name, name)


def label_anchor(x: float, y: float, pad: float = 0.19):
    """Place a label radially outward from the head centre.

    Radial placement keeps labels off the outline, off the ear arcs and off
    the derivation lines, which a fixed vertical offset does not.
    """
    n = float(np.hypot(x, y))
    if n < 1e-6:                      # Cz sits at the centre
        return x, y - 0.17, "center", "top"
    lx, ly = x + pad * x / n, y + pad * y / n
    dx, dy = x / n, y / n
    ha = "left" if dx > 0.25 else "right" if dx < -0.25 else "center"
    va = "bottom" if dy > 0.25 else "top" if dy < -0.25 else "center"
    return lx, ly, ha, va


#: White halo so a label stays readable where it crosses a derivation line.
HALO = dict(boxstyle="round,pad=0.12", facecolor="#fcfcfb", edgecolor="none",
            alpha=0.85)


def draw_head(ax) -> None:
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.6, ec=INK_2, zorder=2))
    ax.add_patch(Polygon([(-0.13, 0.985), (0, 1.14), (0.13, 0.985)],
                         closed=True, fill=False, lw=1.6, ec=INK_2, zorder=2))
    for sign in (-1, 1):
        ax.add_patch(Arc((sign * 1.0, 0.0), 0.22, 0.42,
                         theta1=270 if sign > 0 else 90,
                         theta2=90 if sign > 0 else 270,
                         lw=1.6, ec=INK_2, zorder=2))


def draw_panel(ax, montage: str | None, title: str, subtitle: str,
               label_all: bool = False) -> None:
    draw_head(ax)
    active = set(MONTAGE_CONTACTS[montage]) if montage else set()
    color = MONTAGE_COLOR[montage] if montage else INK_2

    # inactive contacts
    for name, (x, y) in POS.items():
        if name in active:
            continue
        ax.scatter([x], [y], s=46, facecolor="white", edgecolor=GRID,
                   linewidth=1.3, zorder=3)
        if label_all:
            lx, ly, ha, va = label_anchor(x, y)
            ax.text(lx, ly, shown(name), ha=ha, va=va, fontsize=8,
                    color=INK_MUTED, zorder=8, bbox=HALO)

    # derivations
    if montage:
        for ch in MONTAGES[montage]:
            a, _, b = ch.partition("-")
            if a in POS and b in POS:
                (x1, y1), (x2, y2) = POS[a], POS[b]
                ax.plot([x1, x2], [y1, y2], color=color,
                        lw=3.0 if len(MONTAGES[montage]) <= 2 else 1.8,
                        alpha=0.85, zorder=4, solid_capstyle="round")

    # active contacts
    big = len(active) > 6
    for name in sorted(active):
        x, y = POS[name]
        ax.scatter([x], [y], s=78 if big else 120, facecolor=color,
                   edgecolor="white", linewidth=1.5, zorder=6)
        lx, ly, ha, va = label_anchor(x, y, pad=0.20)
        ax.text(lx, ly, shown(name), ha=ha, va=va,
                fontsize=7.5 if big else 9.5, fontweight="bold", color=color,
                zorder=8, bbox=HALO)

    ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, pad=6)
    ax.text(0, -1.50, subtitle, ha="center", va="top", fontsize=8.5,
            color=INK_2, transform=ax.transData)
    ax.set_xlim(-1.72, 1.72)
    ax.set_ylim(-1.78, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    use_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    panels = ["__reference__"] + MONTAGE_ORDER        # 1 + 7 = 8
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.8))
    for ax, key in zip(axes.ravel(), panels):
        if key == "__reference__":
            draw_panel(ax, None, "Standard 10–20 positions",
                       "19 scalp contacts, nose at top\n"
                       "FT9 / FT10 sit below this plane, near the mastoid",
                       label_all=True)
            continue
        n_c, n_ch = len(MONTAGE_CONTACTS[key]), len(MONTAGES[key])
        name = MONTAGE_LABELS[key].split(" (")[0]
        derivs = ", ".join(c.title() for c in MONTAGES[key])
        draw_panel(ax, key, name,
                   f"{n_c} contacts → {n_ch} bipolar "
                   f"channel{'s' if n_ch > 1 else ''}\n"
                   + ", ".join(f"{shown(c.split('-')[0])}–"
                               f"{shown(c.split('-')[1])}"
                               for c in MONTAGES[key])
                   if n_ch <= 2 else
                   f"{n_c} contacts → {n_ch} bipolar channels\n"
                   "chained 10–20 montage")

    fig.suptitle("Electrode configurations compared in this benchmark",
                 fontsize=14, fontweight="bold", color=INK, x=0.012,
                 ha="left", y=0.995)
    fig.text(0.012, 0.955,
             "A bipolar EEG channel is the difference between two physical "
             "contacts — so a four-contact wearable yields two channels, not "
             "four signals. Lines show the derivations; each configuration "
             "keeps the same colour in every results figure.",
             fontsize=10, color=INK_2, ha="left")
    fig.subplots_adjust(top=0.90, bottom=0.02, hspace=0.06, wspace=0.02)
    out = FIGURES_DIR / "fig0_electrode_configurations_scalp.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
