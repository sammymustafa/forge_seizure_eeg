"""Turn the CV outputs into the results tables quoted in the write-up."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from seizure_eeg.benchmark import bootstrap_ci
from seizure_eeg.config import (MONTAGE_CONTACTS, MONTAGE_LABELS, MONTAGE_ORDER,
                                MONTAGES, TABLES_DIR)

REF = "full_18_bipolar"


def _per_patient(patients: pd.DataFrame, metric: str, target: float) -> pd.Series:
    d = patients[patients.target_fa_per_h == target]
    return d.groupby(["montage", "patient_id"])[metric].mean()


def paired_delta(patients: pd.DataFrame, metric: str, target: float,
                 ref: str = REF) -> pd.DataFrame:
    """Per-patient difference against the reference montage, with a paired CI."""
    s = _per_patient(patients, metric, target).unstack(0)
    if ref not in s.columns:
        return pd.DataFrame()
    rows = []
    for mont in s.columns:
        if mont == ref:
            continue
        d = (s[mont] - s[ref]).dropna()
        lo, hi = bootstrap_ci(d.to_numpy())
        rows.append({"montage": mont, "metric": metric, "target_fa_per_h": target,
                     "n_patients": int(d.size), "mean_delta": float(d.mean()),
                     "ci_lo": lo, "ci_hi": hi,
                     "n_better": int((d > 0).sum()),
                     "n_worse": int((d < 0).sum())})
    return pd.DataFrame(rows)


def summary_table(patients: pd.DataFrame, folds: pd.DataFrame,
                  target: float = 6.0) -> pd.DataFrame:
    rows = []
    order = [m for m in MONTAGE_ORDER if m in set(patients.montage)]
    for mont in order:
        rec = {"montage": mont, "configuration": MONTAGE_LABELS[mont],
               "n_contacts": len(MONTAGE_CONTACTS[mont]),
               "n_bipolar_channels": len(MONTAGES[mont]),
               "n_features": int(folds[folds.montage == mont].n_features.iloc[0])}
        for metric, name in [("auprc", "auprc"),
                             ("auroc", "auroc"),
                             ("event_sensitivity", "sensitivity"),
                             ("false_alarms_per_hour", "fa_per_h"),
                             ("median_latency_s", "latency_s"),
                             ("alarm_duty_cycle", "duty_cycle")]:
            v = _per_patient(patients, metric, target).loc[mont].dropna()
            lo, hi = bootstrap_ci(v.to_numpy())
            rec[name] = float(v.mean())
            rec[f"{name}_lo"], rec[f"{name}_hi"] = lo, hi
        fd = folds[(folds.montage == mont) & (folds.target_fa_per_h == target)]
        rec["balanced_accuracy"] = float(fd.balanced_accuracy.mean())
        rec["n_seizures_scored"] = int(fd.n_seizure_events.sum())
        rec["n_detected"] = int(fd.n_detected.sum())
        rec["pooled_sensitivity"] = rec["n_detected"] / max(rec["n_seizures_scored"], 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def to_markdown(summary: pd.DataFrame, target: float) -> str:
    head = (f"| Configuration | Contacts | Bipolar ch | AUPRC [95% CI] | "
            f"Seizure sensitivity @ {target:g} FA/h target | Achieved FA/h | "
            f"Median latency | Seizures detected |\n"
            "|---|---:|---:|---|---|---:|---:|---:|\n")
    lines = []
    for r in summary.itertuples():
        lines.append(
            f"| {r.configuration} | {r.n_contacts} | {r.n_bipolar_channels} | "
            f"{r.auprc:.3f} [{r.auprc_lo:.3f}–{r.auprc_hi:.3f}] | "
            f"{r.sensitivity:.2f} [{r.sensitivity_lo:.2f}–{r.sensitivity_hi:.2f}] | "
            f"{r.fa_per_h:.1f} | {r.latency_s:.0f} s | "
            f"{r.n_detected}/{r.n_seizures_scored} |")
    return head + "\n".join(lines) + "\n"


def main() -> None:
    folds = pd.read_csv(TABLES_DIR / "cv_folds_main.csv")
    patients = pd.read_csv(TABLES_DIR / "cv_patients_main.csv")

    out_md = []
    for target in sorted(folds.target_fa_per_h.unique()):
        s = summary_table(patients, folds, target)
        s.to_csv(TABLES_DIR / f"summary_fa{target:g}.csv", index=False)
        out_md.append(f"\n### Operating point: threshold fitted for "
                      f"{target:g} false alarms/hour\n\n" + to_markdown(s, target))
        if target == 6.0:
            print(to_markdown(s, target))

    deltas = pd.concat(
        [paired_delta(patients, m, t)
         for m in ("auprc", "event_sensitivity")
         for t in sorted(folds.target_fa_per_h.unique())],
        ignore_index=True)
    deltas.to_csv(TABLES_DIR / "paired_deltas_vs_full.csv", index=False)
    print("\nPaired per-patient difference vs full 18-channel montage "
          "(AUPRC, 6 FA/h):")
    d = deltas[(deltas.metric == "auprc") & (deltas.target_fa_per_h == 6.0)]
    for r in d.itertuples():
        print(f"  {MONTAGE_LABELS[r.montage]:40s} "
              f"{r.mean_delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  "
              f"better in {r.n_better}/{r.n_patients} patients")

    (TABLES_DIR / "summary_tables.md").write_text(
        "# Montage comparison summary\n" + "".join(out_md))
    print(f"\nwrote {TABLES_DIR/'summary_tables.md'}")


if __name__ == "__main__":
    main()
