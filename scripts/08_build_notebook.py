"""Build (and optionally execute) a results walkthrough notebook."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "forge_wearable_eeg_benchmark.ipynb"

MD = new_markdown_cell
CODE = new_code_cell

CELLS = [
    MD("""# FORGE — Wearable EEG Benchmark

**Part 1 (§1–7): the CHB-MIT electrode-reduction benchmark.**
**Part 2 (§8–12): the exploratory seizure-type and anatomical-onset analysis on
SeizeIT2, plus the harmonised schema and the TUH-ready adapter.**

This notebook reads the artefacts produced by the scripts in `scripts/`; it does
not recompute them. To regenerate everything from scratch:

```
python scripts/01_inventory.py
python scripts/02_download.py
python scripts/04_extract_and_prune.py --keep chb01
python scripts/06_experiments.py main
python scripts/06_experiments.py random
python scripts/07_report.py
python scripts/05_figures.py explore && python scripts/05_figures.py results
```

**What is being compared.** CHB-MIT channels such as `F7-T7` are *bipolar
differences between two electrodes*. A four-contact wearable mock-up gives two
bipolar channels; a two-contact one gives a single channel. These are
scalp-position proxies, not recordings from real glasses or earbuds."""),

    CODE("""import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from seizure_eeg.config import (DATA_ROOT, FIGURES_DIR, MONTAGES,
                                MONTAGE_CONTACTS, TABLES_DIR)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)
print("repo:", ROOT)"""),

    MD("""## 1. What is in the dataset

Parsed from the per-subject `chbXX-summary.txt` files, not inferred from
filenames. chb01 and chb21 are the **same person**, so the 24 cases are 23
people — they are kept in one cross-validation group."""),

    CODE("""inv = pd.read_csv(TABLES_DIR / "subject_inventory.csv")
print(f"{len(inv)} cases, {inv.n_seizures.sum()} annotated seizures, "
      f"{inv.ictal_seconds.sum()/3600:.1f} ictal hours")
display(inv[["case", "n_files_in_summary", "n_seizure_files", "n_seizures",
             "ictal_seconds"]].head(24))"""),

    MD("""## 2. The electrode configurations

Every configuration below is a list of **bipolar derivations**, with the
physical contacts each one would need."""),

    CODE("""rows = [{"configuration": m, "contacts": len(MONTAGE_CONTACTS[m]),
         "which contacts": ", ".join(MONTAGE_CONTACTS[m]),
         "bipolar channels": len(MONTAGES[m]),
         "derivations": ", ".join(MONTAGES[m])}
        for m in MONTAGES]
display(pd.DataFrame(rows))"""),

    MD("""## 3. Features from one recording

Features are computed **once for the union of every montage's channels**, so a
montage's design matrix is a column slice. That makes the comparison paired by
construction: identical recordings, identical windows, identical labels."""),

    CODE("""from seizure_eeg.dataset import (UNION_CHANNELS, available_montages,
                                 extract_recording, montage_matrix)
from seizure_eeg.summary import parse_summary

edf = DATA_ROOT / "chb01" / "chb01_03.edf"
if edf.exists():
    intervals = parse_summary(DATA_ROOT / "chb01" / "chb01-summary.txt")["chb01_03.edf"]
    X, Q, meta = extract_recording(edf, intervals)
    print("seizure intervals:", intervals)
    print("feature tensor (windows, channels, features):", X.shape)
    print("montages available in this recording:", available_montages(Q))
    for m in ["full_18_bipolar", "temporal_4_contacts", "left_temporal_2_contacts"]:
        d, usable, names = montage_matrix(X, Q, m)
        print(f"  {m:30s} design {d.shape}, usable windows {usable.sum()}")
    display(meta[meta.label == 1][["start_s", "stop_s", "ictal_overlap_s",
                                   "seizure_idx", "label"]])
else:
    print("raw chb01 EDFs not present - run scripts/02_download.py --cases chb01")"""),

    MD("""## 4. The same seizure through each electrode set"""),

    CODE("""display(Image(filename=str(FIGURES_DIR / "fig1_exploratory_montages.png")))"""),

    MD("""## 5. Patient-disjoint benchmark

Shrinkage LDA, 5-fold `GroupKFold` over patients. Decision thresholds are fitted
on out-of-fold scores from the **training** patients and applied unchanged to
the held-out patients. Confidence intervals are percentile bootstraps over
patients."""),

    CODE("""sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module
report = import_module("07_report")

folds = pd.read_csv(TABLES_DIR / "cv_folds_main.csv")
patients = pd.read_csv(TABLES_DIR / "cv_patients_main.csv")
summary = report.summary_table(patients, folds, target=6.0)
display(Markdown(report.to_markdown(summary, 6.0)))"""),

    CODE("""display(Image(filename=str(FIGURES_DIR / "fig2_montage_performance.png")))"""),

    MD("""### Does the trade-off hold at other alarm budgets?"""),

    CODE("""display(Image(filename=str(FIGURES_DIR / "fig3_sensitivity_vs_false_alarms.png")))"""),

    MD("""### Paired per-patient differences against the full montage

A paired comparison is the honest one here: the same patients, the same windows,
one configuration against another."""),

    CODE("""d = report.paired_delta(patients, "auprc", 6.0)
display(d[["montage", "n_patients", "mean_delta", "ci_lo", "ci_hi",
           "n_better", "n_worse"]])"""),

    MD("""## 6. Placement control

Every 2-channel subset of the standard 18 derivations, scored the same way. This
puts the wearable-plausible placements on a percentile instead of comparing them
only to each other."""),

    CODE("""ctrl_path = TABLES_DIR / "random_pair_control.csv"
if ctrl_path.exists():
    ctrl = pd.read_csv(ctrl_path)
    print(f"{len(ctrl)} channel pairs evaluated")
    display(ctrl.head(10))
    for name, key in [("Temporal", "F7-T7+F8-T8"),
                      ("Glasses-like", "FP1-F7+FP2-F8"),
                      ("Posterior-temporal", "T7-P7+T8-P8")]:
        row = ctrl[ctrl.channels == key]
        if not row.empty:
            v = float(row.auprc.iloc[0])
            print(f"{name:20s} AUPRC {v:.3f} - {100*(ctrl.auprc < v).mean():.0f}th percentile")
    display(Image(filename=str(FIGURES_DIR / "fig4_placement_control.png")))
else:
    print("run: python scripts/06_experiments.py random")"""),

    MD("""## 7. Per-patient variability

The spread between patients is much larger than the spread between electrode
sets — which is the main practical message for a wearable."""),

    CODE("""display(Image(filename=str(FIGURES_DIR / "fig5_per_patient.png")))
pp = (patients[patients.target_fa_per_h == 6.0]
      .pivot_table(index="patient_id", columns="montage", values="auprc"))
display(pp.round(3))"""),

    MD("""## Caveats

* CHB-MIT is paediatric, single-centre, and these are **scalp-position proxies**
  for wearables, not real device recordings.
* The task is ictal vs non-ictal classification, not seizure prediction.
* CHB-MIT has no seizure-onset-localisation labels, so the "are around-ear
  contacts better for temporal-lobe seizures?" question needs SeizeIT2.
* SeizeIT2's public release has behind-the-ear wearable signals and clinical
  annotations but **no paired full-scalp EEG**, so it supports external
  validation, not a same-recording electrode ablation."""),

    MD("""---

# Part 2 — Seizure type and anatomical onset

Everything below uses **SeizeIT2** (OpenNeuro `ds005873`): real behind-the-ear
wearable EEG from 125 patients with focal epilepsy, with clinical annotations
for seizure subtype, onset lobe, hemisphere and vigilance.

**These are not interchangeable with Part 1.** CHB-MIT gives *scalp-position
proxies* for a wearable; SeizeIT2 gives a *real* two-channel around-ear device
with no paired full-scalp recording. The two are never plotted on shared axes,
and no full-versus-wearable claim is made from SeizeIT2.

## 8. The harmonised dataset layer

One schema spans all three corpora, so a subgroup analysis is written once."""),

    CODE("""from seizure_eeg.datasets.chbmit import CHBMITAdapter
from seizure_eeg.datasets.seizeit2 import SeizeIT2Adapter
from seizure_eeg.datasets.tuh import TUSZAdapter, mapping_plan_frame

adapters = {
    "CHB-MIT (scalp proxy)": CHBMITAdapter(ROOT / "data" / "chbmit"),
    "SeizeIT2 (real wearable)": SeizeIT2Adapter(ROOT / "data" / "seizeit2"),
    "TUSZ (planned)": TUSZAdapter(ROOT / "data" / "tusz"),
}
avail = pd.DataFrame({name: ad.label_availability()
                      for name, ad in adapters.items()})
display(Markdown("**Which clinical labels each dataset actually carries**"))
display(avail)
display(Markdown(
    "`not_annotated` means the dataset has no such field. It is never filled "
    "in by inferring anatomy from the signal - that would manufacture a "
    "clinical label and make the subgroup analysis circular."))"""),

    MD("""## 9. SeizeIT2 annotation inventory

Parsed straight from the BIDS `*_events.tsv` files. Multilobar and unknown
onsets are kept as their own categories rather than folded into one lobe."""),

    CODE("""ev2 = pd.read_csv(TABLES_DIR / "seizeit2_events_harmonised.csv")
print(f"{len(ev2)} seizures, {ev2.patient_id.nunique()} patients, "
      f"{ev2.recording_id.nunique()} recordings, "
      f"{ev2.duration_s.sum()/3600:.1f} ictal hours")
for col in ["seizure_type", "onset_lobe", "onset_hemisphere", "vigilance"]:
    g = (ev2.groupby(col)
         .agg(seizures=("event_index", "size"),
              patients=("patient_id", "nunique"))
         .sort_values("seizures", ascending=False))
    display(Markdown(f"**by `{col}`**"))
    display(g)"""),

    CODE("""display(Image(filename=str(FIGURES_DIR / "fig8_seizeit2_annotation_inventory.png")))"""),

    MD("""### How the raw ILAE codes are harmonised

`sz_foc_a` (focal aware) and `sz_foc_ia` (focal impaired awareness) share a
string prefix, so matching is on complete underscore segments, not prefixes."""),

    CODE("""from seizure_eeg.datasets.seizeit2 import (harmonise_localization,
                                            harmonise_seizure_type)

display(pd.DataFrame(
    [{"raw eventType": c, "harmonised": harmonise_seizure_type(c)}
     for c in ["sz_foc_a", "sz_foc_a_m_automatisms", "sz_foc_ia",
               "sz_foc_ia_nm_behavior", "sz_foc_ua_m", "sz_foc_f2b",
               "sz_gen_m_tonicClonic", "sz_uo", "sz"]]))
display(pd.DataFrame(
    [{"raw localization": c, "onset_lobe": harmonise_localization(c)[0],
      "all lobes": ", ".join(harmonise_localization(c)[1]) or "-"}
     for c in ["temp", "front", "front_temp", "front_cen_temp",
               "temp_par_occ", "ins", "un"]]))"""),

    MD("""## 10. Detection on real wearable EEG, by seizure subtype

Same pipeline as Part 1: 10 s windows, same feature set, shrinkage LDA,
patient-disjoint 5-fold CV, thresholds fitted on training patients only."""),

    CODE("""folds2 = pd.read_csv(TABLES_DIR / "seizeit2_cv_folds.csv")
out2 = pd.read_csv(TABLES_DIR / "seizeit2_event_outcomes.csv")
m6 = out2[out2.target_fa_per_h == 6.0]
f6 = folds2[folds2.target_fa_per_h == 6.0]
print(f"{int(m6.detected.sum())}/{len(m6)} seizures detected "
      f"({m6.detected.mean():.1%}) at the 6 FA/h target; "
      f"median latency {m6.loc[m6.detected,'latency_s'].median():.0f} s; "
      f"mean AUPRC {f6.auprc.mean():.3f} (prevalence {f6.prevalence.mean():.4f})")
display(pd.read_csv(TABLES_DIR / "seizeit2_subgroup_seizure_type.csv")
        .drop(columns=["dataset"], errors="ignore").round(3))
display(Image(filename=str(FIGURES_DIR / "fig5_seizeit2_subtype.png")))"""),

    MD("""## 11. Anatomical onset — exploratory

> The behind-the-ear sensors were positioned with reference to each patient's
> suspected onset zone, so anatomy and electrode placement are confounded.
> Read every row here as hypothesis-generating."""),

    CODE("""display(pd.read_csv(TABLES_DIR / "seizeit2_subgroup_onset_lobe.csv")
        .drop(columns=["dataset"], errors="ignore").round(3))
display(Image(filename=str(FIGURES_DIR / "fig6_seizeit2_onset_location.png")))"""),

    MD("""### Does the temporal-lobe hypothesis survive its confounds?

`involves_temporal == False` pools genuinely non-temporal onsets with seizures
whose localisation was never annotated. And focal-to-bilateral seizures are
detected almost perfectly, so they have to be excluded to separate the
anatomical question from the seizure-type one."""),

    CODE("""con = pd.read_csv(TABLES_DIR / "seizeit2_subgroup_contrasts.csv")
display(con[con.grouping.isin(["onset_lobe", "involves_temporal"])]
        [["grouping", "subset", "comparison", "n_seizures_a", "n_seizures_b",
          "sensitivity_a", "sensitivity_b", "delta", "ci_lo", "ci_hi"]]
        .round(3))"""),

    CODE("""display(Image(filename=str(FIGURES_DIR / "fig7_seizeit2_proximity_vigilance.png")))
display(con[con.grouping.isin(["onset_vs_device", "vigilance", "seizure_type"])]
        [["grouping", "subset", "comparison", "n_seizures_a", "n_seizures_b",
          "sensitivity_a", "sensitivity_b", "delta", "ci_lo", "ci_hi"]]
        .round(3))"""),

    MD("""## 12. TUH (TUSZ) — prepared, not run

The adapter is a skeleton and refuses to produce results until a real corpus is
mounted, so no TUH number can appear by accident."""),

    CODE("""print("TUSZAdapter.IMPLEMENTED =", TUSZAdapter.IMPLEMENTED)
try:
    adapters["TUSZ (planned)"].recordings()
except NotImplementedError as exc:
    print("\\nrecordings() refuses:\\n ", str(exc)[:240], "...")
display(Markdown("**Seizure-code mapping plan**"))
display(mapping_plan_frame())"""),

    MD("""### Two rules that must survive TUH integration

1. **A seizure-type code is not an anatomical onset label.** `cpsz` means focal
   with impaired awareness; it says nothing about which lobe.
2. **Channel-level annotations are not localisation.** TUSZ marks which
   channels an event was seen on — an electrographic field shaped by montage
   and propagation, not a validated onset lobe.

The planned TUSZ analysis is the one neither corpus here can do: **does the gap
between full-scalp and wearable-inspired configurations depend on seizure
type?** TUSZ is the only one of the three with both full-scalp EEG and clinical
seizure-type codes.

The full write-up, with every number regenerated from the result tables, is in
`REPORT.md`."""),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    nb = new_notebook(cells=CELLS, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print("wrote", OUT)

    if args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(OUT.parent)}})
        client.execute()
        nbf.write(nb, OUT)
        print("executed", OUT)


if __name__ == "__main__":
    main()
