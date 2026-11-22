# Seizure detection with reduced, wearable-plausible electrode sets

How much seizure-detection performance is lost when a full scalp EEG montage is
cut down to the handful of contacts that would fit in smart glasses or an
around-ear wearable?

Every configuration is evaluated on **the same recordings, the same 10-second
windows and the same annotations**, so nothing in the comparison is confounded
by recording conditions. Validation is **patient-disjoint**.

Dataset: [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
(Guttag, 2010) — 24 recorded cases, 23 distinct people, 198 annotated seizures.

## What is actually being compared

CHB-MIT channels such as `F7-T7` are **bipolar differences between two physical
electrodes**, not independent channels. A four-contact wearable mock-up
therefore yields **two bipolar channels**, and a two-contact one yields a single
bipolar channel.

| Configuration | Electrode contacts | Bipolar channels |
|---|---|---|
| Full scalp | 19 | 18 |
| Glasses-like | Fp1, F7, Fp2, F8 | `FP1-F7`, `FP2-F8` |
| Temporal | F7, T7, F8, T8 | `F7-T7`, `F8-T8` |
| Posterior-temporal | T7, P7, T8, P8 | `T7-P7`, `T8-P8` |
| Lower-temporal | T7, FT9, FT10, T8 | `T7-FT9`, `FT10-T8` |
| Two-contact (left / right) | F7, T7 *or* F8, T8 | one channel |

These are **scalp-position proxies**, not recordings from real earbud or glasses
hardware. Real devices differ in contact material, skin coupling, reference
placement, motion artefact and amplifier noise. The lower-temporal pair is the
closest scalp analogue of a behind-the-ear placement, and FT9/FT10 are absent
from three subjects (chb15, chb18, chb19).

## Pipeline

```
scripts/01_inventory.py          # fetch all 24 summaries, parse annotations
scripts/02_download.py           # fetch the EDFs listed in those summaries
scripts/03_extract_features.py   # EDF -> 10 s window features (parallel)
scripts/04_extract_and_prune.py  # same, but deletes each EDF once featurised
scripts/05_figures.py explore    # Fig 1 (needs raw chb01 EDFs)
scripts/06_experiments.py main   # patient-disjoint CV across montages
scripts/06_experiments.py random # placement control over all channel pairs
scripts/06_experiments.py lower  # adds the FT9/FT10 montage where available
scripts/06_experiments.py raw    # ablation: no per-recording normalisation
scripts/07_report.py             # summary tables
scripts/05_figures.py results    # Figs 2-5
```

### Signal processing

* 4th-order zero-phase Butterworth band-pass, 0.5–40 Hz.
* Non-overlapping 10 s windows; a window is **ictal** if it overlaps any
  annotated seizure by any amount.
* 14 features per channel: log absolute power and relative power in
  delta/theta/alpha/beta/low-gamma (Welch, 2 s segments, 50 % overlap), log RMS,
  log line length, normalised spectral entropy, and the log fast/slow power
  ratio. A 2-channel montage therefore has 28 features, the full montage 252.
* Features are computed **once for the union of all montages' channels**, so the
  montage design matrices are column slices of one array. Identical windows by
  construction.
* Recordings stored referentially (`chb12_27/28/29`, against CS2) are converted
  back to the bipolar derivations rather than discarded.
* Per-recording robust normalisation (median / IQR) of each feature removes
  amplitude and impedance offsets between recordings. This uses **no labels**,
  but it is non-causal — it models an offline analysis or a device with a long
  calibration baseline, not a strictly online detector. The `raw` experiment is
  the ablation without it.

### Classifier and validation

* Shrinkage LDA (`lsqr`, Ledoit-Wolf shrinkage) on standardised features.
* 5-fold `GroupKFold` over **patients**. chb01 and chb21 are the same person and
  are kept in one group, so folds are 23 people, not 24 cases.
* Decision thresholds are chosen on **out-of-fold scores from the training
  patients only** and applied unchanged to the held-out patients.

### Endpoints

Window-level F1 is not reported as a headline number. The endpoints are:

* **AUPRC** (threshold-free, the right summary at ~0.3 % positive prevalence).
* **Event-level seizure sensitivity** — a seizure counts as detected if an alarm
  overlaps `[onset − 30 s, offset + 60 s]`.
* **False alarms per hour**, measured over interictal time. An alarm needs 2
  consecutive windows above threshold, and a sustained alarm is re-issued every
  60 s — otherwise a detector stuck permanently in alarm would score one false
  alarm per recording and near-perfect sensitivity.
* **Detection latency** from annotated onset, clipped at −30 s.
* All of these at matched false-alarm budgets (1, 6 and 24 alarms/hour), because
  comparing untuned thresholds across configurations is not meaningful.
* Confidence intervals are percentile bootstraps **over patients**, not windows.

## Reproducing

```bash
pip install -r requirements.txt
```

```bash
python scripts/01_inventory.py && python scripts/02_download.py
```

```bash
python scripts/04_extract_and_prune.py --keep chb01
```

```bash
python scripts/06_experiments.py main && python scripts/07_report.py
```

The full database is ~43 GB of EDF. `04_extract_and_prune.py` deletes each
subject's recordings once its features are written, so peak disk use stays near
the size of the largest subject. Features for all 24 subjects are a few hundred
MB.

## Data and results layout

```
data/chbmit/chbXX/chbXX-summary.txt   annotations (committed; EDFs are not)
results/features/chbXX.npz            (n_windows, n_channels, 14) float32
results/tables/*.csv                  inventories, CV results, summaries
results/figures/*.png                 figures
```

## Caveats

* CHB-MIT is paediatric and from one centre; these are scalp proxies for
  wearables, so the numbers bound what the *scalp positions* carry, not what a
  real device would achieve.
* The task is **ictal vs non-ictal classification**, not seizure prediction.
* CHB-MIT carries no seizure-onset-localisation labels, so the "do around-ear
  contacts do better for temporal-lobe seizures?" question cannot be answered
  here. Left-vs-right 2-contact performance per patient is reported as the
  nearest available proxy; SeizeIT2 has the localisation metadata needed for the
  real analysis.
* SeizeIT2's public release contains the behind-the-ear wearable signals and
  clinical annotations, but **not** paired full-scalp EEG, so it cannot be used
  for a same-recording electrode ablation — only as external validation of
  whether real wearable hardware detects seizures at all.

## References

* Shoeb & Guttag. *Application of machine learning to epileptic seizure
  detection.* ICML 2010. (CHB-MIT)
* Bhagubai et al. *SeizeIT2: a multimodal wearable dataset of behind-the-ear
  EEG.* Scientific Data, 2025.
* Dan et al. *SzCORE: a seizure community open-source research evaluation
  framework.* 2024.
