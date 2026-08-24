# GRNN ensembles on the M4 Yearly benchmark

This repository contains the complete code behind a benchmark study of
generalized regression neural networks (GRNN) for forecasting short time
series. The method under study is a GRNN ensemble built by deterministic
interleaved decomposition of the series into sub-sequences, combined by the
median; a single GRNN, a mean-combined ensemble and a bagged ensemble are run
alongside it. All variants are evaluated on the official M4 Competition
Yearly subset — 23,000 real series with a median training length of 29
observations and a fixed horizon of 6.

Competing methods are never re-implemented. The officially submitted point
forecasts of the M4 winner (Smyl), runner-up (FFORMA), the Comb benchmark,
ARIMA, MLP and Naive2 are downloaded from the M4 organisers' repository and
scored with the same metric code as the GRNN runs. The metric code itself is
gate-checked: the pipeline recomputes each submission's OWA and stops if the
result does not match the published competition values.

## Requirements

- Python 3.12 (a virtualenv is created automatically; dependencies are pinned
  in `requirements.txt`)
- ~250 MB of downloads: the M4 Yearly dataset (~30 MB) and six official
  forecast submissions (~214 MB)
- A rar extractor for the submission archives (`unar`, `unrar`, `7z`, or
  `bsdtar`; on macOS `brew install unar`)

## How to reproduce

```
./reproduce.sh --quick
```

Sets up the environment, downloads all inputs, validates the 23,000-series
dataset, and runs the anchor gate — the recomputed OWA of every official
submission is checked against the published competition values. Takes minutes
(plus download time) and proves the evaluation code end to end.

```
./reproduce.sh
```

Everything above, then the full GRNN family on all 23,000 series (both with
and without first differencing), the comparison tables, stratified analyses
(horizon, length, anomaly level), significance tests, robustness analysis and
plots. About 1–1.5 hours after the downloads.

```
./reproduce.sh --with-ablation
```

Adds the L×M sensitivity grid (lag order × ensemble size). This is the long
tail — several hours.

The script is idempotent: re-running it skips every step whose output already
exists.

## Expected results

The anchor gate (recomputed OWA of the official submissions vs the published
values; matches to the 4th decimal):

| Method | Published OWA |
|---|---|
| Smyl (winner, submission 118) | 0.778 |
| FFORMA (runner-up, submission 245) | 0.799 |
| Comb (competition benchmark) | 0.867 |
| ARIMA | 0.892 |
| MLP | 1.288 |
| Naive2 (OWA denominator) | 1.000 |

The GRNN family, all 23,000 Yearly series, with first differencing
(`reports/final/m4_domain_comparison*_main.csv`):

| Model | OWA | sMAPE | MASE |
|---|---|---|---|
| GRNN (single) | 0.824 | 14.35 | 3.06 |
| GRNN-Mean | 0.816 | 14.17 | 3.05 |
| GRNN-Median | 0.822 | 14.29 | 3.06 |
| GRNN-Bag | 0.819 | 14.24 | 3.05 |

Without differencing the family degrades to OWA 1.27–1.39 — the differencing
variant is reported by the pipeline in the same tables.

## Repository layout

```
reproduce.sh                     one-command reproduction (see above)
requirements.txt                 pinned dependencies (Python 3.12)
data/anchors/                    published M4 OWA anchor values (input)
data/raw/                        downloaded M4 data (created by reproduce.sh)
src/config/config.yaml           all hyperparameters
src/data/                        M4 loaders, windowing, differencing, scaling
src/models/                      GRNN, ensembles, sigma-selection procedures
src/evaluation/                  metrics, harness, stratification, plotting
src/scripts/run_m4_pipeline.py   main pipeline (sync/validate/anchors/grnn/…)
src/scripts/run_lag_m_ablation.py       L×M sensitivity grid
src/scripts/run_robustness_analysis.py  large-error (exceedance) analysis
src/scripts/compare_sigma_optimizers.py sigma-optimizer comparison
src/scripts/build_*.py                  derived tables and the final report
reports/, plots/                 generated outputs (not tracked)
```

## License

All rights reserved. Provided for peer-review purposes only; not for
redistribution.
