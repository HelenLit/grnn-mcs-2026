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

## Prerequisites & Environment Setup

### 1. System Dependencies
- **Python 3.12** is recommended.
- A RAR extractor is required to unpack the official M4 forecast submissions (`unar`, `unrar`, `7z`, or `bsdtar`):
  - **macOS**: `brew install unar`
  - **Ubuntu / Debian**: `sudo apt-get update && sudo apt-get install -y unar`
- Disk space: ~250 MB for downloaded data (M4 Yearly dataset ~30 MB, official competition submissions ~214 MB).

### 2. Environment Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/HelenLit/grnn-mcs-2026.git
cd grnn-mcs-2026
```

Choose one of the following environment options:

#### Option A: Using Conda (e.g. `grnn_env`)
```bash
conda create -n grnn_env python=3.12 -y
conda activate grnn_env
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Using standard Python `venv`
```bash
python3.12 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

*(Note: The automated `./reproduce.sh` script will automatically create and use `.venv` if run directly without an active environment).*

---

## Reproducibility Pipeline

The benchmark can be reproduced either via the automated script or step-by-step manually. The pipeline is **idempotent**: re-running skips stages whose output files already exist.

### Method 1: Automated Reproduction (`reproduce.sh`)

- **Quick validation & anchor gate check** (~minutes):
  Downloads inputs, validates the 23,000-series dataset, and verifies the metric calculation against published M4 competition scores:
  ```bash
  ./reproduce.sh --quick
  ```

- **Full benchmark reproduction** (~1–1.5 hours):
  Runs dataset validation, anchor checks, the full GRNN family across all 23,000 series (with and without first differencing), comparison tables, stratified evaluations (horizon, length, anomaly level), statistical tests, robustness analysis, and plots:
  ```bash
  ./reproduce.sh
  ```

- **Full reproduction + Sensitivity Ablation Grid** (several hours):
  Executes the complete benchmark plus the $L \times M$ (lag order $\times$ ensemble size) ablation grid:
  ```bash
  ./reproduce.sh --with-ablation
  ```

### Method 2: Step-by-Step Manual Execution

If you prefer executing each stage of the research pipeline individually:

1. **Download raw M4 Yearly dataset**:
   ```bash
   mkdir -p data/raw/m4/Train data/raw/m4/Test
   curl -fL --retry 3 -o data/raw/m4/Train/Yearly-train.csv https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Yearly-train.csv
   curl -fL --retry 3 -o data/raw/m4/Test/Yearly-test.csv   https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Yearly-test.csv
   curl -fL --retry 3 -o data/raw/m4/M4-info.csv            https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/M4-info.csv
   ```

2. **Sync official submissions, validate dataset integrity & check anchor gate**:
   ```bash
   python -m src.scripts.run_m4_pipeline --tag main --stages sync,validate,anchors
   ```

3. **Train and evaluate the GRNN family on all 23,000 series**:
   ```bash
   python -m src.scripts.run_m4_pipeline --tag main --stages grnn,table,strat,plots --full --differencing both
   ```

4. **Run robustness analysis and generate derived benchmark reports**:
   ```bash
   python -m src.scripts.run_robustness_analysis --tag main
   python -m src.scripts.build_length_table main
   python -m src.scripts.build_model_config_table
   python -m src.scripts.build_benchmark_report --tag main
   ```

5. *(Optional)* **Run $L \times M$ sensitivity grid ablation**:
   ```bash
   python -m src.scripts.run_lag_m_ablation --tag ablation --scope extended --differencing both
   ```

Outputs are saved under `reports/final/`, `reports/tables/`, and `plots/m4/`.

---

## Interactive demo UI

A local Streamlit application for running the GRNN ensembles interactively —
on the built-in M4 Yearly data or on an uploaded CSV series — with EDA
statistics, forecast plots, quality metrics (sMAPE, MASE, OWA, MAE, RMSE) and
a downloadable results package:

```bash
streamlit run src/ui/app.py
```

The UI reuses the pipeline code unchanged; it computes nothing of its own. It
expects the M4 Yearly files under `data/raw/m4/` (offered for download on
first launch if missing; `./reproduce.sh --quick` also fetches them). An
example upload file ships at `data/synthetic/sinusoid_noise/sinusoid_noise.csv`.
UI language is Ukrainian.

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
| GRNN-Mean | 0.807 | 13.99 | 3.01 |
| GRNN-Median | 0.806 | 13.94 | 3.02 |
| GRNN-Bag | 0.819 | 14.24 | 3.05 |

Without differencing the family degrades to OWA 1.27–1.62 — the differencing
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
src/ui/                          interactive Streamlit demo (app.py entry)
.streamlit/config.toml           UI theme
data/synthetic/                  example series for the UI upload button
reports/, plots/                 generated outputs (not tracked)
```

## License

All rights reserved. Provided for peer-review purposes only; not for
redistribution.
