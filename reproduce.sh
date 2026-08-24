#!/usr/bin/env bash
# Reproduces the results of the article from scratch.
#
#   ./reproduce.sh                   full reproduction (~1-1.5 h after download)
#   ./reproduce.sh --quick           environment + data + dataset validation +
#                                    anchor gate only (minutes; proves the
#                                    metric code against published M4 values)
#   ./reproduce.sh --with-ablation   full reproduction + the L x M sensitivity
#                                    grid (adds several hours)
#
# Idempotent: every step is skipped when its output already exists, so the
# script can be re-run safely after an interruption.
set -euo pipefail
cd "$(dirname "$0")"

QUICK=0
WITH_ABLATION=0
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --with-ablation) WITH_ABLATION=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# ── 1. Environment (Python 3.12 venv) ────────────────────────────────────────
if [ ! -x .venv/bin/python ]; then
    PY=""
    for cand in python3.12 python3; do
        if command -v "$cand" >/dev/null; then PY="$cand"; break; fi
    done
    [ -n "$PY" ] || { echo "python3 not found on PATH" >&2; exit 1; }
    "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' \
        || echo "[reproduce] WARNING: results were produced with Python 3.12," \
                "found $("$PY" --version)"
    echo "[reproduce] creating virtualenv .venv"
    "$PY" -m venv .venv
fi
PYTHON=.venv/bin/python
echo "[reproduce] installing requirements"
"$PYTHON" -m pip install -q --upgrade pip
"$PYTHON" -m pip install -q -r requirements.txt

# A rar extractor is needed to unpack the official-forecast archives.
if ! command -v unar >/dev/null && ! command -v unrar >/dev/null \
   && ! command -v 7z >/dev/null && ! command -v 7zz >/dev/null \
   && ! command -v bsdtar >/dev/null; then
    echo "no rar extractor found — install one first, e.g.:" >&2
    echo "  macOS: brew install unar     Linux: apt-get install unar" >&2
    exit 1
fi

# ── 2. Raw M4 Yearly dataset (immutable input, ~30 MB) ───────────────────────
RAW=data/raw/m4
M4_SRC="https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
mkdir -p "$RAW/Train" "$RAW/Test"
fetch() {  # fetch <relative-source-path> <destination>
    if [ -s "$2" ]; then
        echo "[reproduce] $2 already present — skipping"
    else
        echo "[reproduce] downloading $2"
        curl -fL --retry 3 -o "$2" "$M4_SRC/$1"
    fi
}
fetch "Train/Yearly-train.csv" "$RAW/Train/Yearly-train.csv"
fetch "Test/Yearly-test.csv"   "$RAW/Test/Yearly-test.csv"
fetch "M4-info.csv"            "$RAW/M4-info.csv"

# ── 3. Official forecasts + dataset validation + anchor gate ─────────────────
# sync: downloads/extracts the six required official submissions (~214 MB)
# validate: asserts the 23,000-series Yearly dataset integrity
# anchors: re-scores the official submissions with our metric code and
#          gate-checks the result against the published OWA values
#          (Smyl 0.7780, FFORMA 0.7989, Com 0.8670, ARIMA 0.8920,
#           MLP 1.2882, Naive2 1.0000)
SCORES=reports/results/m4/submission_scores_full_main.csv
if [ -s "$SCORES" ]; then
    echo "[reproduce] $SCORES already present — skipping sync/validate/anchors"
else
    "$PYTHON" -m src.scripts.run_m4_pipeline \
        --tag main --stages sync,validate,anchors
fi

if [ "$QUICK" -eq 1 ]; then
    echo "[reproduce] --quick done: dataset validated, anchor gate passed."
    exit 0
fi

# ── 4. GRNN family on all 23,000 series + tables + plots (~1 h) ──────────────
RESULTS=reports/results/m4/grnn_results_main.csv
if [ -s "$RESULTS" ] && [ -s reports/final/m4_domain_comparison_main.csv ]; then
    echo "[reproduce] $RESULTS already present — skipping the GRNN run"
else
    "$PYTHON" -m src.scripts.run_m4_pipeline \
        --tag main --stages grnn,table,strat,plots --full --differencing both
fi

# ── 5. Derived analyses and report ────────────────────────────────────────────
"$PYTHON" -m src.scripts.run_robustness_analysis --tag main
"$PYTHON" -m src.scripts.build_length_table main
"$PYTHON" -m src.scripts.build_model_config_table
"$PYTHON" -m src.scripts.build_benchmark_report --tag main

# ── 6. Optional L x M sensitivity grid ────────────────────────────────────────
if [ "$WITH_ABLATION" -eq 1 ]; then
    "$PYTHON" -m src.scripts.run_lag_m_ablation \
        --tag ablation --scope extended --differencing both
fi

echo "[reproduce] done. Tables: reports/final/ and reports/tables/. Plots: plots/m4/."
