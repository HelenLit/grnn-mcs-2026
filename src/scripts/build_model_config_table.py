"""Builds `Table 2. Model configurations`.

The table gives another researcher everything needed to repeat the
experiment: the full configuration of every method compared, GRNN and
official M4 benchmarks alike.

**The asymmetry is deliberate and is stated in the table itself.** MLP, ARIMA,
Naive2, Com, Smyl and FFORMA are never re-implemented here — their officially
submitted point forecasts are downloaded and scored with our own metric code.
Reproducing those rows means fetching the identical files, so they need enough
detail to *identify* the method and its provenance, not enough to re-estimate
it. Only the GRNN row needs full re-estimation detail. Saying that directly is
stronger than filling six cells with unverified defaults.

Where the organisers did not document something — ARIMA's information
criterion, maximum orders and drift — the cell says "not documented", because
`Appendix E` of the M4 paper is supplementary material absent from the PDF.
Those cells are never filled from memory.

The GRNN row is derived from `src/config/config.yaml` and the sigma-optimizer
defaults rather than typed out, so the table cannot drift away from the code
that produced the numbers.

Usage:
    python -m src.scripts.build_model_config_table
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import yaml

from src.models.sigma_gradient import select_sigma_gradient
from src.models.sigma_optimizers import select_sigma_brent
from src.models.sigma_optimizers import select_sigma_coarse2fine
from src.models.sigma_optimizers import select_sigma_silverman

CONFIG_PATH = Path("src/config/config.yaml")
OUTPUT_DIR = Path("reports/tables")
OUTPUT_STEM = "table_model_configs"

COLUMNS = (
    "Method",
    "Implementation / source",
    "Hyperparameters",
    "Selection procedure",
    "Preprocessing",
    "Forecasting scheme",
    "Provenance",
)


# One entry per value `evaluation.sigma_search` accepts. The description is
# looked up rather than hardcoded so that the identity of the procedure is
# guarded against drift the same way the numbers are: switching the optimizer
# in the config can never leave Table 2 describing a procedure other than the
# one that actually ran.
SIGMA_SEARCH_DESCRIPTIONS = {
    "grid": (
        "вичерпним перебором сітки з кроком {sigma_step}",
        None,
    ),
    "gradient": (
        "методом L-BFGS-B у логарифмічній параметризації з {n_starts} "
        "геометрично рознесених початкових точок з аналітичною похідною "
        "dMSE/dsigma",
        select_sigma_gradient,
    ),
    "brent": (
        "методом Брента з обмеженнями (золотий перетин із параболічною "
        "інтерполяцією) на {n_starts} логарифмічно рознесених брекетах, "
        "без використання похідних",
        select_sigma_brent,
    ),
    "coarse2fine": (
        "двоетапним пошуком: логарифмічна сітка, далі локальне уточнення",
        select_sigma_coarse2fine,
    ),
    "silverman": (
        "аналітичним правилом підстановки Сільвермана, без обчислень "
        "цільової функції",
        select_sigma_silverman,
    ),
}


def _sigma_procedure(config: dict) -> str:
    """Describes the sigma search actually configured, not a fixed one."""
    mode = config["evaluation"].get("sigma_search", "grid")
    if mode not in SIGMA_SEARCH_DESCRIPTIONS:
        raise SystemExit(
            f"Unknown evaluation.sigma_search '{mode}'. Add it to "
            "SIGMA_SEARCH_DESCRIPTIONS before generating the table — a wrong "
            "description here is worse than a missing one."
        )
    template, selector = SIGMA_SEARCH_DESCRIPTIONS[mode]
    fields = {"sigma_step": config["grnn"]["sigma_step"]}
    if selector is not None:
        signature = inspect.signature(selector)
        fields["n_starts"] = signature.parameters["n_starts"].default
    return template.format(**fields)


def _grnn_row(config: dict, label: str, aggregation: str) -> dict[str, str]:
    preparation = config["preparation"]
    evaluation = config["evaluation"]
    ensemble = config["ensemble_robust"]

    sigma_low = ensemble["sigma_min"]
    sigma_high = ensemble["sigma_max"]
    sigma_procedure = _sigma_procedure(config)
    validation_size = evaluation["validation_size"]
    m_values = ensemble["M_values"]

    return {
        "Method": label,
        "Implementation / source": (
            "This work — `src/models/grnn*.py`, Python 3.12, NumPy. "
            "Gaussian-kernel GRNN (Specht, 1991); no iterative training."
        ),
        "Hyperparameters": (
            f"Kernel width sigma in [{sigma_low}, {sigma_high}] (single free "
            f"parameter). Lag order L = {preparation['lag_order']}. "
            + (
                f"Ensemble size M in {{{', '.join(str(value) for value in m_values)}}}."
                if aggregation != "single"
                else "No ensemble."
            )
        ),
        "Selection procedure": (
            f"sigma \u2208 [{sigma_low}, {sigma_high}] визначали {sigma_procedure} "
            f"за критерієм мінімуму середньоквадратичної похибки на валідаційному "
            f"фрагменті. Валідаційним фрагментом були хронологічно ОСТАННІ "
            f"ceil({validation_size} * n) лагових прикладів навчальної частини; "
            f"тестовий горизонт у виборі параметрів не використовували. "
            f"Пошук sigma виконували В ТОМУ САМОМУ просторі ознак, у якому "
            f"працює навчена модель (MaxAbs), причому масштабування визначали "
            f"лише за fit-частиною, щоб валідаційні вікна не впливали на "
            f"перетворення, яке їх оцінює. "
            + (
                "Кількість базових мереж M обирали на тому самому фрагменті "
                "за критерієм RMSE."
                if aggregation != "single"
                else ""
            )
        ),
        "Preprocessing": (
            f"First differencing, then sliding window of {preparation['lag_order']} "
            f"raw lags, then {preparation['scaler']} fitted on the fit split only "
            "(no leakage into validation). No feature expansion of any kind — "
            "the scaled raw lags are the features. Reported with "
            "differencing both ON and OFF."
        ),
        "Forecasting scheme": (
            "Recursive one-step-ahead over h = 1..6, each prediction fed back "
            "into the lag window, with clipping to a robust range derived from "
            "the training target; the horizon is then integrated back to levels. "
            + {
                "single": "Single model.",
                "mean": "Deterministic interleaved decomposition into M "
                        "sub-sequences; members combined by the MEAN.",
                "median": "Deterministic interleaved decomposition into M "
                          "sub-sequences; members combined by the MEDIAN.",
                "bagging": "Bootstrap resampling of the lagged training set; "
                           "members combined by the mean over seeds.",
            }[aggregation]
        ),
        "Provenance": (
            "Re-estimated here on every series. Configuration and input SHA256s "
            "recorded in `reports/results/m4/manifest_main.json`."
        ),
    }


# Official M4 submissions. Every factual cell traces to the official
# M4-methods repository files; nothing here is recalled from memory.
OFFICIAL_ROWS = (
    {
        "Method": "MLP (M4 benchmark)",
        "Implementation / source": (
            "M4 organisers' `ML_benchmarks.py`, `mlp_bench`; "
            "scikit-learn `MLPRegressor` v0.19.1."
        ),
        "Hyperparameters": (
            "One hidden layer, 6 units; activation `identity` (linear); "
            "solver `adam`; learning rate `adaptive`, initial 0.001; "
            "max_iter = 100; early stopping NOT enabled; random_state = 42."
        ),
        "Selection procedure": (
            "None — no validation split and no hyperparameter search of any "
            "kind. The configuration above is fixed for every series."
        ),
        "Preprocessing": (
            "Multiplicative deseasonalisation (a no-op at Yearly frequency, "
            "m = 1), then LINEAR DETRENDING via a degree-1 least-squares fit; the "
            "forecast is produced in detrended space and the trend re-added. "
            "Three input lags. No scaling."
        ),
        "Forecasting scheme": (
            "Recursive one-step-ahead, prediction fed back into the lag "
            "window. Negatives clipped to 0; values above 1000 * max(ts) "
            "clipped to max(ts). Ensemble of 30 fits combined by the MEDIAN."
        ),
        "Provenance": (
            "Not re-implemented. Official submitted forecasts "
            "`submission-MLP.csv`, scored with our metric code."
        ),
    },
    {
        "Method": "ARIMA (M4 benchmark)",
        "Implementation / source": (
            "R `forecast` package v8.2 (Hyndman, 2017), automatic ARIMA — "
            "cited by the organisers as (auto)ARIMA (Hyndman & Khandakar, 2008)."
        ),
        "Hyperparameters": (
            "(p, d, q) chosen per series by the automatic procedure. "
            "Maximum orders NOT DOCUMENTED by the organisers; presence of "
            "drift NOT DOCUMENTED."
        ),
        "Selection procedure": (
            "Automatic stepwise order search, best model by information "
            "criterion. Which criterion (AIC / AICc / BIC) is NOT DOCUMENTED: "
            "the M4 paper's Appendix E, where the benchmark details live, is "
            "supplementary material not contained in the published PDF. No "
            "argument overrides are documented, so the package defaults apply "
            "— stated here as an inference, not as a quoted configuration."
        ),
        "Preprocessing": "As implemented inside `auto.arima`; none applied by us.",
        "Forecasting scheme": "Multi-step forecast over h = 1..6 from the fitted model.",
        "Provenance": (
            "Not re-implemented. Official submitted forecasts "
            "`submission-ARIMA.csv`, scored with our metric code."
        ),
    },
    {
        "Method": "Naive2",
        "Implementation / source": "M4 organisers' `Benchmarks and Evaluation.R`.",
        "Hyperparameters": "None.",
        "Selection procedure": "None.",
        "Preprocessing": (
            "Seasonal adjustment where a series is seasonal; at Yearly "
            "frequency (m = 1) this is inactive, so Naive, sNaive and Naive2 "
            "coincide (all three score 16.342 / 3.974)."
        ),
        "Forecasting scheme": "Repeats the last observation over the horizon.",
        "Provenance": (
            "Not re-implemented. Official `submission-Naive2.csv`. Also the OWA "
            "denominator, which is why its OWA is 1.000 by construction."
        ),
    },
    {
        "Method": "Com (Comb)",
        "Implementation / source": "M4 organisers' `Benchmarks and Evaluation.R`.",
        "Hyperparameters": "None beyond its three components.",
        "Selection procedure": "None — fixed equal-weight arithmetic mean.",
        "Preprocessing": "As in the organisers' benchmark script.",
        "Forecasting scheme": (
            "Simple average of SES, Holt and Damped exponential smoothing. "
            "The organisers' designated comparison benchmark (M4 paper §2)."
        ),
        "Provenance": "Not re-implemented. Official `submission-Com.csv`.",
    },
    {
        "Method": "118-Smyl (M4 winner)",
        "Implementation / source": (
            "Slawek Smyl, Uber Technologies. Hybrid ES-RNN; competition rank 1 "
            "on point forecasts."
        ),
        "Hyperparameters": (
            "Author's submission; not re-specified here. Trained GLOBALLY "
            "across the M4 collection, not per series."
        ),
        "Selection procedure": "Author's own; see the M4 methods description.",
        "Preprocessing": "Author's own.",
        "Forecasting scheme": "Author's own.",
        "Provenance": (
            "Not re-implemented. Official `submission-118.csv`. Reported "
            "computational cost: 8 056 min over all 100 000 series "
            "(M4 paper Table 7, different hardware and a wider scope than ours)."
        ),
    },
    {
        "Method": "245-FFORMA",
        "Implementation / source": (
            "Montero-Manso, Talagala, Hyndman & Athanasopoulos "
            "(A Coruña / Monash). Feature-based combination (statistical + ML); "
            "competition rank 2."
        ),
        "Hyperparameters": (
            "Authors' submission; not re-specified here. Meta-learner trained "
            "GLOBALLY across the M4 collection."
        ),
        "Selection procedure": "Authors' own.",
        "Preprocessing": "Authors' own.",
        "Forecasting scheme": "Authors' own.",
        "Provenance": (
            "Not re-implemented. Official `submission-245.csv`. Reported "
            "computational cost: 46 108 min over all 100 000 series "
            "(M4 paper Table 7)."
        ),
    },
)

PREAMBLE = """\
Reproducibility here is asymmetric **by design**, and the asymmetry is the point rather than a
gap. For MLP, ARIMA, Naive2, Com, 118-Smyl and 245-FFORMA nothing is re-implemented: the
officially submitted point forecasts are downloaded and scored with our own metric code, so
repeating those rows means fetching the identical files. They therefore need enough detail to
identify the method and its provenance, not enough to re-estimate it. Only the GRNN rows describe
a model estimated in this work, and those are given in full.

Cells marked NOT DOCUMENTED are not omissions on our side — the M4 paper's Appendix E, which
holds the benchmark details, is supplementary material absent from the published PDF. They are
left explicit rather than filled from package defaults.

Two consequences worth stating alongside the table:

1. MLP and GRNN both use three input lags but remove trend differently — MLP by linear
   detrending, this work by first differencing. That is the honest account of why two
   methods on the same lag structure behave so differently, and it is why the differencing
   ablation is reported throughout.
2. There are no published Yearly-per-domain reference values for any method, so the anchor gate
   that validates our metric code can only be applied at the Yearly-total level. This is a
   property of the published material, not a shortcut in the protocol.
"""


def build_table() -> pd.DataFrame:
    """Assembles the configuration table, GRNN rows derived from the config."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    rows = [
        _grnn_row(config, "GRNN (single)", "single"),
        _grnn_row(config, "GRNN-Mean", "mean"),
        _grnn_row(config, "GRNN-Median (this work)", "median"),
        _grnn_row(config, "GRNN-Bag", "bagging"),
        *OFFICIAL_ROWS,
    ]
    return pd.DataFrame(rows, columns=list(COLUMNS))


def write_markdown(table: pd.DataFrame, path: Path) -> None:
    """Writes the table as Markdown, one method per section.

    A seven-column grid is unreadable at page width, so each method becomes a
    small block. The CSV keeps the grid form for machine use.
    """
    lines = ["# Table 2. Model configurations", "", PREAMBLE, ""]
    for _, row in table.iterrows():
        lines.append(f"## {row['Method']}")
        lines.append("")
        for column in COLUMNS[1:]:
            lines.append(f"- **{column}.** {row[column]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    csv_path = OUTPUT_DIR / f"{OUTPUT_STEM}.csv"
    md_path = OUTPUT_DIR / f"{OUTPUT_STEM}.md"
    table.to_csv(csv_path, index=False)
    write_markdown(table, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
