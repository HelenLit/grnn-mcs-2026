import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preparation import PreparedDataset
from src.data.preparation import decompose_into_subsequences
from src.data.preparation import prepare_raw
from src.data.preparation import prepare_synthetic
from src.evaluation.metrics import compute_metrics
from src.models.baselines import ARIMAWrapper
from src.models.baselines import MLPWrapper
from src.models.baselines import SVRWrapper
from src.models.baselines import select_arima_order
from src.models.baselines import select_mlp_hyperparams
from src.models.baselines import select_svr_hyperparams
from src.models.grnn import GRNN
from src.models.grnn import select_sigma
from src.models.grnn_bagging import GRNNBagging
from src.models.grnn_ensemble_avg import GRNNEnsembleAvg
from src.models.grnn_ensemble_robust import GRNNRobustEnsemble
from src.models.grnn_ensemble_robust import select_sigma_ensemble
from src.models.sigma_gradient import select_sigma_gradient
from src.models.sigma_optimizers import NEW_METHODS


DATASET_KEYS = (
    "sinusoid_noise",
    "sinusoid_anomalies",
    "trend_seasonality",
    "logistic_map",
    "mackey_glass",
    "obd",
    "body_fat",
)

MODEL_KEYS = (
    "single_grnn",
    "grnn_robust",
    "grnn_avg",
    "grnn_bagging",
    "svr",
    "mlp",
    "arima",
)

INTERLEAVED_MODEL_KEYS = ("grnn_robust", "grnn_avg")
ENSEMBLE_MODEL_KEYS = ("grnn_robust", "grnn_avg", "grnn_bagging")
BASELINE_MODEL_KEYS = ("svr", "mlp", "arima")
METRIC_COLUMNS = ("MAE", "RMSE", "MedAE", "MAPE", "R2")
RESULT_COLUMNS = ("series", "model", "n_models", "sigma", *METRIC_COLUMNS)


def load_all_datasets(
    paths: dict[str, str],
    prep_config: dict[str, Any]
) -> dict[str, PreparedDataset]:
    """Load all synthetic and raw benchmark datasets."""
    data_synthetic_dir = Path(paths["DATA_SYNTHETIC_DIR"])
    scaler_type = prep_config["scaler"]

    datasets = {}
    for dataset_key in DATASET_KEYS[:5]:
        csv_path = data_synthetic_dir / dataset_key / f"{dataset_key}.csv"
        datasets[dataset_key] = prepare_synthetic(
            filepath=str(csv_path),
            lag_order=prep_config["lag_order"],
            test_size=prep_config["test_size"],
            scaler_type=scaler_type,
        )

    datasets["obd"] = prepare_raw(
        train_path=paths["OBD_TRAIN"],
        test_path=paths["OBD_TEST"],
        scaler_type=scaler_type,
    )
    datasets["body_fat"] = prepare_raw(
        train_path=paths["BODY_FAT_TRAIN"],
        test_path=paths["BODY_FAT_TEST"],
        scaler_type=scaler_type,
    )

    return datasets


def sigma_config_key(model_key: str) -> str:
    """Maps a registry model key to its `config.yaml` hyperparameter section."""
    return {
        "single_grnn": "grnn",
        "grnn_robust": "ensemble_robust",
        "grnn_avg": "ensemble_avg",
        "grnn_bagging": "ensemble_bagging",
    }[model_key]


def build_sigma_grid(model_config: dict[str, float]) -> np.ndarray:
    """Build a sigma grid from min/max/step config values."""
    sigma_min = model_config["sigma_min"]
    sigma_max = model_config["sigma_max"]
    sigma_step = model_config["sigma_step"]
    return np.arange(sigma_min, sigma_max + sigma_step / 2.0, sigma_step)


def make_model(
    model_key: str,
    n_models: int | float,
    sigma: float,
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> object:
    """Construct a model by registry key.

    Args:
        model_key: Registry key (see MODEL_KEYS).
        n_models: Ensemble size (ignored for non-ensemble models).
        sigma: GRNN smoothing parameter (ignored for baselines).
        config: Full loaded config.yaml.
        overrides: For SVR/MLP/ARIMA only — validation-tuned hyperparameters
            from `select_baseline_hyperparams`, taking precedence over
            the fixed `config.yaml` defaults. None (default) uses the fixed
            defaults unchanged — existing callers (e.g. ensemble validation
            RMSE, which never tunes baselines) are unaffected.
    """
    if model_key == "single_grnn":
        return GRNN(sigma=sigma)
    if model_key == "grnn_robust":
        return GRNNRobustEnsemble(n_models=int(n_models), sigma=sigma)
    if model_key == "grnn_avg":
        return GRNNEnsembleAvg(n_models=int(n_models), sigma=sigma)
    if model_key == "grnn_bagging":
        return GRNNBagging(
            n_models=int(n_models),
            sigma=sigma,
            random_seed=config["ensemble_bagging"]["random_seed"],
        )
    if model_key == "svr":
        svr_config = config["baselines"]["svr"]
        params = overrides or {}
        return SVRWrapper(
            kernel=svr_config["kernel"],
            C=params.get("C", svr_config["C"]),
            epsilon=params.get("epsilon", svr_config["epsilon"]),
            gamma=params.get("gamma", svr_config["gamma"]),
        )
    if model_key == "mlp":
        mlp_config = config["baselines"]["mlp"]
        params = overrides or {}
        return MLPWrapper(
            hidden_layer_sizes=params.get(
                "hidden_layer_sizes", mlp_config["hidden_layer_sizes"]
            ),
            max_iter=mlp_config["max_iter"],
            random_state=mlp_config["random_state"],
            alpha=params.get("alpha", mlp_config["alpha"]),
            early_stopping=mlp_config["early_stopping"],
        )
    if model_key == "arima":
        arima_config = config["baselines"]["arima"]
        params = overrides or {}
        return ARIMAWrapper(order=tuple(params.get("order", arima_config["order"])))

    raise ValueError(f"Unknown model_key '{model_key}'.")


def select_baseline_hyperparams(
    model_key: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    config: dict[str, Any],
    validation_size: float,
) -> dict[str, Any]:
    """Selects tuned hyperparameters for a baseline model.

    Mirrors `select_sigma_dispatch` for the GRNN family: every baseline gets
    validation-based tuning on the same held-out split used for sigma, so no
    model family is compared against a fixed-default competitor.

    Args:
        model_key: One of "svr", "mlp", "arima".
        train_X: Training feature matrix (unused by ARIMA).
        train_y: Training target array.
        config: Full loaded config.yaml (reads `baselines.<model>.tuning`).
        validation_size: Fraction held out for SVR/MLP tuning (ARIMA's AIC
            selection needs no held-out split).

    Returns:
        A dict of tuned hyperparameters, passed as `overrides` to `make_model`.
    """
    if model_key == "svr":
        tuning = config["baselines"]["svr"]["tuning"]
        C, gamma, epsilon = select_svr_hyperparams(
            train_X, train_y,
            C_values=tuning["C_values"],
            gamma_values=tuning["gamma_values"],
            epsilon_values=tuning["epsilon_values"],
            validation_size=validation_size,
        )
        return {"C": C, "gamma": gamma, "epsilon": epsilon}

    if model_key == "mlp":
        mlp_config = config["baselines"]["mlp"]
        tuning = mlp_config["tuning"]
        hidden_layer_sizes, alpha = select_mlp_hyperparams(
            train_X, train_y,
            hidden_layer_sizes_grid=tuning["hidden_layer_sizes_grid"],
            alpha_values=tuning["alpha_values"],
            max_iter=mlp_config["max_iter"],
            random_state=mlp_config["random_state"],
            early_stopping=mlp_config["early_stopping"],
            validation_size=validation_size,
        )
        return {"hidden_layer_sizes": hidden_layer_sizes, "alpha": alpha}

    if model_key == "arima":
        tuning = config["baselines"]["arima"]["tuning"]
        order = select_arima_order(
            train_y, order_grid=[tuple(order) for order in tuning["order_grid"]]
        )
        return {"order": order}

    raise ValueError(f"No baseline tuning configured for '{model_key}'.")


def model_needs_sigma(model_key: str) -> bool:
    """Return whether a model uses a GRNN sigma hyperparameter."""
    return model_key in {
        "single_grnn",
        "grnn_robust",
        "grnn_avg",
        "grnn_bagging",
    }


def select_sigma_dispatch(
    model_key: str,
    n_models: int | float,
    train_X: np.ndarray,
    train_y: np.ndarray,
    sigma_values: np.ndarray,
    config: dict[str, Any]
) -> float:
    """Select sigma for GRNN-family models; return NaN for fixed baselines.

    `evaluation.sigma_search` chooses the mode:
      'grid'        exhaustive search over `sigma_values`
      'gradient'    multi-start L-BFGS-B with the analytic dMSE/dsigma
      'brent'       multi-start bounded Brent, derivative-free
      'coarse2fine' two-stage log grid then local refinement
      'silverman'   closed-form plug-in rule, no objective evaluations

    For the last three, `sigma_values` supplies only the [min, max] bounds.
    All five were compared on the full 23,000 series in the deployment
    feature space (`compare_sigma_optimizers`): brent was most accurate in
    most cells and cheaper than grid, gradient was the slowest for the
    median ensemble without winning on accuracy.
    """
    if not model_needs_sigma(model_key):
        return float("nan")

    validation_size = config["evaluation"]["validation_size"]
    mode = config["evaluation"].get("sigma_search", "grid")

    if mode in NEW_METHODS:
        aggregate = {"grnn_robust": "median", "grnn_avg": "mean"}.get(
            model_key, "single"
        )
        result = NEW_METHODS[mode](
            train_X=train_X,
            train_y=train_y,
            n_models=int(n_models) if aggregate != "single" else 1,
            aggregate=aggregate,
            sigma_bounds=(float(sigma_values[0]), float(sigma_values[-1])),
            validation_size=validation_size,
        )
        return float(result.sigma)

    if mode == "gradient":
        aggregate = {"grnn_robust": "median", "grnn_avg": "mean"}.get(
            model_key, "single"
        )
        return select_sigma_gradient(
            train_X=train_X,
            train_y=train_y,
            n_models=int(n_models) if aggregate != "single" else 1,
            aggregate=aggregate,
            sigma_bounds=(float(sigma_values[0]), float(sigma_values[-1])),
            validation_size=validation_size,
        )

    if model_key == "single_grnn" or model_key == "grnn_bagging":
        return select_sigma(
            train_X=train_X,
            train_y=train_y,
            sigma_values=sigma_values,
            validation_size=validation_size,
        )

    if model_key in INTERLEAVED_MODEL_KEYS:
        return select_sigma_ensemble(
            train_X=train_X,
            train_y=train_y,
            n_models=int(n_models),
            sigma_values=sigma_values,
            validation_size=validation_size,
        )

    raise ValueError(f"Sigma selection is not configured for '{model_key}'.")


def compute_validation_rmse(
    model_key: str,
    n_models: int,
    sigma: float,
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_size: float,
    config: dict[str, Any]
) -> float:
    """Evaluate one fully-specified ensemble config on the held-out validation
    split used for sigma selection. Used only to choose n_models for
    `comparison_best.csv` — never touches the test set."""
    n_samples = len(train_y)
    n_val = math.ceil(n_samples * validation_size)
    n_train_split = n_samples - n_val

    val_train_X = train_X[:n_train_split]
    val_train_y = train_y[:n_train_split]
    val_X = train_X[n_train_split:]
    val_y = train_y[n_train_split:]

    model = make_model(model_key, n_models, sigma, config)
    fit_model(model, val_train_X, val_train_y)
    y_pred = model.predict(val_X)
    return compute_metrics(val_y, y_pred)["RMSE"]


def fit_model(model: object, train_X: np.ndarray, train_y: np.ndarray) -> None:
    """Fit a model, decomposing training data first if the model requires it."""
    if getattr(model, "requires_decomposed_training", False):
        subsequences = decompose_into_subsequences(
            train_X,
            train_y,
            model.n_models,
        )
        train_X = [X_m for X_m, _ in subsequences]
        train_y = [y_m for _, y_m in subsequences]

    model.fit(train_X, train_y)


def predict_model(prepared: PreparedDataset, model: object) -> np.ndarray:
    """Fit a model and return predictions for the prepared test set."""
    fit_model(model, prepared.train_X, prepared.train_y)
    return model.predict(prepared.test_X)


def evaluate_model(prepared: PreparedDataset, model: object) -> dict[str, float]:
    """Fit a model, predict the prepared test set, and compute metrics."""
    y_pred = predict_model(prepared, model)
    return compute_metrics(prepared.test_y, y_pred)


def results_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble result rows using the canonical result schema."""
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def iter_model_configs(config: dict[str, Any]) -> list[tuple[str, int | float]]:
    """Return the applicability grid for one dataset: every model_key,
    with every configured M for ensembles (single_grnn/baselines get one
    (model_key, NaN) entry each)."""
    model_configs = [("single_grnn", 1)]

    for model_key, config_key in [
        ("grnn_robust", "ensemble_robust"),
        ("grnn_avg", "ensemble_avg"),
        ("grnn_bagging", "ensemble_bagging"),
    ]:
        for n_models in config[config_key]["M_values"]:
            model_configs.append((model_key, n_models))

    model_configs.extend([
        ("svr", float("nan")),
        ("mlp", float("nan")),
        ("arima", float("nan")),
    ])
    return model_configs


def select_best_configs(
    all_results: pd.DataFrame,
    validation_rmse: dict[tuple[str, str, int], float],
    dataset_keys: list[str],
) -> pd.DataFrame:
    """Keep one row per dataset x model. For ensembles, `n_models` is chosen
    by validation RMSE (from `validation_rmse`), never by the test-set RMSE
    reported in `all_results` — avoids selecting on the metric being reported.
    Non-ensemble models have exactly one row already."""
    best_rows = []
    for dataset_key in dataset_keys:
        for model_key in MODEL_KEYS:
            group = all_results[
                (all_results["series"] == dataset_key)
                & (all_results["model"] == model_key)
            ]
            if model_key in ENSEMBLE_MODEL_KEYS:
                best_n_models = min(
                    group["n_models"].tolist(),
                    key=lambda n: validation_rmse[(dataset_key, model_key, int(n))],
                )
                row = group[group["n_models"] == best_n_models]
            else:
                row = group
            best_rows.append(row.iloc[0].to_dict())

    return results_frame(best_rows)


def run_grid(
    datasets: dict[str, PreparedDataset],
    model_configs: list[tuple[str, int | float]],
    config: dict[str, Any],
    print_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the full model_configs grid over the given datasets and selects
    the best per-(dataset, model) row via validation RMSE.

    This is the shared core of the comparison pipeline: a single-seed driver
    and a multi-seed/CV driver can both call this unchanged, so a
    single "repeat" (one seed's synthetic realization, or one CV fold) is
    evaluated identically either way — no forked model-selection logic.

    Args:
        datasets: Mapping of dataset_key -> PreparedDataset to evaluate.
        model_configs: Output of `iter_model_configs`.
        config: Full loaded config.yaml.
        print_progress: If True, prints one line per (dataset, model_config)
            cell — appropriate for a single full-grid run, not for a
            multi-repeat driver that would print thousands of lines.

    Returns:
        (all_results, best_results). `all_results` has one row per
        (dataset, model_config) cell (every swept ensemble M + every
        baseline); `best_results` keeps one row per (dataset, model).
    """
    rows = []
    validation_rmse: dict[tuple[str, str, int], float] = {}
    dataset_keys = list(datasets.keys())
    total_cells = len(dataset_keys) * len(model_configs)
    completed_cells = 0

    for dataset_key in dataset_keys:
        prepared = datasets[dataset_key]
        for model_key, n_models in model_configs:
            completed_cells += 1

            if model_needs_sigma(model_key):
                sigma_values = build_sigma_grid(config[sigma_config_key(model_key)])
                sigma = select_sigma_dispatch(
                    model_key=model_key,
                    n_models=n_models,
                    train_X=prepared.train_X,
                    train_y=prepared.train_y,
                    sigma_values=sigma_values,
                    config=config,
                )
            else:
                sigma = float("nan")

            if model_key in ENSEMBLE_MODEL_KEYS:
                validation_rmse[(dataset_key, model_key, int(n_models))] = (
                    compute_validation_rmse(
                        model_key=model_key,
                        n_models=int(n_models),
                        sigma=sigma,
                        train_X=prepared.train_X,
                        train_y=prepared.train_y,
                        validation_size=config["evaluation"]["validation_size"],
                        config=config,
                    )
                )

            if model_key in BASELINE_MODEL_KEYS:
                overrides = select_baseline_hyperparams(
                    model_key=model_key,
                    train_X=prepared.train_X,
                    train_y=prepared.train_y,
                    config=config,
                    validation_size=config["evaluation"]["validation_size"],
                )
            else:
                overrides = None

            model = make_model(
                model_key=model_key,
                n_models=n_models,
                sigma=sigma,
                config=config,
                overrides=overrides,
            )
            metrics = evaluate_model(prepared, model)
            row = {
                "series": dataset_key,
                "model": model_key,
                "n_models": n_models,
                "sigma": sigma,
                **metrics,
            }
            rows.append(row)
            if print_progress:
                tuned_note = f", tuned={overrides}" if model_key in BASELINE_MODEL_KEYS else ""
                print(
                    f"[{completed_cells:02d}/{total_cells}] "
                    f"{dataset_key}, {model_key}, n_models={n_models}, "
                    f"sigma={sigma}, RMSE={metrics['RMSE']:.6f}{tuned_note}"
                )

    all_results = results_frame(rows)
    best_results = select_best_configs(all_results, validation_rmse, dataset_keys)
    return all_results, best_results
