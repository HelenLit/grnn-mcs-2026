from collections.abc import Sequence

import numpy as np

from src.models.grnn import GRNN


def fit_decomposed_grnns(
    train_X: Sequence[np.ndarray],
    train_y: Sequence[np.ndarray],
    n_models: int,
    sigma: float
) -> list[GRNN]:
    """Fit GRNN base models on already decomposed sub-sequences."""
    if len(train_X) != n_models or len(train_y) != n_models:
        raise ValueError(
            "Interleaved GRNN ensembles expect pre-decomposed training "
            f"sequences with length n_models={n_models}."
        )

    base_models = []
    for X_m, y_m in zip(train_X, train_y, strict=True):
        grnn_m = GRNN(sigma=sigma)
        grnn_m.fit(X_m, y_m)
        base_models.append(grnn_m)

    return base_models


def stack_base_predictions(base_models: list[GRNN], test_X: np.ndarray) -> np.ndarray:
    """Return base-model predictions as a matrix shaped (n_models, n_test)."""
    return np.vstack([model.predict(test_X) for model in base_models])
