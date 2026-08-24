import math
from collections.abc import Sequence

import numpy as np

from src.data.preparation import decompose_into_subsequences
from src.models.grnn_ensemble_common import fit_decomposed_grnns
from src.models.grnn_ensemble_common import stack_base_predictions


class GRNNRobustEnsemble:
    """GRNN Ensemble with robust median aggregation.

    The developed method: trains an ensemble of M base GRNNs
    on structured interleaved sub-sequences of the training data, and aggregates
    their predictions using the median to provide robustness against anomalies.
    """

    def __init__(self, n_models: int, sigma: float) -> None:
        self.n_models = n_models
        self.sigma = sigma
        self.base_models_ = []
        self.requires_decomposed_training = True

    def fit(
        self,
        train_X: Sequence[np.ndarray],
        train_y: Sequence[np.ndarray]
    ) -> None:
        """Fits M base GRNN models from already decomposed sub-sequences."""
        self.base_models_ = fit_decomposed_grnns(
            train_X=train_X,
            train_y=train_y,
            n_models=self.n_models,
            sigma=self.sigma
        )

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """Predicts using the median of all base models' predictions."""
        if not self.base_models_:
            raise ValueError("Ensemble has not been fitted yet.")

        predictions_matrix = stack_base_predictions(self.base_models_, test_X)
        return np.median(predictions_matrix, axis=0)


def select_sigma_ensemble(
    train_X: np.ndarray,
    train_y: np.ndarray,
    n_models: int,
    sigma_values: np.ndarray,
    validation_size: float = 0.2
) -> float:
    """Selects the sigma minimising RMSE for the full ensemble on a validation split.

    Uses temporal chronological split (no shuffling) identical to single GRNN,
    but evaluates the full GRNNRobustEnsemble (incorporating interleaving + median).
    """
    n_samples = len(train_y)
    n_val = math.ceil(n_samples * validation_size)
    n_train_split = n_samples - n_val

    val_train_X = train_X[:n_train_split]
    val_train_y = train_y[:n_train_split]
    val_X = train_X[n_train_split:]
    val_y = train_y[n_train_split:]

    best_sigma = sigma_values[0]
    best_rmse = float('inf')

    for sigma in sigma_values:
        subsequences = decompose_into_subsequences(
            val_train_X,
            val_train_y,
            n_models
        )
        train_X_sequences = [X_m for X_m, _ in subsequences]
        train_y_sequences = [y_m for _, y_m in subsequences]
        ensemble = GRNNRobustEnsemble(n_models=n_models, sigma=sigma)
        ensemble.fit(train_X_sequences, train_y_sequences)
        y_pred = ensemble.predict(val_X)

        rmse = np.sqrt(np.mean((y_pred - val_y) ** 2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_sigma = sigma

    return float(best_sigma)
