from collections.abc import Sequence

import numpy as np

from src.models.grnn_ensemble_common import fit_decomposed_grnns
from src.models.grnn_ensemble_common import stack_base_predictions


class GRNNEnsembleAvg:
    """GRNN ensemble using structured decomposition and mean aggregation."""

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
        """Predicts using the mean of all base models' predictions."""
        if not self.base_models_:
            raise ValueError("Ensemble has not been fitted yet.")

        predictions_matrix = stack_base_predictions(self.base_models_, test_X)
        return np.mean(predictions_matrix, axis=0)
