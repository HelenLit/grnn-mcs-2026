import numpy as np

from src.models.grnn import GRNN
from src.models.grnn_ensemble_common import stack_base_predictions


class GRNNBagging:
    """Bootstrap GRNN ensemble with median aggregation."""

    def __init__(self, n_models: int, sigma: float, random_seed: int) -> None:
        self.n_models = n_models
        self.sigma = sigma
        self.random_seed = random_seed
        self.base_models_ = []
        self.bootstrap_indices_ = []

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        """Fits each base GRNN on a bootstrap sample of the training data."""
        n_samples = len(train_X)
        self.base_models_ = []
        self.bootstrap_indices_ = []

        for m in range(self.n_models):
            rng = np.random.default_rng(self.random_seed + m)
            sample_indices = rng.integers(0, n_samples, size=n_samples)
            grnn_m = GRNN(sigma=self.sigma)
            grnn_m.fit(train_X[sample_indices], train_y[sample_indices])

            self.base_models_.append(grnn_m)
            self.bootstrap_indices_.append(sample_indices)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """Predicts using the median of all bootstrapped base models."""
        if not self.base_models_:
            raise ValueError("Ensemble has not been fitted yet.")

        predictions_matrix = stack_base_predictions(self.base_models_, test_X)
        return np.median(predictions_matrix, axis=0)
