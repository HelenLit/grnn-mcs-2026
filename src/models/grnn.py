import math
import numpy as np


class GRNN:
    """General Regression Neural Network."""
    
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma
        self.train_X = None
        self.train_y = None

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        """
        Stores the training data (lazy learning).
        """
        self.train_X = train_X
        self.train_y = train_y

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """
        Predicts output for test_X using the trained GRNN.
        Vectorised over the full test_X matrix.
        """
        if self.train_X is None or self.train_y is None:
            raise ValueError("Model has not been fitted yet.")

        # diff shape: (n_test, n_train, n_features)
        diff = test_X[:, np.newaxis, :] - self.train_X[np.newaxis, :, :]
        
        # dist_sq shape: (n_test, n_train)
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # weights shape: (n_test, n_train)
        weights = np.exp(-dist_sq / (2 * self.sigma ** 2))
        
        # guards against zero-sum weights (query far from all training points)
        epsilon = 1e-7
        weights_sum = np.sum(weights, axis=1)
        weights_sum = np.maximum(weights_sum, epsilon)
        
        # ŷ(x_q) = Σ(weights[j] · train_y[j]) / max(Σ weights[j], ε)
        # weights * self.train_y uses broadcasting over train_y shape (n_train,)
        y_pred = np.sum(weights * self.train_y, axis=1) / weights_sum
        
        return y_pred


def select_sigma(
    train_X: np.ndarray,
    train_y: np.ndarray,
    sigma_values: np.ndarray,
    validation_size: float = 0.2
) -> float:
    """
    Selects the sigma minimising RMSE on a held-out validation split 
    of the training data.
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
        model = GRNN(sigma=sigma)
        model.fit(val_train_X, val_train_y)
        y_pred = model.predict(val_X)
        rmse = np.sqrt(np.mean((y_pred - val_y) ** 2))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_sigma = sigma
            
    return float(best_sigma)
