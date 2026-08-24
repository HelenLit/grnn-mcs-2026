import math
import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tools.sm_exceptions import ConvergenceWarning as ArimaConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA


def _temporal_validation_split(
    train_X: np.ndarray, train_y: np.ndarray, validation_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological (non-shuffled) train/validation split, matching
    `src.models.grnn.select_sigma`'s discipline — the same held-out split
    used for GRNN sigma selection, reused here for baseline tuning."""
    n_samples = len(train_y)
    n_val = math.ceil(n_samples * validation_size)
    n_train_split = n_samples - n_val
    return (
        train_X[:n_train_split], train_y[:n_train_split],
        train_X[n_train_split:], train_y[n_train_split:],
    )


class SVRWrapper:
    """Uniform fit/predict wrapper around sklearn.svm.SVR."""

    def __init__(self, kernel: str, C: float, epsilon: float, gamma: str | float = "scale") -> None:
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        self.model.fit(train_X, train_y)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        return self.model.predict(test_X)


class MLPWrapper:
    """Uniform fit/predict wrapper around sklearn.neural_network.MLPRegressor.

    Scales the target internally (fit on train_y, inverse-transformed
    on predict — transparent to callers, who still pass/receive raw-scale y),
    with a raised max_iter and early_stopping enabled. Unscaled targets
    combined with a short iteration budget lead to ConvergenceWarnings and
    severely degraded fits.
    """

    def __init__(
        self,
        hidden_layer_sizes: list[int],
        max_iter: int,
        random_state: int,
        alpha: float = 0.0001,
        early_stopping: bool = False,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha,
            early_stopping=early_stopping,
        )
        self.target_scaler_ = StandardScaler()

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        train_y_scaled = self.target_scaler_.fit_transform(
            train_y.reshape(-1, 1)
        ).ravel()
        if self.model.early_stopping:
            # sklearn's early_stopping carves an internal validation split of
            # size validation_fraction * n_samples; on very short training
            # sets (e.g. M4 series after differencing + lag windowing, some
            # under 20 rows) that split can round to <2 samples and sklearn
            # raises "The validation set is too small" instead of fitting.
            # Fall back to training without early stopping for this fit
            # rather than crashing the whole (series, model) cell.
            min_required = int(len(train_y) * self.model.validation_fraction)
            if min_required < 2:
                self.model.early_stopping = False
        self.model.fit(train_X, train_y_scaled)

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        y_pred_scaled = self.model.predict(test_X)
        return self.target_scaler_.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()


def select_svr_hyperparams(
    train_X: np.ndarray,
    train_y: np.ndarray,
    C_values: list[float],
    gamma_values: list,
    epsilon_values: list[float],
    validation_size: float = 0.2,
) -> tuple[float, float | str, float]:
    """Selects (C, gamma, epsilon) minimising RMSE on a temporal validation
    split — kernel is left fixed (not part of the tuned grid)."""
    val_train_X, val_train_y, val_X, val_y = _temporal_validation_split(
        train_X, train_y, validation_size
    )

    best_params = (C_values[0], gamma_values[0], epsilon_values[0])
    best_rmse = float("inf")
    for C in C_values:
        for gamma in gamma_values:
            for epsilon in epsilon_values:
                model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
                model.fit(val_train_X, val_train_y)
                y_pred = model.predict(val_X)
                rmse = float(np.sqrt(np.mean((y_pred - val_y) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (C, gamma, epsilon)
    return best_params


def select_mlp_hyperparams(
    train_X: np.ndarray,
    train_y: np.ndarray,
    hidden_layer_sizes_grid: list[list[int]],
    alpha_values: list[float],
    max_iter: int,
    random_state: int,
    early_stopping: bool,
    validation_size: float = 0.2,
) -> tuple[list[int], float]:
    """Selects (hidden_layer_sizes, alpha) minimising RMSE on a temporal
    validation split. max_iter/random_state/early_stopping
    are held fixed — only the requested grid dimensions are tuned."""
    val_train_X, val_train_y, val_X, val_y = _temporal_validation_split(
        train_X, train_y, validation_size
    )

    best_params = (hidden_layer_sizes_grid[0], alpha_values[0])
    best_rmse = float("inf")
    for hidden_layer_sizes in hidden_layer_sizes_grid:
        for alpha in alpha_values:
            model = MLPWrapper(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
                alpha=alpha,
                early_stopping=early_stopping,
            )
            model.fit(val_train_X, val_train_y)
            y_pred = model.predict(val_X)
            rmse = float(np.sqrt(np.mean((y_pred - val_y) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (hidden_layer_sizes, alpha)
    return best_params


def select_arima_order(
    train_y: np.ndarray,
    order_grid: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Selects (p, d, q) minimising in-sample AIC over a small order grid
    — AIC needs no held-out split, it is computed from the fit
    itself. Orders that fail to fit, OR that fit but emit a
    ConvergenceWarning (MLE did not converge — a valid AIC on an unreliable
    fit), are excluded from selection, not treated as candidates; falls back
    to the first grid entry if every candidate is excluded."""
    best_order = tuple(order_grid[0])
    best_aic = float("inf")
    any_converged = False
    for order in order_grid:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", category=ArimaConvergenceWarning)
                result = ARIMA(train_y, order=tuple(order)).fit()
        except (ValueError, np.linalg.LinAlgError):
            continue
        if any(issubclass(w.category, ArimaConvergenceWarning) for w in caught):
            continue
        if result.aic < best_aic:
            best_aic = result.aic
            best_order = tuple(order)
            any_converged = True
    if not any_converged:
        return tuple(order_grid[0])
    return best_order


class ARIMAWrapper:
    """Uniform interface wrapper around statsmodels ARIMA."""

    def __init__(self, order: tuple[int, int, int]) -> None:
        self.order = order
        self.model = None
        self.result = None

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        self.model = ARIMA(train_y, order=self.order)
        self.result = self.model.fit()

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        if self.result is None:
            raise ValueError("Model has not been fitted yet.")

        forecast = self.result.get_forecast(steps=len(test_X))
        return np.asarray(forecast.predicted_mean)
