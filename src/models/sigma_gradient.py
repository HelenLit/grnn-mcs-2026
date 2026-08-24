"""Gradient-based sigma selection for the GRNN family.

Replaces the exhaustive sigma grid search with multi-start bounded L-BFGS-B
on the within-train validation MSE, using the ANALYTIC derivative of the
GRNN prediction with respect to sigma.

Math (per validation query x_q with squared distances d_j to fit points):
    w_j(s)  = exp(-d_j / (2 s^2))
    yhat    = sum_j w_j y_j / S,          S = sum_j w_j
    dw_j/ds = w_j * d_j / s^3
    dyhat/ds = (1/S) * sum_j dw_j (y_j - yhat)
    L(s)    = mean_i (yhat_i - t_i)^2
    dL/ds   = mean_i 2 (yhat_i - t_i) dyhat_i/ds

Ensemble aggregation:
    mean   -> gradient is the member-average gradient;
    median -> almost-everywhere gradient of the member(s) attaining the
              median (exact for odd M; even M averages the two middle members).

Optimisation runs in log-sigma space (dL/dtheta = s * dL/ds, theta = ln s)
for better conditioning, from `n_starts` geometrically spaced starts, and is
fully deterministic — no RNG anywhere.

The temporal validation split (last ceil(v*n) windows held out, no shuffling)
is IDENTICAL to the grid path (`src.models.grnn.select_sigma`), so the
two search modes differ only in how the same objective is minimised.

One deliberate difference from the grid path: `select_sigma_ensemble`
evaluates the median ensemble for BOTH interleaved variants, whereas here
`aggregate` follows the actual model ('mean' for grnn_avg, 'median' for
grnn_robust).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from src.data.preparation import decompose_into_subsequences

_EPS = 1e-7  # same zero-weight-sum guard as GRNN.predict


def _temporal_split(
    train_X: np.ndarray, train_y: np.ndarray, validation_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(train_y)
    n_val = math.ceil(n_samples * validation_size)
    n_fit = n_samples - n_val
    return train_X[:n_fit], train_y[:n_fit], train_X[n_fit:], train_y[n_fit:]


def _member_pred_and_grad(
    sigma: float,
    dist_sq: np.ndarray,
    fit_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prediction and d(prediction)/d(sigma) for one GRNN member.

    Args:
        sigma: Kernel width.
        dist_sq: Precomputed squared distances, shape (n_query, n_fit).
        fit_y: Member training targets, shape (n_fit,).

    Returns:
        (y_pred, dy_dsigma), each shaped (n_query,).
    """
    weights = np.exp(-dist_sq / (2.0 * sigma**2))
    weights_sum = np.maximum(weights.sum(axis=1), _EPS)
    y_pred = (weights * fit_y).sum(axis=1) / weights_sum

    d_weights = weights * dist_sq / sigma**3
    dy_dsigma = (d_weights * (fit_y[np.newaxis, :] - y_pred[:, np.newaxis])).sum(
        axis=1
    ) / weights_sum
    return y_pred, dy_dsigma


def _aggregate_pred_and_grad(
    preds: np.ndarray, grads: np.ndarray, aggregate: str
) -> tuple[np.ndarray, np.ndarray]:
    """Combine member predictions/gradients, shapes (M, n_query)."""
    if aggregate in ("single", "mean"):
        return preds.mean(axis=0), grads.mean(axis=0)
    if aggregate == "median":
        n_models, n_query = preds.shape
        order = np.argsort(preds, axis=0)
        cols = np.arange(n_query)
        if n_models % 2 == 1:
            mid = order[n_models // 2, :]
            return preds[mid, cols], grads[mid, cols]
        lo, hi = order[n_models // 2 - 1, :], order[n_models // 2, :]
        return (
            0.5 * (preds[lo, cols] + preds[hi, cols]),
            0.5 * (grads[lo, cols] + grads[hi, cols]),
        )
    raise ValueError(f"Unknown aggregate '{aggregate}'.")


def validation_mse_and_grad(
    sigma: float,
    member_dist_sq: list[np.ndarray],
    member_fit_y: list[np.ndarray],
    val_y: np.ndarray,
    aggregate: str,
) -> tuple[float, float]:
    """Validation MSE and dMSE/dsigma of the aggregated forecast."""
    preds, grads = [], []
    for dist_sq, fit_y in zip(member_dist_sq, member_fit_y, strict=True):
        p, g = _member_pred_and_grad(sigma, dist_sq, fit_y)
        preds.append(p)
        grads.append(g)
    y_pred, dy_dsigma = _aggregate_pred_and_grad(
        np.vstack(preds), np.vstack(grads), aggregate
    )
    residual = y_pred - val_y
    mse = float(np.mean(residual**2))
    grad = float(np.mean(2.0 * residual * dy_dsigma))
    return mse, grad


def select_sigma_gradient(
    train_X: np.ndarray,
    train_y: np.ndarray,
    n_models: int = 1,
    aggregate: str = "single",
    sigma_bounds: tuple[float, float] = (0.1, 10.0),
    validation_size: float = 0.2,
    n_starts: int = 5,
) -> float:
    """Select sigma by multi-start L-BFGS-B on the validation MSE.

    Deterministic drop-in replacement for the grid-search selectors: same
    temporal split, same objective (MSE argmin == RMSE argmin), but the
    optimum is found with analytic gradients instead of a fixed 0.1 grid.

    Args:
        train_X/train_y: Lag-windowed training data (chronological order).
        n_models: Ensemble size (1 for single GRNN / bagging selection).
        aggregate: 'single', 'mean' (grnn_avg) or 'median' (grnn_robust).
        sigma_bounds: Search interval, inclusive.
        validation_size: Held-out fraction, identical to the grid path.
        n_starts: Geometrically spaced L-BFGS-B restarts across the bounds.

    Returns:
        The sigma achieving the lowest validation MSE across all starts.
    """
    fit_X, fit_y, val_X, val_y = _temporal_split(train_X, train_y, validation_size)

    if aggregate == "single":
        members = [(fit_X, fit_y)]
    else:
        subsequences = decompose_into_subsequences(fit_X, fit_y, n_models)
        members = [(X_m, y_m) for X_m, y_m in subsequences]

    # Distances do not depend on sigma: precompute once per member.
    member_dist_sq = [
        np.sum((val_X[:, np.newaxis, :] - X_m[np.newaxis, :, :]) ** 2, axis=2)
        for X_m, _ in members
    ]
    member_fit_y = [y_m for _, y_m in members]

    lo, hi = sigma_bounds
    log_bounds = (math.log(lo), math.log(hi))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        sigma = math.exp(theta[0])
        mse, dmse_dsigma = validation_mse_and_grad(
            sigma, member_dist_sq, member_fit_y, val_y, aggregate
        )
        return mse, np.array([dmse_dsigma * sigma])  # chain rule to log-space

    best_sigma, best_mse = lo, float("inf")
    for start in np.geomspace(lo, hi, n_starts):
        result = minimize(
            objective,
            x0=np.array([math.log(start)]),
            jac=True,
            method="L-BFGS-B",
            bounds=[log_bounds],
        )
        sigma = float(np.clip(math.exp(result.x[0]), lo, hi))
        # Re-evaluate at the clipped sigma so comparisons are consistent.
        mse, _ = validation_mse_and_grad(
            sigma, member_dist_sq, member_fit_y, val_y, aggregate
        )
        if mse < best_mse:
            best_mse, best_sigma = mse, sigma

    return best_sigma
