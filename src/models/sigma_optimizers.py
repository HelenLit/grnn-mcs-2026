"""Sigma-selection strategies for the GRNN family behind one common interface.

Implements three search methods:

    brent        multi-start bounded Brent (derivative-free — formally correct
                 for the piecewise-smooth median-ensemble objective)
    coarse2fine  3-level refined grid (log-spaced coarse level, linear zooms)
    silverman    analytic bandwidth rule (zero objective evaluations)

plus counting wrappers exposing the SAME interface for the grid and gradient
modes:

    grid         exhaustive grid over sigma_values (argmin re-expressed
                 on the shared objective)
    gradient     multi-start L-BFGS-B with analytic dMSE/dsigma (mirrors
                 src.models.sigma_gradient.select_sigma_gradient exactly)

All five minimise the byte-identical objective — the validation MSE of the
aggregated forecast on the same temporal 80/20 split as the M4 pipeline runs
(`validation_mse_and_grad` from src.models.sigma_gradient) — and differ only
in HOW the minimum is located. Distances are precomputed once per call, so
the microbenchmark isolates optimizer overhead (number of evaluations), not
implementation overhead. Everything is deterministic: no RNG anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

from src.data.preparation import decompose_into_subsequences
from src.models.sigma_gradient import _temporal_split
from src.models.sigma_gradient import validation_mse_and_grad


@dataclass(frozen=True)
class SigmaSearchResult:
    """Outcome of one sigma search.

    Fields:
        sigma: Selected kernel width, clipped to the search bounds.
        val_mse: Objective value at `sigma` (NaN for the silverman rule,
            which never evaluates the objective).
        n_evals: Objective evaluations consumed (0 for silverman).
    """
    sigma: float
    val_mse: float
    n_evals: int


class _CountingObjective:
    """Validation-MSE objective with precomputed distances and an eval counter."""

    def __init__(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        n_models: int,
        aggregate: str,
        validation_size: float,
    ) -> None:
        fit_X, fit_y, val_X, self.val_y = _temporal_split(
            train_X, train_y, validation_size
        )
        self.fit_X = fit_X
        self.fit_y = fit_y
        if aggregate == "single":
            members = [(fit_X, fit_y)]
        else:
            members = list(decompose_into_subsequences(fit_X, fit_y, n_models))
        self.member_dist_sq = [
            np.sum((val_X[:, np.newaxis, :] - X_m[np.newaxis, :, :]) ** 2, axis=2)
            for X_m, _ in members
        ]
        self.member_fit_y = [y_m for _, y_m in members]
        self.aggregate = aggregate
        self.n_evals = 0

    def mse(self, sigma: float) -> float:
        return self.mse_and_grad(sigma)[0]

    def mse_and_grad(self, sigma: float) -> tuple[float, float]:
        self.n_evals += 1
        return validation_mse_and_grad(
            sigma, self.member_dist_sq, self.member_fit_y, self.val_y,
            self.aggregate,
        )


def select_sigma_brent(
    train_X: np.ndarray,
    train_y: np.ndarray,
    n_models: int = 1,
    aggregate: str = "median",
    sigma_bounds: tuple[float, float] = (0.1, 10.0),
    validation_size: float = 0.2,
    n_starts: int = 4,
) -> SigmaSearchResult:
    """Multi-start bounded Brent (golden section + parabolic interpolation).

    [lo, hi] is split into `n_starts` log-spaced brackets; SciPy's bounded
    Brent runs inside each and the best minimum wins. Derivative-free, so the
    kinks of the median-ensemble objective are irrelevant to correctness.
    """
    objective = _CountingObjective(
        train_X, train_y, n_models, aggregate, validation_size
    )
    lo, hi = sigma_bounds
    edges = np.geomspace(lo, hi, n_starts + 1)

    best_sigma, best_mse = lo, float("inf")
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        result = minimize_scalar(
            objective.mse,
            bounds=(float(left), float(right)),
            method="bounded",
            options={"xatol": 1e-3},
        )
        sigma = float(np.clip(result.x, lo, hi))
        if result.fun < best_mse:
            best_mse, best_sigma = float(result.fun), sigma
    return SigmaSearchResult(best_sigma, best_mse, objective.n_evals)


def select_sigma_coarse2fine(
    train_X: np.ndarray,
    train_y: np.ndarray,
    n_models: int = 1,
    aggregate: str = "median",
    sigma_bounds: tuple[float, float] = (0.1, 10.0),
    validation_size: float = 0.2,
    n_levels: int = 3,
) -> SigmaSearchResult:
    """Successively refined grid: 12 log-spaced points, then linear zooms.

    Each refinement level places 9 linear points across the two grid
    neighbours of the current argmin; the best sigma seen anywhere wins.
    Deterministic and derivative-free, ~30 evaluations at n_levels=3.
    """
    objective = _CountingObjective(
        train_X, train_y, n_models, aggregate, validation_size
    )
    lo, hi = sigma_bounds
    points = np.geomspace(lo, hi, 12)

    best_sigma, best_mse = lo, float("inf")
    for level in range(n_levels):
        values = [objective.mse(float(sigma)) for sigma in points]
        idx = int(np.argmin(values))
        if values[idx] < best_mse:
            best_mse, best_sigma = values[idx], float(points[idx])
        left = points[max(idx - 1, 0)]
        right = points[min(idx + 1, len(points) - 1)]
        if level < n_levels - 1:
            points = np.linspace(left, right, 9)
    return SigmaSearchResult(
        float(np.clip(best_sigma, lo, hi)), best_mse, objective.n_evals
    )


def select_sigma_silverman(
    train_X: np.ndarray,
    train_y: np.ndarray,
    n_models: int = 1,
    aggregate: str = "median",
    sigma_bounds: tuple[float, float] = (0.1, 10.0),
    validation_size: float = 0.2,
) -> SigmaSearchResult:
    """Silverman-type kernel bandwidth — no search at all.

        sigma = c * s * n^(-1/(d+4)),   c = (4/(d+2))^(1/(d+4))

    with d = number of lag features, n = fit-split size, s = mean per-feature
    std of the fit inputs. A density-estimation rule, not a forecast-error
    minimiser — the zero-cost reference the searched methods must beat.
    n_models/aggregate are accepted (uniform interface) but unused.
    """
    fit_X, _, _, _ = _temporal_split(train_X, train_y, validation_size)
    n, d = fit_X.shape
    spread = float(np.mean(np.std(fit_X, axis=0, ddof=0)))
    if spread <= 0.0:
        spread = 1.0  # near-constant differenced series; clip logs the rest
    c = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    sigma = c * spread * n ** (-1.0 / (d + 4.0))
    lo, hi = sigma_bounds
    return SigmaSearchResult(float(np.clip(sigma, lo, hi)), float("nan"), 0)


@dataclass(frozen=True)
class ObjectiveDiagnostics:
    """Health probe of one cell's sigma objective (v2 hardening).

    Fields:
        flat: True when L(sigma) is numerically constant across the probe
            points {lo, sqrt(lo*hi), hi} — the v1 failure mode in which every
            optimizer degenerates to its tie-breaking convention.
        underflow_frac: Fraction of (validation query, member) pairs whose
            total kernel weight falls below the 1e-7 guard at the SELECTED
            sigma, i.e. where the GRNN predicts ~0 regardless of the data.
    """
    flat: bool
    underflow_frac: float


def diagnose_objective(
    train_X: np.ndarray,
    train_y: np.ndarray,
    n_models: int = 1,
    aggregate: str = "median",
    sigma: float = 1.0,
    sigma_bounds: tuple[float, float] = (0.1, 10.0),
    validation_size: float = 0.2,
) -> ObjectiveDiagnostics:
    """Probe whether the sigma objective is informative for one cell.

    Costs three objective evaluations plus one weight pass — negligible next
    to the fit/forecast work of the cell. Logged per cell so degenerate
    regimes (the root cause found in run `sigma_opt` v1: selection on raw
    features underflows the kernel) can never again pass silently.
    """
    objective = _CountingObjective(
        train_X, train_y, n_models, aggregate, validation_size
    )
    lo, hi = sigma_bounds
    probes = np.array([
        objective.mse(float(s)) for s in (lo, math.sqrt(lo * hi), hi)
    ])
    reference = max(float(np.max(np.abs(probes))), 1e-300)
    flat = bool(
        np.all(np.isfinite(probes))
        and float(probes.max() - probes.min()) <= 1e-9 * reference
    )
    member_sums = [
        np.exp(-dist_sq / (2.0 * sigma**2)).sum(axis=1)
        for dist_sq in objective.member_dist_sq
    ]
    underflow_frac = float(np.mean([
        float((sums < 1e-7).mean()) for sums in member_sums
    ]))
    return ObjectiveDiagnostics(flat, underflow_frac)


def select_sigma_grid_counted(
    train_X: np.ndarray,
    train_y: np.ndarray,
    n_models: int = 1,
    aggregate: str = "median",
    sigma_bounds: tuple[float, float] = (0.1, 10.0),
    validation_size: float = 0.2,
    step: float = 0.1,
) -> SigmaSearchResult:
    """Exhaustive grid argmin on the shared objective (microbenchmark wrapper).

    Same grid layout as `src.models.grnn.select_sigma` (lo..hi inclusive,
    fixed step). This wrapper exists so timing/eval counts are measured on
    the SAME objective machinery as every other method.
    """
    objective = _CountingObjective(
        train_X, train_y, n_models, aggregate, validation_size
    )
    lo, hi = sigma_bounds
    grid = np.arange(lo, hi + step / 2.0, step)
    values = [objective.mse(float(sigma)) for sigma in grid]
    idx = int(np.argmin(values))
    return SigmaSearchResult(float(grid[idx]), values[idx], objective.n_evals)


def select_sigma_gradient_counted(
    train_X: np.ndarray,
    train_y: np.ndarray,
    n_models: int = 1,
    aggregate: str = "median",
    sigma_bounds: tuple[float, float] = (0.1, 10.0),
    validation_size: float = 0.2,
    n_starts: int = 5,
) -> SigmaSearchResult:
    """Multi-start L-BFGS-B with analytic gradient, with evaluation counting.

    Mirrors `src.models.sigma_gradient.select_sigma_gradient` step for step
    (log-space, 5 geomspace starts, clip + re-evaluate); only the bookkeeping
    differs.
    """
    objective = _CountingObjective(
        train_X, train_y, n_models, aggregate, validation_size
    )
    lo, hi = sigma_bounds
    log_bounds = (math.log(lo), math.log(hi))

    def wrapped(theta: np.ndarray) -> tuple[float, np.ndarray]:
        sigma = math.exp(theta[0])
        mse, dmse_dsigma = objective.mse_and_grad(sigma)
        return mse, np.array([dmse_dsigma * sigma])

    best_sigma, best_mse = lo, float("inf")
    for start in np.geomspace(lo, hi, n_starts):
        result = minimize(
            wrapped,
            x0=np.array([math.log(start)]),
            jac=True,
            method="L-BFGS-B",
            bounds=[log_bounds],
        )
        sigma = float(np.clip(math.exp(result.x[0]), lo, hi))
        mse = objective.mse(sigma)
        if mse < best_mse:
            best_mse, best_sigma = mse, sigma
    return SigmaSearchResult(best_sigma, best_mse, objective.n_evals)


# Runner-facing registry: the brent/coarse2fine/silverman methods produce
# quality rows; the counted wrappers additionally serve the timing
# microbenchmark for all five modes.
NEW_METHODS = {
    "brent": select_sigma_brent,
    "coarse2fine": select_sigma_coarse2fine,
    "silverman": select_sigma_silverman,
}
ALL_METHODS = {
    "grid": select_sigma_grid_counted,
    "gradient": select_sigma_gradient_counted,
    **NEW_METHODS,
}


def _sanity_demo() -> None:
    """Sanity checks on synthetic series."""
    from src.data.preparation import create_sliding_window

    rng_free_series = [
        np.sin(np.linspace(0.0, 12.0, 60)) + 0.05 * np.cos(np.linspace(0, 90, 60)),
        np.linspace(0.0, 5.0, 50) % 1.7,
        np.cumsum(np.sin(np.arange(45) * 1.7)),
    ]
    for index, series in enumerate(rng_free_series):
        X, y = create_sliding_window(series, 3)
        exhaustive = select_sigma_grid_counted(X, y, 3, "median")
        for name in ("brent", "coarse2fine"):
            first = ALL_METHODS[name](X, y, 3, "median")
            second = ALL_METHODS[name](X, y, 3, "median")
            assert first == second, f"{name}: not deterministic on series {index}"
            gap = abs(first.sigma - exhaustive.sigma)
            match = "OK " if (gap <= 0.1 or first.val_mse <= exhaustive.val_mse)\
                else "MISS"
            print(
                f"series {index} {name:12s} sigma={first.sigma:7.4f} "
                f"(grid {exhaustive.sigma:4.1f})  mse={first.val_mse:.6f} "
                f"(grid {exhaustive.val_mse:.6f})  evals={first.n_evals:3d}  {match}"
            )
            assert gap <= 0.1 or first.val_mse <= exhaustive.val_mse, (
                f"{name} missed the grid optimum on series {index}"
            )
        rule = select_sigma_silverman(X, y)
        assert 0.1 <= rule.sigma <= 10.0 and rule.n_evals == 0
        print(f"series {index} {'silverman':12s} sigma={rule.sigma:7.4f}")

    # v2 regression check — REAL-SCALE data (the v1 sanity block only used
    # O(1) synthetic values, which is exactly the regime where the kernel
    # never underflows, so it could not catch the raw-feature degeneracy).
    from src.data.preparation import fit_feature_pipeline

    big_series = 1.0e4 * (
        np.cumsum(np.sin(np.arange(40) * 1.3) + 0.4) + 5.0
    )
    X_big, y_big = create_sliding_window(np.diff(big_series), 3)
    raw_diag = diagnose_objective(X_big, y_big, 5, "median", sigma=1.0)
    assert raw_diag.flat and raw_diag.underflow_frac > 0.99, (
        "expected the RAW-feature objective to be degenerate on M4-scale "
        "data (documents the v1 defect) — if this fires, the probe is broken"
    )
    n_fit = len(y_big) - math.ceil(len(y_big) * 0.2)
    _, pipeline = fit_feature_pipeline(X_big[:n_fit], "MaxAbsScaler", 2)
    fixed_diag = diagnose_objective(
        pipeline.transform(X_big), y_big, 5, "median", sigma=1.0
    )
    assert not fixed_diag.flat and fixed_diag.underflow_frac < 0.5, (
        "deployment-space objective must be informative on M4-scale data — "
        "the selection-space consistency fix is not working"
    )
    print(f"real-scale check: raw flat={raw_diag.flat} "
          f"underflow={raw_diag.underflow_frac:.0%} | deployment-space "
          f"flat={fixed_diag.flat} underflow={fixed_diag.underflow_frac:.0%}")
    print("sanity: all checks passed")


if __name__ == "__main__":
    _sanity_demo()
