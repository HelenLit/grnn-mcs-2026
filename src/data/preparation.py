import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

_SCALER_CLASSES = {
    "MaxAbsScaler": MaxAbsScaler,
    "MinMaxScaler": MinMaxScaler,
    "StandardScaler": StandardScaler,
}


@dataclass
class PreparedDataset:
    """Container for holding prepared training and testing datasets.

    Fields:
        train_X: np.ndarray of training features.
        train_y: np.ndarray of training target values.
        test_X: np.ndarray of testing features.
        test_y: np.ndarray of testing target values.
        scaler: The fitted sklearn scaler object used for feature scaling.
        test_prev_levels: When differencing is applied, the actual
            original-series level immediately preceding each test target, used
            to invert difference predictions back to levels
            (level_hat = test_prev_levels + diff_pred). None otherwise.
        test_true_levels: When differencing is applied, the actual
            original-series levels corresponding to the (differenced) test
            targets — the ground truth for level-space metrics. None otherwise.
    """
    train_X: np.ndarray
    train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    scaler: object
    test_prev_levels: np.ndarray | None = None
    test_true_levels: np.ndarray | None = None


def load_synthetic_series(filepath: str) -> np.ndarray:
    """Reads the 'y' column from a two-column (t, y) CSV file.
    
    Args:
        filepath: Absolute or relative path to the synthetic series CSV file.
        
    Returns:
        A 1-D float64 numpy array of values from the 'y' column.
        
    Raises:
        FileNotFoundError: If the file at filepath does not exist.
    """
    df = pd.read_csv(filepath)
    if "y" not in df.columns:
        raise ValueError(f"CSV file at {filepath} does not contain required column 'y'.")
    return df["y"].to_numpy(dtype=np.float64)


def load_raw_dataset(train_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reads train and test files for tabular benchmark datasets.
    
    The files are headerless space or comma-delimited text files.
    The last column represents the target y. All preceding columns are features X.
    
    Args:
        train_path: Path to the training text file.
        test_path: Path to the testing text file.
        
    Returns:
        A tuple of (train_X, train_y, test_X, test_y) as float64 numpy arrays.
    """
    # Detect delimiter based on the first line of the train file
    with open(train_path, "r") as f:
        first_line = f.readline()
    
    sep = "," if "," in first_line else r"\s+"
    
    df_train = pd.read_csv(train_path, header=None, sep=sep)
    df_test = pd.read_csv(test_path, header=None, sep=sep)
    
    train_data = df_train.to_numpy(dtype=np.float64)
    test_data = df_test.to_numpy(dtype=np.float64)
    
    train_X = train_data[:, :-1]
    train_y = train_data[:, -1]
    test_X = test_data[:, :-1]
    test_y = test_data[:, -1]
    
    return train_X, train_y, test_X, test_y


def difference_series(series: np.ndarray) -> tuple[np.ndarray, float]:
    """First-differences a univariate series (trend handling).

    GRNN is a Nadaraya-Watson interpolator: its predictions are a weighted
    average of training targets and therefore cannot leave the training range.
    On trending series the test tail exceeds that range and the model
    collapses. Modelling first differences d(t) = s(t) - s(t-1) — which are
    roughly stationary — sidesteps this: the network predicts increments that
    fall within the training range, and forecasts are mapped back to levels
    via `invert_difference` (or by adding the increment to the known previous
    level). This removes a constant/stochastic trend but NOT acceleration,
    level shifts, or regime changes — the no-extrapolation limit remains.

    Args:
        series: 1-D array of length N.

    Returns:
        A tuple (diffs, anchor) where diffs has length N-1 with
        diffs[k] = series[k+1] - series[k], and anchor = series[0] is the
        level needed to reconstruct the original series.
    """
    assert series.ndim == 1, f"Expected 1-D array, got ndim={series.ndim}"
    assert len(series) >= 2, f"series must have >= 2 points, got {len(series)}"
    return np.diff(series), float(series[0])


def invert_difference(diffs: np.ndarray, anchor: float) -> np.ndarray:
    """Reconstructs a level series from its first differences and anchor.

    Exact inverse of `difference_series`: given diffs of length N-1 and the
    original first value, returns the level series of length N. Used to map
    a fully-differenced series (or a recursively-forecast difference horizon
    anchored on the last known level) back to levels.

    Args:
        diffs: 1-D array of first differences (length N-1).
        anchor: The level immediately before the first difference.

    Returns:
        A 1-D array of levels of length N (anchor prepended to the cumulative
        sum of diffs).
    """
    return np.concatenate(([anchor], anchor + np.cumsum(diffs)))


def create_sliding_window(series: np.ndarray, lag_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Constructs a supervised learning dataset from a univariate series using a sliding window.
    
    For a series of length N and lag_order L:
    - Output X has shape (N - L, L)
    - Output y has shape (N - L,)
    - X[i] = [series[i], series[i+1], ..., series[i+L-1]]
    - y[i] = series[i+L]
    
    Args:
        series: 1-D numpy array of univariate time series data.
        lag_order: Number of past values to use as input features.
        
    Returns:
        A tuple of (X, y) as numpy arrays.
    """
    assert series.ndim == 1, f"Expected 1-D array for series, got ndim={series.ndim}"
    assert lag_order >= 1, f"lag_order must be >= 1, got {lag_order}"
    assert len(series) > lag_order, f"series length ({len(series)}) must be greater than lag_order ({lag_order})"
    
    N = len(series)
    X = np.empty((N - lag_order, lag_order), dtype=series.dtype)
    y = np.empty(N - lag_order, dtype=series.dtype)
    
    for i in range(N - lag_order):
        X[i] = series[i : i + lag_order]
        y[i] = series[i + lag_order]
        
    return X, y


def split_train_test(
    X: np.ndarray, y: np.ndarray, test_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits features and target values chronologically (without shuffling).
    
    The last ceil(len(y) * test_size) samples become the test set,
    and all preceding samples become the training set.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        test_size: Fraction of samples to assign to the test set (0 < test_size < 1).
        
    Returns:
        A tuple of (train_X, train_y, test_X, test_y).
    """
    assert 0.0 < test_size < 1.0, f"test_size must be between 0 and 1 (exclusive), got {test_size}"
    assert len(X) == len(y), f"X and y must have the same length, got {len(X)} and {len(y)}"
    
    n_samples = len(y)
    test_count = math.ceil(n_samples * test_size)
    train_count = n_samples - test_count
    
    train_X = X[:train_count]
    train_y = y[:train_count]
    test_X = X[train_count:]
    test_y = y[train_count:]
    
    return train_X, train_y, test_X, test_y


def scale_features(
    train_X: np.ndarray,
    test_X: np.ndarray,
    scaler_type: str
) -> tuple[np.ndarray, np.ndarray, object]:
    """Fits the specified scaler on train_X, then transforms both train_X and test_X.
    
    Supported scaler types:
    - "MaxAbsScaler": sklearn.preprocessing.MaxAbsScaler
    - "MinMaxScaler": sklearn.preprocessing.MinMaxScaler
    - "StandardScaler": sklearn.preprocessing.StandardScaler
    
    Args:
        train_X: Training feature matrix.
        test_X: Testing feature matrix.
        scaler_type: Name of the scaler class to use.
        
    Returns:
        A tuple of (train_X_scaled, test_X_scaled, fitted_scaler).
        
    Raises:
        ValueError: If scaler_type is not one of the supported strings.
    """
    if scaler_type not in _SCALER_CLASSES:
        raise ValueError(
            f"Unknown scaler_type '{scaler_type}'. "
            f"Supported scalers: {list(_SCALER_CLASSES.keys())}"
        )

    scaler = _SCALER_CLASSES[scaler_type]()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)
    
    return train_X_scaled, test_X_scaled, scaler


@dataclass
class FeaturePipeline:
    """A fitted scaler transform, replayable on new raw feature rows.

    `scale_features` fits-and-transforms two FIXED matrices (train, test) in
    one call. A recursive multi-step forecaster (src/evaluation/m4_harness.py)
    instead needs to push ONE NEW raw lag row per step through the SAME
    fitted scaler used at training time — this wraps that fitted object for
    reuse.

    Fields:
        scaler: Fitted sklearn scaler (see `scale_features`).
    """
    scaler: object

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies the fitted scaler."""
        return self.scaler.transform(X)


def fit_feature_pipeline(
    train_X: np.ndarray,
    scaler_type: str,
) -> tuple[np.ndarray, FeaturePipeline]:
    """Fits a scaler on train_X once and returns a reusable transform.

    Equivalent in effect to `scale_features` called on (train_X, train_X),
    but also returns the fitted object so a caller can transform future
    single rows with the identical parameters.

    Args:
        train_X: Training feature matrix (raw lags).
        scaler_type: Scaler class name string (see `scale_features`).

    Returns:
        A tuple (train_X_transformed, pipeline).

    Raises:
        ValueError: If scaler_type is not one of the supported strings.
    """
    if scaler_type not in _SCALER_CLASSES:
        raise ValueError(
            f"Unknown scaler_type '{scaler_type}'. "
            f"Supported scalers: {list(_SCALER_CLASSES.keys())}"
        )

    scaler = _SCALER_CLASSES[scaler_type]()
    train_X_scaled = scaler.fit_transform(train_X)
    return train_X_scaled, FeaturePipeline(scaler=scaler)


def prepare_synthetic_from_series(
    series: np.ndarray,
    lag_order: int,
    test_size: float,
    scaler_type: str,
    difference: bool = False
) -> PreparedDataset:
    """Preprocesses an in-memory univariate series into a PreparedDataset.

    Applies the full pipeline:
    [difference] -> window -> split -> scale -> PreparedDataset

    This is `prepare_synthetic` without the CSV load step, so a caller that
    already has a series in memory (e.g. a multi-seed driver regenerating a
    fresh seeded realization per repeat) does not need to round-trip it
    through disk. `prepare_synthetic` is a thin wrapper around this function.

    When `difference` is True, the series is first-differenced BEFORE
    windowing, so `train_y`/`test_y` become increments and the model learns to
    forecast differences. The dataset then also carries `test_prev_levels` and
    `test_true_levels`, which let a caller invert difference predictions back
    to level space (`level_hat = test_prev_levels + diff_pred`) and score them
    against the true levels. Differencing is off by default: it fixes trends
    but inflates the variance of already-stationary series, so it should be
    enabled per-dataset (`config.yaml -> preparation.difference`), not globally.

    Args:
        series: 1-D array of the univariate time series.
        lag_order: Feature lag order.
        test_size: Train-test partition test fraction.
        scaler_type: Scaler class name string.
        difference: If True, first-difference the series before windowing and
            populate the level-inversion fields on the returned dataset.

    Returns:
        A PreparedDataset containing the processed train/test splits and scaler.
    """
    if not difference:
        X, y = create_sliding_window(series, lag_order)
        train_X, train_y, test_X, test_y = split_train_test(X, y, test_size)
        train_X_scaled, test_X_scaled, scaler = scale_features(train_X, test_X, scaler_type)
        return PreparedDataset(
            train_X=train_X_scaled,
            train_y=train_y,
            test_X=test_X_scaled,
            test_y=test_y,
            scaler=scaler,
        )

    # Differenced path. Window the differences; keep aligned original-series
    # levels so difference predictions can be inverted back to levels.
    # For windowed sample i: y[i] = diffs[i+L] = series[i+L+1] - series[i+L],
    # so the preceding level is series[i+L] and the true level is series[i+L+1].
    diffs, _ = difference_series(series)
    X, y = create_sliding_window(diffs, lag_order)
    prev_levels = series[lag_order : len(series) - 1]
    true_levels = series[lag_order + 1 : len(series)]

    train_X, train_y, test_X, test_y = split_train_test(X, y, test_size)
    _, _, test_prev_levels, test_true_levels = split_train_test(
        prev_levels, true_levels, test_size
    )
    train_X_scaled, test_X_scaled, scaler = scale_features(train_X, test_X, scaler_type)

    return PreparedDataset(
        train_X=train_X_scaled,
        train_y=train_y,
        test_X=test_X_scaled,
        test_y=test_y,
        scaler=scaler,
        test_prev_levels=test_prev_levels,
        test_true_levels=test_true_levels,
    )


def prepare_synthetic(
    filepath: str,
    lag_order: int,
    test_size: float,
    scaler_type: str,
    difference: bool = False
) -> PreparedDataset:
    """Preprocesses a synthetic univariate series into a PreparedDataset.

    Applies the full pipeline:
    load -> [difference] -> window -> split -> scale -> PreparedDataset

    See `prepare_synthetic_from_series` for the pipeline itself and the
    `difference` semantics; this wrapper only adds the CSV load step.

    Args:
        filepath: Path to the synthetic time series CSV.
        lag_order: Feature lag order.
        test_size: Train-test partition test fraction.
        scaler_type: Scaler class name string.
        difference: If True, first-difference the series before windowing and
            populate the level-inversion fields on the returned dataset.

    Returns:
        A PreparedDataset containing the processed train/test splits and scaler.
    """
    series = load_synthetic_series(filepath)
    return prepare_synthetic_from_series(
        series, lag_order, test_size, scaler_type, difference
    )


def decompose_into_subsequences(
    train_X: np.ndarray,
    train_y: np.ndarray,
    n_models: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decomposes training data into M interleaved sub-sequences.

    Implements the structured decomposition step of the GRNN-Robust ensemble.
    Sub-sequence m (0-indexed) receives samples at indices m, m+n_models, m+2*n_models, ...
    No samples are duplicated or omitted across sub-sequences.

    For n_models=3 and N=77 training samples:
      m=0 -> indices 0, 3, 6, ..., 75  (26 samples)
      m=1 -> indices 1, 4, 7, ..., 76  (26 samples)
      m=2 -> indices 2, 5, 8, ..., 74  (25 samples)

    Args:
        train_X: Training feature matrix of shape (N, n_features).
        train_y: Training target array of shape (N,).
        n_models: Number of base models M; determines the interleaving stride.

    Returns:
        List of M (X_m, y_m) tuples, one per base model, in order m=0..M-1.
    """
    assert n_models >= 2, f"n_models must be >= 2, got {n_models}"
    assert len(train_X) >= n_models, (
        f"Training set size ({len(train_X)}) must be >= n_models ({n_models})"
    )

    return [(train_X[m::n_models], train_y[m::n_models]) for m in range(n_models)]


def prepare_raw(
    train_path: str,
    test_path: str,
    scaler_type: str,
) -> PreparedDataset:
    """Preprocesses a raw benchmark dataset into a PreparedDataset.

    Applies the pipeline (no windowing since features are already tabular):
    load -> scale -> PreparedDataset

    Args:
        train_path: Path to the raw train split file.
        test_path: Path to the raw test split file.
        scaler_type: Scaler class name string.

    Returns:
        A PreparedDataset containing the processed train/test splits and scaler.
    """
    train_X, train_y, test_X, test_y = load_raw_dataset(train_path, test_path)
    train_X_scaled, test_X_scaled, scaler = scale_features(train_X, test_X, scaler_type)

    return PreparedDataset(
        train_X=train_X_scaled,
        train_y=train_y,
        test_X=test_X_scaled,
        test_y=test_y,
        scaler=scaler
    )


def prepare_raw_cv_folds(
    train_path: str,
    test_path: str,
    scaler_type: str,
    cv_folds: int,
    random_state: int,
) -> list[PreparedDataset]:
    """Preprocesses a raw benchmark dataset into K-fold cross-validation splits.

    The fixed single train/test split (`prepare_raw`) evaluates only
    the 6-7 held-out rows in the original test file — too few points to
    support a conclusion. This pools train+test into one dataset, then
    performs K-fold CV (shuffled — OBD/body_fat are non-temporal tabular
    data, so shuffling is valid, unlike the synthetic series' chronological
    split). Each fold refits its own scaler on that fold's training rows
    only, so there is no leakage from a fold's held-out rows.

    Args:
        train_path: Path to the raw train split file.
        test_path: Path to the raw test split file.
        scaler_type: Scaler class name string.
        cv_folds: Number of folds K.
        random_state: Seed for the fold shuffling (reproducible fold membership).

    Returns:
        A list of `cv_folds` PreparedDataset objects, one per fold.
    """
    train_X, train_y, test_X, test_y = load_raw_dataset(train_path, test_path)
    pooled_X = np.concatenate([train_X, test_X], axis=0)
    pooled_y = np.concatenate([train_y, test_y], axis=0)

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    folds = []
    for fold_train_idx, fold_test_idx in kfold.split(pooled_X):
        fold_train_X_scaled, fold_test_X_scaled, scaler = scale_features(
            pooled_X[fold_train_idx], pooled_X[fold_test_idx], scaler_type
        )
        folds.append(PreparedDataset(
            train_X=fold_train_X_scaled,
            train_y=pooled_y[fold_train_idx],
            test_X=fold_test_X_scaled,
            test_y=pooled_y[fold_test_idx],
            scaler=scaler,
        ))
    return folds
