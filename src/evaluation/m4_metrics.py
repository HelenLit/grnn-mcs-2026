"""Official M4 Competition point-forecast metrics: sMAPE, MASE, OWA.

Definitions follow Makridakis, Spiliotis & Assimakopoulos (2020), IJF 36(1),
and the official evaluation code at github.com/Mcompetitions/M4-methods:
- sMAPE and MASE are computed PER SERIES over its forecast horizon, then
  averaged across series with equal weights;
- the MASE denominator uses the in-sample (training) seasonal-naive errors
  with seasonality m (m=1 for Yearly);
- OWA is computed once over the aggregated averages, relative to Naive2.

These metrics are for the M4 real-data benchmark only. The generic regression
metric contract (`src/evaluation/metrics.py compute_metrics`) is untouched.
"""
import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE of one series' forecast horizon, in percent.

    sMAPE = mean( 2*|y - yhat| / (|y| + |yhat|) ) * 100.

    M4 series are shifted so their minimum is 10, so the denominator is safe;
    a zero denominator would make the result NaN by design rather than crash.

    Args:
        y_true: Actual values over the forecast horizon.
        y_pred: Forecast values over the forecast horizon.

    Returns:
        The sMAPE value in percent.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(
        np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
        * 100.0
    )


def mase(
    insample: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    m: int,
) -> float:
    """Mean Absolute Scaled Error of one series (Hyndman & Koehler 2006).

    MASE = mean(|y - yhat|) / mean(|insample_t - insample_{t-m}|),
    with the denominator computed on the TRAINING series only.

    Args:
        insample: Training values of the series.
        y_true: Actual values over the forecast horizon.
        y_pred: Forecast values over the forecast horizon.
        m: Seasonality period for the naive scaling (1 for M4-Yearly).

    Returns:
        The MASE value.

    Raises:
        ValueError: If the training series is too short for seasonality m.
    """
    insample = np.asarray(insample, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(insample) <= m:
        raise ValueError(
            f"Training series of length {len(insample)} is too short for "
            f"seasonality m={m}"
        )
    scale = np.mean(np.abs(insample[m:] - insample[:-m]))
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def owa(
    smape_value: float,
    mase_value: float,
    naive2_smape: float,
    naive2_mase: float,
) -> float:
    """Overall Weighted Average relative to Naive2 (M4's official criterion).

    OWA = 0.5 * (sMAPE / sMAPE_Naive2 + MASE / MASE_Naive2), computed on
    metric values already averaged across the evaluated sample.

    Args:
        smape_value: Aggregated sMAPE of the evaluated method.
        mase_value: Aggregated MASE of the evaluated method.
        naive2_smape: Naive2 sMAPE over the SAME sample (or the published
            reference value when comparing against the official leaderboard).
        naive2_mase: Naive2 MASE over the same sample (or published value).

    Returns:
        The OWA value.
    """
    return 0.5 * (smape_value / naive2_smape + mase_value / naive2_mase)


def naive2_forecast_yearly(train: np.ndarray, horizon: int) -> np.ndarray:
    """The M4 Naive2 benchmark forecast for YEARLY series.

    Naive2 is a random walk on seasonally adjusted data. Yearly series carry
    no within-year seasonality in the M4 protocol (m=1, and the official
    90% autocorrelation seasonality test is only applied for m > 1), so
    Naive2 reduces to Naive1: repeat the last observed training value.
    Confirmed by the official leaderboard, where Naive, sNaive and Naive2
    all score identically on Yearly (sMAPE 16.342, MASE 3.974).

    Args:
        train: Training values of the series.
        horizon: Number of steps to forecast.

    Returns:
        Array of `horizon` copies of the last training value.
    """
    return np.full(horizon, np.asarray(train, dtype=np.float64)[-1])
