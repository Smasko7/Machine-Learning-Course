"""
timeseries.py
-------------
Reusable time series utilities extracted from the ML course exercises.

Covers:
- Lag (sliding-window) feature matrix creation  (Exercise 3)
- Log-return transformation and its inverse      (Exercise 3)
- ADF stationarity test wrapper                  (Exercise 3)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lag feature creation
# ---------------------------------------------------------------------------

def create_lag_features(series, n_lags: int):
    """Build a supervised feature matrix from a univariate time series.

    Each row contains the previous *n_lags* values as features (X) and the
    next value as the target (y).

    Parameters
    ----------
    series : array-like or pd.Series of shape (T,)
    n_lags : number of past time steps to use as features

    Returns
    -------
    X : np.ndarray of shape (T - n_lags, n_lags)
    y : np.ndarray of shape (T - n_lags,)

    Example
    -------
    >>> X, y = create_lag_features(close_prices, n_lags=5)
    """
    series = np.array(series)
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Log-return transformation
# ---------------------------------------------------------------------------

def to_log_returns(series) -> np.ndarray:
    """Transform a price series into log returns: log(P_t / P_{t-1}).

    Log returns are approximately stationary and normally distributed,
    making them more suitable for regression models than raw prices.

    Parameters
    ----------
    series : array-like of shape (T,)

    Returns
    -------
    log_returns : np.ndarray of shape (T - 1,)
    """
    series = np.array(series, dtype=float)
    return np.log(series[1:] / series[:-1])


def from_log_returns(log_returns, initial_value: float) -> np.ndarray:
    """Reconstruct a price series from log returns and an initial price.

    Parameters
    ----------
    log_returns   : array-like of shape (T,)
    initial_value : the price at time 0 (the value before the first return)

    Returns
    -------
    prices : np.ndarray of shape (T + 1,) including initial_value at index 0
    """
    log_returns = np.array(log_returns, dtype=float)
    prices = np.empty(len(log_returns) + 1)
    prices[0] = initial_value
    prices[1:] = initial_value * np.exp(np.cumsum(log_returns))
    return prices


# ---------------------------------------------------------------------------
# Stationarity test
# ---------------------------------------------------------------------------

def adf_test(series, significance: float = 0.05) -> dict:
    """Run the Augmented Dickey-Fuller test and return a summary dict.

    Requires statsmodels: pip install statsmodels

    Parameters
    ----------
    series       : array-like time series
    significance : significance level for the stationarity decision

    Returns
    -------
    result : dict with keys
        "adf_statistic", "p_value", "n_lags_used",
        "critical_values", "is_stationary"
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        raise ImportError("statsmodels is required: pip install statsmodels")

    adf_stat, p_value, n_lags, _, crit_values, _ = adfuller(series, autolag="AIC")

    result = {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "n_lags_used": n_lags,
        "critical_values": crit_values,
        "is_stationary": p_value < significance,
    }

    print(f"ADF Statistic : {adf_stat:.4f}")
    print(f"p-value       : {p_value:.4f}")
    for level, val in crit_values.items():
        print(f"  Critical ({level}): {val:.4f}")
    verdict = "STATIONARY" if result["is_stationary"] else "NON-STATIONARY"
    print(f"=> Series is {verdict} at {significance:.0%} significance level")

    return result
