"""
Utility functions for 1d linear regression
"""

import numpy as np


def empirical_risk(
    theta: float,
    b: float,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Compute the empirical risk of a set of parameters
    for a linear prediction, in 1 dimension.

    fix this function
    """
    y_pred = X * theta +b
    error = (y_pred -y)
    return np.mean(error**2)


def compute_optimal_params(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Compute the optimal theta and b, obtained by
    gradient cancellation
    """
    x_bar = np.mean(X)
    y_bar = np.mean(y)

    numerator = np.sum((X - x_bar)*(y - y_bar))
    denominator = np.sum((X - x_bar)**2)

    theta_star = numerator / denominator

    b_star = y_bar - theta_star * x_bar
    return theta_star, b_star
