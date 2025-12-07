"""Noise Handling Module.

Provides robust regression and uncertainty quantification for
handling real-world noisy data with outliers.

Components:
    - robust_regression: Huber, RANSAC, IRLS methods
    - uncertainty: Confidence intervals, p-values, error propagation
"""

from .robust_regression import (
    huber_loss_regression,
    ransac_regression,
    iteratively_reweighted_lst_sq,
    detect_outliers,
    robust_fit,
)
from .uncertainty import (
    bootstrap_confidence_interval,
    prediction_interval,
    coefficient_significance,
    monte_carlo_uncertainty,
    format_with_uncertainty,
)

__all__ = [
    # Robust Regression
    'huber_loss_regression',
    'ransac_regression',
    'iteratively_reweighted_lst_sq',
    'detect_outliers',
    'robust_fit',
    
    # Uncertainty Quantification
    'bootstrap_confidence_interval',
    'prediction_interval',
    'coefficient_significance',
    'monte_carlo_uncertainty',
    'format_with_uncertainty',
]
