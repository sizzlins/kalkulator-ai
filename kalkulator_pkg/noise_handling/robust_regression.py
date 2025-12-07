"""Robust regression methods for noisy data.

Implements outlier-resistant regression algorithms:
- Huber regression: Uses Huber loss which is less sensitive to outliers
- RANSAC: Random Sample Consensus for extreme outlier scenarios
- M-estimators: General iteratively reweighted least squares

These methods are critical for handling real-world experimental data
where measurement errors and outliers are common.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy import stats
from sklearn.linear_model import HuberRegressor, RANSACRegressor, LinearRegression


def huber_loss_regression(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1.35,
    max_iter: int = 1000,
    fit_intercept: bool = True
) -> tuple[np.ndarray, float, dict]:
    """Huber regression: robust to moderate outliers.
    
    The Huber loss function is quadratic for small residuals and linear for
    large residuals, providing a balance between MSE and absolute error.
    
    Args:
        X: Input features of shape (n_samples, n_features)
        y: Target values of shape (n_samples,)
        epsilon: Threshold below which loss is quadratic (default: 1.35)
        max_iter: Maximum iterations for convergence
        fit_intercept: Whether to fit an intercept term
        
    Returns:
        Tuple of:
        - coefficients: Array of shape (n_features,)
        - intercept: Float intercept value
        - info: Dict with additional information (outlier mask, etc.)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        model = HuberRegressor(
            epsilon=epsilon,
            max_iter=max_iter,
            fit_intercept=fit_intercept,
        )
        model.fit(X, y)
    
    # Identify outliers based on residuals
    predictions = model.predict(X)
    residuals = np.abs(y - predictions)
    threshold = epsilon * np.std(residuals)
    outlier_mask = residuals > threshold
    
    info = {
        'outlier_mask': outlier_mask,
        'n_outliers': np.sum(outlier_mask),
        'residual_std': np.std(residuals),
        'residuals': residuals,
    }
    
    return model.coef_, model.intercept_, info


def ransac_regression(
    X: np.ndarray,
    y: np.ndarray,
    min_samples: int | float = 0.5,
    residual_threshold: float | None = None,
    max_trials: int = 100,
    fit_intercept: bool = True
) -> tuple[np.ndarray, float, dict]:
    """RANSAC regression: robust to extreme outliers.
    
    Random Sample Consensus iteratively fits models on random subsets
    of data and finds the model with maximum inlier consensus.
    
    Args:
        X: Input features
        y: Target values
        min_samples: Minimum number of samples for fitting
        residual_threshold: Maximum residual for inliers (default: MAD-based)
        max_trials: Maximum iterations
        fit_intercept: Whether to fit intercept
        
    Returns:
        Tuple of (coefficients, intercept, info)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Default threshold based on median absolute deviation
    if residual_threshold is None:
        # First fit a rough model
        lr = LinearRegression(fit_intercept=fit_intercept)
        lr.fit(X, y)
        residuals = np.abs(y - lr.predict(X))
        # MAD-based threshold
        mad = np.median(residuals)
        residual_threshold = 3.0 * mad if mad > 0 else np.std(residuals)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        model = RANSACRegressor(
            estimator=LinearRegression(fit_intercept=fit_intercept),
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )
        model.fit(X, y)
    
    # Get inlier/outlier mask
    inlier_mask = model.inlier_mask_
    
    info = {
        'inlier_mask': inlier_mask,
        'outlier_mask': ~inlier_mask,
        'n_outliers': np.sum(~inlier_mask),
        'n_inliers': np.sum(inlier_mask),
        'residual_threshold': residual_threshold,
    }
    
    return model.estimator_.coef_, model.estimator_.intercept_, info


def iteratively_reweighted_lst_sq(
    X: np.ndarray,
    y: np.ndarray,
    loss: str = 'huber',
    max_iter: int = 50,
    tol: float = 1e-6,
    fit_intercept: bool = True
) -> tuple[np.ndarray, float, dict]:
    """Iteratively Reweighted Least Squares (IRLS) for M-estimators.
    
    General framework for robust regression using different loss functions.
    
    Args:
        X: Input features
        y: Target values
        loss: Loss function ('huber', 'tukey', 'cauchy')
        max_iter: Maximum iterations
        tol: Convergence tolerance
        fit_intercept: Whether to fit intercept
        
    Returns:
        Tuple of (coefficients, intercept, info)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # Add intercept column if needed
    if fit_intercept:
        X_aug = np.column_stack([np.ones(n_samples), X])
    else:
        X_aug = X
    
    # Weight functions for different loss types
    def huber_weights(r, c=1.345):
        """Huber weight function."""
        return np.where(np.abs(r) <= c, 1.0, c / np.abs(r + 1e-10))
    
    def tukey_weights(r, c=4.685):
        """Tukey's biweight function."""
        return np.where(np.abs(r) <= c, (1 - (r/c)**2)**2, 0.0)
    
    def cauchy_weights(r, c=2.385):
        """Cauchy weight function."""
        return 1.0 / (1 + (r/c)**2)
    
    weight_funcs = {
        'huber': huber_weights,
        'tukey': tukey_weights,
        'cauchy': cauchy_weights,
    }
    
    weight_func = weight_funcs.get(loss, huber_weights)
    
    # Initialize with OLS
    try:
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    except Exception:
        beta = np.zeros(X_aug.shape[1])
    
    # IRLS iterations
    for iteration in range(max_iter):
        # Calculate residuals
        residuals = y - X_aug @ beta
        
        # Scale estimate (MAD)
        scale = np.median(np.abs(residuals - np.median(residuals))) / 0.6745
        if scale < 1e-10:
            scale = 1.0
        
        # Standardized residuals
        r_std = residuals / scale
        
        # Calculate weights
        weights = weight_func(r_std)
        weights = np.clip(weights, 1e-6, 1.0)
        
        # Weighted least squares
        W = np.diag(weights)
        try:
            XtWX = X_aug.T @ W @ X_aug
            XtWy = X_aug.T @ W @ y
            beta_new = np.linalg.solve(XtWX + 1e-6 * np.eye(XtWX.shape[0]), XtWy)
        except Exception:
            break
        
        # Check convergence
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        
        beta = beta_new
    
    # Extract intercept and coefficients
    if fit_intercept:
        intercept = beta[0]
        coef = beta[1:]
    else:
        intercept = 0.0
        coef = beta
    
    # Final residuals and weights
    final_residuals = y - X_aug @ beta
    scale = np.median(np.abs(final_residuals - np.median(final_residuals))) / 0.6745
    if scale < 1e-10:
        scale = 1.0
    final_weights = weight_func(final_residuals / scale)
    
    info = {
        'weights': final_weights,
        'outlier_mask': final_weights < 0.5,
        'n_outliers': np.sum(final_weights < 0.5),
        'residuals': final_residuals,
        'iterations': iteration + 1,
        'scale': scale,
    }
    
    return coef, intercept, info


def detect_outliers(
    y: np.ndarray,
    y_pred: np.ndarray,
    method: str = 'mad'
) -> np.ndarray:
    """Detect outliers in residuals.
    
    Args:
        y: True values
        y_pred: Predicted values
        method: Detection method ('mad', 'iqr', 'zscore')
        
    Returns:
        Boolean mask where True indicates outlier
    """
    residuals = y - y_pred
    
    if method == 'mad':
        # Median Absolute Deviation
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        threshold = 3.5 * mad / 0.6745  # Normalized MAD
        outliers = np.abs(residuals - med) > threshold
        
    elif method == 'iqr':
        # Interquartile Range
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = (residuals < lower) | (residuals > upper)
        
    elif method == 'zscore':
        # Z-score based
        z_scores = np.abs(stats.zscore(residuals))
        outliers = z_scores > 3.0
        
    else:
        outliers = np.zeros(len(residuals), dtype=bool)
    
    return outliers


def robust_fit(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'auto',
    **kwargs
) -> tuple[np.ndarray, float, dict]:
    """Automatically choose and apply robust regression.
    
    Args:
        X: Input features
        y: Target values
        method: 'auto', 'huber', 'ransac', or 'irls'
        **kwargs: Additional arguments for specific method
        
    Returns:
        Tuple of (coefficients, intercept, info)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if method == 'auto':
        # First try regular OLS to check residual distribution
        lr = LinearRegression()
        lr.fit(X, y)
        residuals = y - lr.predict(X)
        
        # Check for outliers using MAD
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        if mad > 0:
            outlier_ratio = np.mean(np.abs(residuals - med) > 3.5 * mad / 0.6745)
        else:
            outlier_ratio = 0
        
        # Choose method based on outlier ratio
        if outlier_ratio > 0.2:
            method = 'ransac'  # High outlier ratio
        elif outlier_ratio > 0.05:
            method = 'huber'   # Moderate outliers
        else:
            # Low outliers, use regular OLS
            return lr.coef_, lr.intercept_, {'method': 'ols', 'n_outliers': 0}
    
    if method == 'huber':
        return huber_loss_regression(X, y, **kwargs)
    elif method == 'ransac':
        return ransac_regression(X, y, **kwargs)
    elif method == 'irls':
        return iteratively_reweighted_lst_sq(X, y, **kwargs)
    else:
        # Fallback to Huber
        return huber_loss_regression(X, y, **kwargs)
