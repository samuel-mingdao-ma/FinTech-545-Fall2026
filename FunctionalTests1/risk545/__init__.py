"""Reusable numerical routines; this package never reads reference answers."""
from .covariance import missing_covariance, ew_covariance, correlation, mixed_ew_covariance
from .simulation import near_psd, higham_psd, chol_psd, simulate_normal, simulate_pca, pca_target
from .distributions import fit_normal, fit_t, fit_t_regression, fit_nig_moments, fit_nig_mle, aicc

__all__ = [
    "missing_covariance", "ew_covariance", "correlation", "mixed_ew_covariance",
    "near_psd", "higham_psd", "chol_psd", "simulate_normal", "simulate_pca", "pca_target",
    "fit_normal", "fit_t", "fit_t_regression", "fit_nig_moments", "fit_nig_mle", "aicc",
]
