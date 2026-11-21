"""Verification tools: Method of Manufactured Solutions (MMS), GCI, and PDE residuals."""

from mechpde_verify.verification.mms import (
    ConvergenceStudy,
    ErrorMetrics,
    compute_error_metrics,
    grid_convergence_index,
    run_convergence_study,
)
from mechpde_verify.verification.residuals import (
    discrete_heat_residual_1d,
    discrete_poisson_residual_2d,
)

__all__ = [
    "ConvergenceStudy",
    "ErrorMetrics",
    "compute_error_metrics",
    "discrete_heat_residual_1d",
    "discrete_poisson_residual_2d",
    "grid_convergence_index",
    "run_convergence_study",
]
