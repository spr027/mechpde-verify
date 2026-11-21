"""mechpde-verify: Reproducible PDE solvers and numerical verification for computational mechanics."""

from mechpde_verify.solvers.beam1d import Beam1DResult, solve_euler_bernoulli_beam
from mechpde_verify.solvers.burgers1d import Burgers1DResult, solve_burgers_1d
from mechpde_verify.solvers.heat1d import Heat1DResult, solve_heat_1d
from mechpde_verify.solvers.poisson2d import Poisson2DResult, solve_poisson_2d
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

__version__ = "0.1.0"

__all__ = [
    "Beam1DResult",
    "Burgers1DResult",
    "ConvergenceStudy",
    "ErrorMetrics",
    "Heat1DResult",
    "Poisson2DResult",
    "compute_error_metrics",
    "discrete_heat_residual_1d",
    "discrete_poisson_residual_2d",
    "grid_convergence_index",
    "run_convergence_study",
    "solve_burgers_1d",
    "solve_euler_bernoulli_beam",
    "solve_heat_1d",
    "solve_poisson_2d",
]
