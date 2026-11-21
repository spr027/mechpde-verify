"""Canonical PDE solvers for computational mechanics benchmarks."""

from mechpde_verify.solvers.beam1d import Beam1DResult, solve_euler_bernoulli_beam
from mechpde_verify.solvers.burgers1d import Burgers1DResult, solve_burgers_1d
from mechpde_verify.solvers.heat1d import Heat1DResult, solve_heat_1d
from mechpde_verify.solvers.poisson2d import Poisson2DResult, solve_poisson_2d

__all__ = [
    "Beam1DResult",
    "Burgers1DResult",
    "Heat1DResult",
    "Poisson2DResult",
    "solve_burgers_1d",
    "solve_euler_bernoulli_beam",
    "solve_heat_1d",
    "solve_poisson_2d",
]
