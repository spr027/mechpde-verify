"""Discrete physics-informed PDE residual evaluators for numerical and surrogate solutions."""

from __future__ import annotations

from typing import Callable

import numpy as np


def discrete_poisson_residual_2d(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    conductivity: float = 1.0,
    source: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Evaluate interior 5-point discrete residual R = -k*(u_xx + u_yy) - f(x, y)."""
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    xx, yy = np.meshgrid(x[1:-1], y[1:-1], indexing="ij")

    u_xx = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx * dx)
    u_yy = (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy * dy)
    return -conductivity * (u_xx + u_yy) - source(xx, yy)


def discrete_heat_residual_1d(
    u_prev: np.ndarray,
    u_curr: np.ndarray,
    x: np.ndarray,
    dt: float,
    *,
    alpha: float = 1.0,
    source_mid: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate Crank-Nicolson discrete residual on interior nodes between two time levels."""
    dx = float(x[1] - x[0])
    u_t = (u_curr[1:-1] - u_prev[1:-1]) / dt
    u_xx_prev = (u_prev[2:] - 2.0 * u_prev[1:-1] + u_prev[:-2]) / (dx * dx)
    u_xx_curr = (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2]) / (dx * dx)
    rhs = 0.5 * alpha * (u_xx_prev + u_xx_curr)
    if source_mid is not None:
        rhs = rhs + source_mid
    return u_t - rhs
