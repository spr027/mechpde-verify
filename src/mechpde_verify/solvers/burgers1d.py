"""1D viscous Burgers equation solver (conservative finite-volume + RK2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Burgers1DResult:
    """Result container for 1D viscous Burgers equation."""

    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    dx: float
    dt: float
    nu: float


def viscous_burgers_traveling_wave(
    x: np.ndarray,
    t: float,
    *,
    nu: float = 0.05,
    x0: float = 0.25,
    c: float = 0.5,
) -> np.ndarray:
    """Exact traveling-wave solution to u_t + u u_x = nu u_xx.

    u(x, t) = c * (1 - tanh(c * (x - x0 - c * t) / (2 * nu)))
    """
    return c * (1.0 - np.tanh(c * (x - x0 - c * t) / (2.0 * nu)))


def _rhs_conservative(
    u: np.ndarray,
    dx: float,
    nu: float,
) -> np.ndarray:
    """Compute spatial RHS -(0.5 * u^2)_x + nu * u_xx on interior nodes."""
    flux = 0.5 * u * u
    adv = -(flux[2:] - flux[:-2]) / (2.0 * dx)
    diff = nu * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    return adv + diff


def solve_burgers_1d(
    nx: int,
    nt: int,
    *,
    length: float = 1.0,
    t_final: float = 0.2,
    nu: float = 0.05,
    u0: Callable[[np.ndarray], np.ndarray],
    bc_left: Callable[[float], float],
    bc_right: Callable[[float], float],
) -> Burgers1DResult:
    """Solve u_t + (0.5 * u^2)_x = nu * u_xx on [0, length] x [0, t_final] using Heun RK2."""
    if nx < 2:
        raise ValueError("nx must be >= 2")
    if nt < 1:
        raise ValueError("nt must be >= 1")
    if length <= 0.0 or t_final <= 0.0 or nu <= 0.0:
        raise ValueError("length, t_final, and nu must be positive")

    x = np.linspace(0.0, length, nx + 1)
    dx = length / nx
    dt = t_final / nt
    t = np.linspace(0.0, t_final, nt + 1)

    u = np.asarray(u0(x), dtype=np.float64).copy()
    u[0] = bc_left(0.0)
    u[-1] = bc_right(0.0)

    # Check diffusion stability bound for explicit RK2
    diff_cfl = nu * dt / (dx * dx)
    if diff_cfl > 0.5:
        raise ValueError(
            f"Explicit RK2 diffusion number nu*dt/dx^2 = {diff_cfl:.4f} > 0.5; "
            "increase nt or reduce t_final."
        )

    for n in range(nt):
        tn1 = t[n + 1]
        k1 = _rhs_conservative(u, dx, nu)

        u_star = u.copy()
        u_star[1:-1] = u[1:-1] + dt * k1
        u_star[0] = bc_left(tn1)
        u_star[-1] = bc_right(tn1)

        k2 = _rhs_conservative(u_star, dx, nu)
        u[1:-1] = u[1:-1] + 0.5 * dt * (k1 + k2)
        u[0] = bc_left(tn1)
        u[-1] = bc_right(tn1)

    return Burgers1DResult(x=x, t=t, u=u, dx=dx, dt=dt, nu=nu)
