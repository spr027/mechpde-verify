"""1D transient heat conduction solver (FTCS, Backward Euler, Crank-Nicolson)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.linalg import solve_banded

Scheme1D = Literal["ftcs", "backward_euler", "crank_nicolson"]


@dataclass(frozen=True)
class Heat1DResult:
    """Result container for 1D transient heat conduction."""

    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    dx: float
    dt: float
    fourier_number: float
    scheme: Scheme1D


def solve_heat_1d(
    nx: int,
    nt: int,
    *,
    length: float = 1.0,
    t_final: float = 0.2,
    alpha: float = 1.0,
    u0: Callable[[np.ndarray], np.ndarray],
    source: Callable[[np.ndarray, float], np.ndarray] | None = None,
    bc_left: Callable[[float], float] | None = None,
    bc_right: Callable[[float], float] | None = None,
    scheme: Scheme1D = "crank_nicolson",
) -> Heat1DResult:
    """Solve u_t = alpha * u_xx + source(x, t) on [0, length] x [0, t_final].

    Parameters
    ----------
    nx:
        Number of spatial intervals (nx + 1 grid nodes). Must be >= 2.
    nt:
        Number of time steps. Must be >= 1.
    length:
        Domain length L > 0.
    t_final:
        Final simulation time T > 0.
    alpha:
        Thermal diffusivity > 0.
    u0:
        Initial condition callable u(x, 0).
    source:
        Optional volumetric heat source f(x, t).
    bc_left, bc_right:
        Optional Dirichlet boundary callables g_left(t), g_right(t). Defaults to 0.
    scheme:
        Time-integration scheme: ``"ftcs"``, ``"backward_euler"``, or ``"crank_nicolson"``.
    """
    if nx < 2:
        raise ValueError("nx must be >= 2")
    if nt < 1:
        raise ValueError("nt must be >= 1")
    if length <= 0.0 or t_final <= 0.0 or alpha <= 0.0:
        raise ValueError("length, t_final, and alpha must be positive")

    x = np.linspace(0.0, length, nx + 1)
    dx = length / nx
    dt = t_final / nt
    t = np.linspace(0.0, t_final, nt + 1)
    r = alpha * dt / (dx * dx)

    if scheme == "ftcs" and r > 0.5 + 1e-12:
        raise ValueError(
            f"FTCS scheme is unstable for Fourier number r={r:.4f} > 0.5 "
            f"(dx={dx:.4g}, dt={dt:.4g}, alpha={alpha:.4g})."
        )

    g_left = bc_left if bc_left is not None else (lambda _t: 0.0)
    g_right = bc_right if bc_right is not None else (lambda _t: 0.0)
    f_src = source if source is not None else (lambda arr, _t: np.zeros_like(arr))

    u = np.asarray(u0(x), dtype=np.float64).copy()
    u[0] = g_left(0.0)
    u[-1] = g_right(0.0)

    m = nx - 1  # interior nodes
    x_int = x[1:-1]

    if scheme == "ftcs":
        for n in range(nt):
            tn = t[n]
            tn1 = t[n + 1]
            u_next = u.copy()
            u_next[1:-1] = (
                u[1:-1]
                + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
                + dt * f_src(x_int, tn)
            )
            u_next[0] = g_left(tn1)
            u_next[-1] = g_right(tn1)
            u = u_next
    elif scheme == "backward_euler":
        ab = np.zeros((3, m), dtype=np.float64)
        ab[0, 1:] = -r
        ab[1, :] = 1.0 + 2.0 * r
        ab[2, :-1] = -r
        for n in range(nt):
            tn1 = t[n + 1]
            gl1 = g_left(tn1)
            gr1 = g_right(tn1)
            rhs = u[1:-1] + dt * f_src(x_int, tn1)
            rhs[0] += r * gl1
            rhs[-1] += r * gr1
            u[1:-1] = solve_banded((1, 1), ab, rhs)
            u[0] = gl1
            u[-1] = gr1
    elif scheme == "crank_nicolson":
        half_r = 0.5 * r
        ab = np.zeros((3, m), dtype=np.float64)
        ab[0, 1:] = -half_r
        ab[1, :] = 1.0 + r
        ab[2, :-1] = -half_r
        for n in range(nt):
            tn = t[n]
            tn1 = t[n + 1]
            gl1 = g_left(tn1)
            gr1 = g_right(tn1)
            rhs = (
                (1.0 - r) * u[1:-1]
                + half_r * (u[:-2] + u[2:])
                + 0.5 * dt * (f_src(x_int, tn) + f_src(x_int, tn1))
            )
            rhs[0] += half_r * gl1
            rhs[-1] += half_r * gr1
            u[1:-1] = solve_banded((1, 1), ab, rhs)
            u[0] = gl1
            u[-1] = gr1
    else:
        raise ValueError(f"Unsupported scheme: {scheme!r}")

    return Heat1DResult(
        x=x,
        t=t,
        u=u,
        dx=dx,
        dt=dt,
        fourier_number=r,
        scheme=scheme,
    )
