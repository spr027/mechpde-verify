"""2D steady-state Poisson / heat conduction solver via sparse finite differences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Poisson2DResult:
    """Result container for 2D steady-state elliptic Poisson problem."""

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    dx: float
    dy: float
    residual_l2: float


def solve_poisson_2d(
    nx: int,
    ny: int | None = None,
    *,
    lx: float = 1.0,
    ly: float = 1.0,
    conductivity: float = 1.0,
    source: Callable[[np.ndarray, np.ndarray], np.ndarray],
    bc: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Poisson2DResult:
    """Solve -k * (u_xx + u_yy) = source(x, y) on [0, lx] x [0, ly] with Dirichlet BCs.

    Uses a 5-point central finite-difference / finite-volume stencil assembled as a
    sparse CSR linear system and solved directly via ``scipy.sparse.linalg.spsolve``.
    """
    ny = nx if ny is None else ny
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2")
    if lx <= 0.0 or ly <= 0.0 or conductivity <= 0.0:
        raise ValueError("lx, ly, and conductivity must be positive")

    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    dx = lx / nx
    dy = ly / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")

    g_bc = bc if bc is not None else (lambda a, _b: np.zeros_like(a))
    u = np.zeros((nx + 1, ny + 1), dtype=np.float64)

    # Populate Dirichlet boundary values
    u[0, :] = g_bc(xx[0, :], yy[0, :])
    u[-1, :] = g_bc(xx[-1, :], yy[-1, :])
    u[:, 0] = g_bc(xx[:, 0], yy[:, 0])
    u[:, -1] = g_bc(xx[:, -1], yy[:, -1])

    mx = nx - 1
    my = ny - 1
    inv_dx2 = conductivity / (dx * dx)
    inv_dy2 = conductivity / (dy * dy)

    # 1D Laplacian operators on interior nodes
    tx = sparse.diags(
        [-inv_dx2, 2.0 * inv_dx2, -inv_dx2],
        offsets=[-1, 0, 1],
        shape=(mx, mx),
        format="csr",
    )
    ty = sparse.diags(
        [-inv_dy2, 2.0 * inv_dy2, -inv_dy2],
        offsets=[-1, 0, 1],
        shape=(my, my),
        format="csr",
    )
    a_mat = sparse.kronsum(ty, tx, format="csr")

    rhs = np.asarray(source(xx[1:-1, 1:-1], yy[1:-1, 1:-1]), dtype=np.float64).copy()
    rhs[0, :] += inv_dx2 * u[0, 1:-1]
    rhs[-1, :] += inv_dx2 * u[-1, 1:-1]
    rhs[:, 0] += inv_dy2 * u[1:-1, 0]
    rhs[:, -1] += inv_dy2 * u[1:-1, -1]

    rhs_flat = rhs.reshape(-1)
    sol_flat = spsolve(a_mat, rhs_flat)
    u[1:-1, 1:-1] = sol_flat.reshape((mx, my))

    linear_residual = a_mat @ sol_flat - rhs_flat
    residual_l2 = float(np.sqrt(dx * dy * np.sum(linear_residual**2)))

    return Poisson2DResult(
        x=x,
        y=y,
        u=u,
        dx=dx,
        dy=dy,
        residual_l2=residual_l2,
    )
