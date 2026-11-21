"""1D Euler-Bernoulli static beam bending solver (fourth-order structural PDE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.linalg import solve_banded

BeamBC = Literal["simply_supported"]


@dataclass(frozen=True)
class Beam1DResult:
    """Result container for 1D Euler-Bernoulli static beam deflection."""

    x: np.ndarray
    w: np.ndarray
    moment: np.ndarray
    dx: float
    ei: float


def solve_euler_bernoulli_beam(
    nx: int,
    *,
    length: float = 1.0,
    ei: float = 1.0,
    load: Callable[[np.ndarray], np.ndarray],
    bc: BeamBC = "simply_supported",
) -> Beam1DResult:
    """Solve EI * d^4w/dx^4 = q(x) on [0, length] by coupled second-order decomposition.

    For a simply supported beam (w(0) = w(L) = 0, M(0) = M(L) = 0), the fourth-order
    equation decomposes cleanly into two second-order Dirichlet boundary-value problems:
        1. -d^2 M / dx^2 = q(x),        M(0) = M(L) = 0
        2. -EI * d^2 w / dx^2 = M(x),   w(0) = w(L) = 0

    This preserves O(dx^2) accuracy with optimal O(nx) condition-number scaling.
    """
    if nx < 2:
        raise ValueError("nx must be >= 2")
    if length <= 0.0 or ei <= 0.0:
        raise ValueError("length and ei must be positive")
    if bc != "simply_supported":
        raise ValueError(f"Unsupported boundary condition: {bc!r}")

    x = np.linspace(0.0, length, nx + 1)
    dx = length / nx
    m = nx - 1
    inv_dx2 = 1.0 / (dx * dx)

    ab = np.zeros((3, m), dtype=np.float64)
    ab[0, 1:] = -inv_dx2
    ab[1, :] = 2.0 * inv_dx2
    ab[2, :-1] = -inv_dx2

    q_int = np.asarray(load(x[1:-1]), dtype=np.float64)
    moment = np.zeros(nx + 1, dtype=np.float64)
    moment[1:-1] = solve_banded((1, 1), ab, q_int)

    w = np.zeros(nx + 1, dtype=np.float64)
    w[1:-1] = solve_banded((1, 1), ab, moment[1:-1] / ei)

    return Beam1DResult(x=x, w=w, moment=moment, dx=dx, ei=ei)
