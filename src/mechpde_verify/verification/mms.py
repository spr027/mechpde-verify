"""Method of Manufactured Solutions (MMS) error metrics and Grid Convergence Index (GCI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ErrorMetrics:
    """Discrete L1, L2, and L-infinity error norms."""

    h: float
    l1: float
    l2: float
    linf: float


@dataclass(frozen=True)
class ConvergenceStudy:
    """Grid-refinement convergence report."""

    name: str
    formal_order: float
    metrics: tuple[ErrorMetrics, ...]
    observed_orders_l2: tuple[float, ...]
    observed_orders_linf: tuple[float, ...]
    gci_fine_percent: float

    @property
    def asymptotic_order_l2(self) -> float:
        """Observed L2 convergence rate on the finest mesh pair."""
        return self.observed_orders_l2[-1]

    @property
    def asymptotic_order_linf(self) -> float:
        """Observed L-infinity convergence rate on the finest mesh pair."""
        return self.observed_orders_linf[-1]


def compute_error_metrics(
    numerical: np.ndarray,
    exact: np.ndarray,
    *,
    cell_measure: float,
    h: float,
) -> ErrorMetrics:
    """Compute discrete L1, L2, and L-infinity norms weighted by cell measure."""
    diff = np.abs(np.asarray(numerical, dtype=np.float64) - np.asarray(exact, dtype=np.float64))
    l1 = float(cell_measure * np.sum(diff))
    l2 = float(np.sqrt(cell_measure * np.sum(diff * diff)))
    linf = float(np.max(diff))
    return ErrorMetrics(h=h, l1=l1, l2=l2, linf=linf)


def grid_convergence_index(
    error_coarse: float,
    error_fine: float,
    refinement_ratio: float,
    observed_order: float,
    *,
    safety_factor: float = 1.25,
) -> float:
    """Compute Roache's Grid Convergence Index (GCI) in percent.

    GCI = F_s * |e_coarse - e_fine| / (r^p - 1) * 100  (normalized to fine error scale).
    For MMS problems where exact error is known, we report the relative fine-grid
    discretization uncertainty band in percent.
    """
    if refinement_ratio <= 1.0:
        raise ValueError("refinement_ratio must be > 1.0")
    denom = (refinement_ratio**observed_order) - 1.0
    if denom <= 0.0:
        return float("inf")
    rel_change = abs(error_coarse - error_fine) / max(abs(error_fine), 1e-15)
    return float(safety_factor * rel_change / denom * 100.0)


def run_convergence_study(
    name: str,
    grid_sizes: Sequence[int],
    evaluator: Callable[[int], ErrorMetrics],
    *,
    formal_order: float = 2.0,
) -> ConvergenceStudy:
    """Run a multi-grid MMS convergence study and compute observed orders + GCI."""
    if len(grid_sizes) < 2:
        raise ValueError("grid_sizes must contain at least 2 resolutions")

    metrics = tuple(evaluator(n) for n in grid_sizes)
    orders_l2: list[float] = []
    orders_linf: list[float] = []

    for prev, curr in zip(metrics[:-1], metrics[1:]):
        r = prev.h / curr.h
        if r <= 1.0:
            raise ValueError("grid_sizes must produce strictly decreasing mesh spacing h")
        orders_l2.append(float(np.log(prev.l2 / curr.l2) / np.log(r)))
        orders_linf.append(float(np.log(prev.linf / curr.linf) / np.log(r)))

    r_fine = metrics[-2].h / metrics[-1].h
    gci_pct = grid_convergence_index(
        metrics[-2].l2,
        metrics[-1].l2,
        refinement_ratio=r_fine,
        observed_order=orders_l2[-1],
    )

    return ConvergenceStudy(
        name=name,
        formal_order=formal_order,
        metrics=metrics,
        observed_orders_l2=tuple(orders_l2),
        observed_orders_linf=tuple(orders_linf),
        gci_fine_percent=gci_pct,
    )
